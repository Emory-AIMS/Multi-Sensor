# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer, ConvNet, Flatten
from torch import nn
from transformers.modeling_bert import (
    ACT2FN,
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)


logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask

def _get_cos_sim(a, b):
    # reshape tensor and get cosine similarity
    a = torch.reshape(a, (-1,))
    b = torch.reshape(b, (-1,))
    cos_sim = torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))
    return cos_sim

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ACT2FN["gelu"](x)

class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_PRED_NUM = config.max_pred_num
        hidden_size = config.hidden_size
        dropout_prob = config.hidden_dropout_prob

        self.pred_type_embeddings = nn.Embedding(MAX_PRED_NUM+1, hidden_size)

        self.emb_layer_norm = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(dropout_prob)

    def forward(self, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.int64
        pred_type_embeddings = self.pred_type_embeddings(prev_inds)
        embeddings = self.emb_layer_norm(pred_type_embeddings)
        embeddings = self.emb_dropout(embeddings)

        return embeddings

class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self, emb, mask):
        encoder_inputs = emb
        attention_mask = mask
        dec_max_num = self.config.max_dec_length

        # mask shape: [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        if self.config.model_type == "realtime":
            causal_mask = _get_causal_mask(dec_max_num, encoder_inputs.device)

            # 1. decoding step elements can attend to themselves in a causal manner
            # if self.config.training_head_type == "classification":
            extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = causal_mask

            # 2. mask the cls token
            extended_attention_mask[:, :, 0, :] = 0
            extended_attention_mask[:, :, :, 0] = 0

            # 3. create complete causal mask for real time prediction, 6 = len(["temp2", "hr2", "magnitude", "yaxis", "demographic"]) + 1
            for i in range(1, 6):
                extended_attention_mask[:, :, -dec_max_num:, -(i+1)*dec_max_num:-i*dec_max_num] = causal_mask
                for j in range(1, 6):
                    extended_attention_mask[:, :, -(j+1)*dec_max_num:-j*dec_max_num, -(i+1)*dec_max_num:-i*dec_max_num] = causal_mask

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        seq_output = sequence_output[:, 1:4*dec_max_num+1] # number of PT1 keys
        dec_output = sequence_output[:, -dec_max_num:]

        results = {
            "mmt_pooled_output": pooled_output,
            "mmt_seq_output": seq_output,
            "mmt_dec_output": dec_output
        }
        return results

class MultiSensorBase(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mmt_config = BertConfig(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")
        self.keys = ["temp2", "hr2", "magnitude", "yaxis", "demographic"]

        # GRU modules for each modality
        self.gru_temp = nn.GRU(input_size=self.config.data_in_dim.double_vital, **self.config.gru)
        self.gru_hr = nn.GRU(input_size=self.config.data_in_dim.double_vital, **self.config.gru)
        self.gru_magnitude = nn.GRU(input_size=self.config.data_in_dim.single_vital, **self.config.gru)
        self.gru_yaxis = nn.GRU(input_size=self.config.data_in_dim.single_vital, **self.config.gru)
        self.gru_modules = [self.gru_temp, self.gru_hr, self.gru_magnitude, self.gru_yaxis]

        # layer norm modules for each modality
        self.ln_temp = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ln_hr = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ln_magnitude = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ln_yaxis = nn.LayerNorm(self.mmt_config.hidden_size)
        self.layr_norm_modules = [self.ln_temp, self.ln_hr, self.ln_magnitude, self.ln_yaxis]

        # drop out modules for each modality
        self.drop_temp = nn.Dropout(self.mmt_config.hidden_dropout_prob)
        self.drop_hr = nn.Dropout(self.mmt_config.hidden_dropout_prob)
        self.drop_magnitude = nn.Dropout(self.mmt_config.hidden_dropout_prob)
        self.drop_yaxis = nn.Dropout(self.mmt_config.hidden_dropout_prob)
        self.dropout_modules = [self.drop_temp, self.drop_hr, self.drop_magnitude, self.drop_yaxis]

        # linear transform for demographic
        self.linear_dem = nn.Linear(self.config.data_in_dim.demographic, self.mmt_config.hidden_size)
        self.ln_dem = nn.LayerNorm(self.mmt_config.hidden_size)
        self.drop_dem = nn.Dropout(self.mmt_config.hidden_dropout_prob)

        # GRU transform for concat information, for decoding query
        cat_in_dim = 2 * (self.config.data_in_dim.single_vital + self.config.data_in_dim.double_vital) + self.config.data_in_dim.demographic
        self.gru_cat = nn.GRU(input_size=cat_in_dim, **self.config.gru)
        self.ln_cat = nn.LayerNorm(self.mmt_config.hidden_size)
        self.drop_cat = nn.Dropout(self.mmt_config.hidden_dropout_prob)

        # shared positional embeddings
        self.position_embeddings = nn.Embedding(self.mmt_config.max_dec_length, self.mmt_config.hidden_size)
        self.ln_pos = nn.LayerNorm(self.mmt_config.hidden_size)

        self.mmt = MMT(self.mmt_config)

    def forward(self, sample_list):
        data_mask = sample_list.data_mask
        batch_size = data_mask.size(0)
        seq_length = data_mask.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=data_mask.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        pos_emb = self.position_embeddings(position_ids)        

        emb_list = [sample_list.cls_token]
        mask_list = [sample_list.cls_mask]
        for key, gru_module, layr_norm_module, dropout_module in zip(self.keys, self.gru_modules, self.layr_norm_modules, self.dropout_modules):
            gru_module.flatten_parameters()
            key_emb, _ = gru_module(sample_list[key])
            final_emb = layr_norm_module(key_emb) + self.ln_pos(pos_emb)
            final_emb = dropout_module(final_emb)
            emb_list.append(final_emb)
            mask_list.append(data_mask)

        emb_list.append(self.drop_dem(self.ln_dem(self.linear_dem(sample_list.demographic))))
        mask_list.append(data_mask)

        cat_list = []
        for key in self.keys:
            cat_list.append(sample_list[key])
        cat_gru_emb, _ = self.gru_cat(torch.cat(cat_list, dim=-1))
        cat_emb = self.ln_cat(cat_gru_emb) + self.ln_pos(pos_emb)
        cat_emb = self.drop_cat(cat_emb)
        cat_mask = data_mask

        if self.config.model_type == "realtime":
            cat_mask = torch.zeros_like(cat_mask, dtype=torch.float32, device=cat_emb.device)
        if self.config.training_head_type == "pretraining":
            cat_emb = torch.zeros_like(cat_emb, dtype=torch.float32, device=cat_emb.device)
            cat_mask = torch.zeros_like(cat_mask, dtype=torch.float32, device=cat_emb.device)

        emb = torch.cat(emb_list + [cat_emb], dim=1)
        mask = torch.cat(mask_list + [cat_mask], dim=1)

        if self.config.training_head_type == "pretraining":
            # PT1: replace embedding with mask_token
            if self.config.use_pt1:
                pt1_keep_masks = torch.cat([torch.ones(emb.size(0), 1, device=emb.device), 
                    sample_list.pt1_keep_masks, torch.ones(emb.size(0), emb.size(1)-1-sample_list.pt1_keep_masks.size(1), device=emb.device)], dim=-1)
                pt1_keep_masks = pt1_keep_masks.unsqueeze(-1)
                emb = emb * pt1_keep_masks + sample_list.mask_token * (1-pt1_keep_masks)
            # PT3: input augmentation with random span masking (p=0.1)
            if self.config.use_pt3:
                emb_bar = emb[0].clone()
                mask_bar = mask[0].clone()

                pt3_length = mask_bar.size(0)-seq_length
                index = torch.randint(1, pt3_length, (int(pt3_length * 0.1),))
                for idx in index:
                    emb_bar[idx] = sample_list.mask_token[0]

                # append augmented data at the beginning of the mini-batch
                emb = torch.cat([emb_bar.unsqueeze(0), emb], dim=0)
                mask = torch.cat([mask_bar.unsqueeze(0), mask], dim=0)

        mmt_results = self.mmt(
            emb=emb,
            mask=mask
        )

        return mmt_results

class MultiSensorForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mmt_model = MultiSensorBase(self.config)

        if self.config.use_pt1:
            self.pt1_regressor0 = nn.Sequential(
                nn.Linear(self.config.mmt.hidden_size, self.config.mmt.hidden_size * 2),
                GeLU(),
                nn.LayerNorm(self.config.mmt.hidden_size * 2, eps=1e-12),
                nn.Linear(self.config.mmt.hidden_size * 2, self.config.data_in_dim.single_vital),
            )
            self.pt1_regressor1 = nn.Sequential(
                nn.Linear(self.config.mmt.hidden_size, self.config.mmt.hidden_size * 2),
                GeLU(),
                nn.LayerNorm(self.config.mmt.hidden_size * 2, eps=1e-12),
                nn.Linear(self.config.mmt.hidden_size * 2, self.config.data_in_dim.single_vital),
            )
            self.pt1_regressor2 = nn.Sequential(
                nn.Linear(self.config.mmt.hidden_size, self.config.mmt.hidden_size * 2),
                GeLU(),
                nn.LayerNorm(self.config.mmt.hidden_size * 2, eps=1e-12),
                nn.Linear(self.config.mmt.hidden_size * 2, self.config.data_in_dim.single_vital),
            )
            self.pt1_regressor3 = nn.Sequential(
                nn.Linear(self.config.mmt.hidden_size, self.config.mmt.hidden_size * 2),
                GeLU(),
                nn.LayerNorm(self.config.mmt.hidden_size * 2, eps=1e-12),
                nn.Linear(self.config.mmt.hidden_size * 2, self.config.data_in_dim.single_vital),
            )
            self.pt1_regressors = [self.pt1_regressor0, self.pt1_regressor1, self.pt1_regressor2, self.pt1_regressor3]

        if self.config.use_pt2:
            classifier_config = deepcopy(self.config.pt2_classifier)
            self.pt2_classifier = ClassifierLayer(
                classifier_config.type, **classifier_config.params
            )

    def forward(self, sample_list):
        output = {}

        mmt_results = self.mmt_model(sample_list)
        mmt_seq_output = mmt_results["mmt_seq_output"]
        mmt_pooled_output = mmt_results["mmt_pooled_output"]

        # Pretraining task 1
        if self.config.use_pt1:
            pt1_loss_func = torch.nn.MSELoss(reduction="none")
            pt1_loss = 0
            dec_max_num = self.config.mmt.max_dec_length

            for i in range(len(self.pt1_regressors)):
                pt1_pred_value = self.pt1_regressors[i](mmt_seq_output[:, i*dec_max_num : (i+1)*dec_max_num])
                pt1_loss += pt1_loss_func(pt1_pred_value, sample_list.pt1_labels[:, i*dec_max_num : (i+1)*dec_max_num]) * sample_list.pt1_masks[:, i*dec_max_num : (i+1)*dec_max_num].unsqueeze(-1)

            output["pt1_loss"] = pt1_loss

        # Pretraining task 2
        if self.config.use_pt2:
            pt2_score = self.pt2_classifier(mmt_pooled_output)
            pt2_loss = F.binary_cross_entropy_with_logits(pt2_score, sample_list.pt2_target, reduction="mean")
            
            output["pt2_loss"] = pt2_loss * pt2_score.size(1)

        # Pretraining task 3
        if self.config.use_pt3:
            # fetch augmented data and calculate similarities between positive pairs
            x = mmt_seq_output[1]
            x_bar = mmt_seq_output[0]
            pos_sim = _get_cos_sim(x, x_bar).unsqueeze(-1)

            # similarities between negative pairs
            neg_sim = []
            other_idx = [i for i in range(1, mmt_seq_output.size(0))]
            for idx in other_idx:
                x_neg = mmt_seq_output[idx]
                neg_sim.append(_get_cos_sim(x, x_neg).unsqueeze(-1))

            # InfoNCE loss to cluster positive pairs and push away negative pairs
            t = 0.1 # temperature parameter, smaller penalty:0.07, larger penalty:0.1
            logits = torch.cat([pos_sim]+neg_sim, dim=0)/t
            exp = torch.exp(logits)
            pt3_loss = - torch.log(exp[0]/torch.sum(exp[1:]))

            output["pt3_loss"] = pt3_loss

        return output

class MultiSensorForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mmt_model = MultiSensorBase(self.config)

        # momentum knowledge distillation (KD) teacher model
        self.mmt_model_m = MultiSensorBase(self.config) 
        


        classifier_config = deepcopy(self.config.classifier)
        if self.config.model_type == "hybrid": # a hybrid of onetime and realtime tasks present in the model
           self.classifier_aki = ClassifierLayer(
               classifier_config.type, **classifier_config.params
           )
           self.classifier_usg = ClassifierLayer(
               classifier_config.type, **classifier_config.params
           )  
           self.classifier_temp = ClassifierLayer(
               classifier_config.type, **classifier_config.params
           ) 
        else:
           self.classifier = ClassifierLayer(
               classifier_config.type, **classifier_config.params
           )

        # classifier teacher model for momentum KD
        self.classifier_m = ClassifierLayer(classifier_config.type, **classifier_config.params) 

        self.model_pairs = [[self.mmt_model, self.mmt_model_m],
                            [self.classifier, self.classifier_m]]
        self.copy_params()
        self.momentum = 0.95

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient  


    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                

    def forward(self, sample_list):
        output = {}
        
        if self.config.model_type == "hybrid": # both realtime and onetime tasks exist
            mmt_results_pooled = self.mmt_model(sample_list)["mmt_pooled_output"]
            mmt_results_dec = self.mmt_model(sample_list)["mmt_dec_output"]
            scores_aki = self.classifier_aki(mmt_results_pooled)
            scores_usg = self.classifier_usg(mmt_results_pooled)
            scores_temp = self.classifier_temp(mmt_results_dec)
            output["scores_aki"] = scores_aki
            output["scores_usg"] = scores_usg
            output["scores_temp"] = scores_temp
        elif self.config.model_type == "onetime":
            mmt_results = self.mmt_model(sample_list)["mmt_pooled_output"]
            scores = self.classifier(mmt_results)
            output["scores"] = scores
        else:
            mmt_results = self.mmt_model(sample_list)["mmt_dec_output"]
            scores = self.classifier(mmt_results)
            output["scores"] = scores
        

        # momentum KD 
        if self.config.model_type == "onetime": # aki and usg
            with torch.no_grad():
                self._momentum_update()
                mmt_results_m = self.mmt_model_m(sample_list)["mmt_pooled_output"]
                scores_m = self.classifier_m(mmt_results_m)
                output["scores_m"] = scores_m

        

        return output

@registry.register_model("multi_sensor_pt2")
class MultiSensorPT2(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/multi_sensor_pt2/defaults.yaml"

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = MultiSensorForPretraining(self.config)
        else:
            self.model = MultiSensorForClassification(self.config)

    def forward(self, sample_list):
        if self.config.training_head_type == "pretraining":
            output_dict = self.model(sample_list)
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            if "pt1_loss" in output_dict.keys():
                output_dict["losses"][loss_key + "/pt1_loss"] = output_dict.pop(
                    "pt1_loss"
                )
            if "pt2_loss" in output_dict.keys():
                output_dict["losses"][loss_key + "/pt2_loss"] = output_dict.pop(
                    "pt2_loss"
                )
            if "pt3_loss" in output_dict.keys():
                output_dict["losses"][loss_key + "/pt3_loss"] = output_dict.pop(
                    "pt3_loss"
                )
        else:
            output_dict = self.model(sample_list)
        
       
        return output_dict

