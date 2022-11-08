# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import copy
import torch
import torch.nn.functional as F
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset

class MaskedPhysionetSepsisDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("masked_physionet_sepsis", config, dataset_type, imdb_file_index, *args, **kwargs)
        self.model_type = config.model_type
        self.max_seq_length = config.max_seq_length
        self.use_pt1 = config.use_pt1
        self.use_pt2 = config.use_pt2
        self.pt1_probability = 0.15
        self.pt2_probability = 0.50
        self.BOS_IDX = config.max_pred_num
        np.random.seed(0)
        self.cls_token = np.random.normal(0, 1, (1, config.hidden_size)).astype(np.float32)
        np.random.seed(1)
        self.mask_token = np.random.normal(0, 1, (1, config.hidden_size)).astype(np.float32)
        np.random.seed(seed=None)

    def __getitem__(self, idx):
        sample_info = copy.deepcopy(self.annotation_db[idx])
        rand_idx = np.random.randint(0, len(self.annotation_db) - 1)
        rand_info = copy.deepcopy(self.annotation_db[rand_idx])
        
        current_sample = Sample()   

        current_sample = self.add_sample_details(sample_info, rand_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        return current_sample

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=-1)
        predictions = []

        for i in range(answers.shape[0]):
            predictions.append(
                {
                    "instance_id": i,
                    "prediction": answers[i].cpu().numpy().tolist(),
                }
            )

        return predictions

    def add_sample_details(self, sample_info, rand_info, sample):
        # Add cls/mask token into sample (normal distribution)
        sample.cls_token = torch.from_numpy(self.cls_token)
        sample.mask_token = torch.from_numpy(self.mask_token)
        sample.cls_mask = torch.tensor([1], dtype=torch.float32)

        # Random mask data and create pt1 regression ground truth
        if self.use_pt1:
            pt1_keys = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
            pt1_masks = []
            pt1_keep_masks = []
            pt1_labels = []
            for key in pt1_keys:
                pt1_mask = np.zeros(self.max_seq_length)
                pt1_keep_mask = np.ones(self.max_seq_length)
                pt1_label = np.zeros((self.max_seq_length, sample_info[key].shape[1]), dtype=np.float32)
                for i in range(sample_info[key].shape[0]):
                    pt1_prob = np.random.random()
                    if pt1_prob < self.pt1_probability:
                        pt1_label[i] = sample_info[key][i]
                        pt1_mask[i] = 1
                        pt1_prob /= self.pt1_probability
                        # 80% randomly change value to 1
                        if pt1_prob < 0.8:
                            pt1_keep_mask[i] = 0
                        # 10% randomly change to random value
                        elif pt1_prob < 0.9:
                            rand_sample_idx = np.random.randint(0, sample_info[key].shape[0])
                            sample_info[key][i] = sample_info[key][rand_sample_idx]
                        # rest 10% keep the original value as it is
                        else:
                            pass
                    
                pt1_masks.append(pt1_mask)
                pt1_keep_masks.append(pt1_keep_mask)
                pt1_labels.append(pt1_label)
            pt1_masks = np.concatenate(pt1_masks, axis=0)
            pt1_keep_masks = np.concatenate(pt1_keep_masks, axis=0)
            pt1_labels = np.concatenate(pt1_labels, axis=0)
            sample.pt1_masks = torch.tensor(pt1_masks, dtype=torch.float32)
            sample.pt1_keep_masks = torch.tensor(pt1_keep_masks, dtype=torch.float32)
            sample.pt1_labels = torch.from_numpy(pt1_labels)

        # Random replace one modality with other instances
        if self.use_pt2:
            pt2_keys = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
            pt2_idx = np.random.randint(len(pt2_keys))
            pt2_key = pt2_keys[pt2_idx]
            pt2_prob = np.random.random()
            if pt2_prob < self.pt2_probability:
                sample_info[pt2_key] = rand_info[pt2_key]
                is_correct = 0
            else:
                is_correct = 1
            pt2_label = torch.tensor(is_correct, dtype=torch.int64)
            pt2_target = F.one_hot(pt2_label, num_classes=2)
            sample.pt2_target = pt2_target.float()

        # Load data from corresponding keys and zero-pad up to same sequence length
        included_keys = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "laboratory", "demographic"]
        data_mask = [1] * len(sample_info[included_keys[0]]) + [0] * (self.max_seq_length - len(sample_info[included_keys[0]]))
        processed_data = {}
        for key in included_keys:
            raw_data = sample_info[key]
            pad_data = np.zeros((1,raw_data.shape[-1]), dtype=np.float32)
            while len(raw_data) < self.max_seq_length:
                raw_data = np.append(raw_data, pad_data, axis=0)
            processed_data[key] = torch.from_numpy(raw_data)
        sample.update(processed_data)
        sample.data_mask = torch.tensor(data_mask, dtype=torch.float32)

        return sample

    def add_answer_info(self, sample_info, sample):
        raw_label = list(sample_info["label"][-self.max_seq_length:])
        train_loss_mask = [1] * len(raw_label)

        while len(raw_label) < self.max_seq_length:
            raw_label.append(raw_label[-1])
            train_loss_mask.append(0)

        sample.train_loss_mask = torch.tensor(train_loss_mask, dtype=torch.float32)

        train_prev_inds = torch.zeros(self.max_seq_length, dtype=torch.int64)
        train_prev_inds[0] = self.BOS_IDX
        train_prev_inds[1:] = torch.tensor(raw_label[:-1], dtype=torch.int64)
        sample.train_prev_inds = train_prev_inds

        if self.model_type == "onetime":
            label = torch.tensor(raw_label[-1], dtype=torch.int64)
            label = F.one_hot(label, num_classes=2)
            sample.targets = label.float()
        else:
            label = torch.tensor(raw_label, dtype=torch.int64)
            label = F.one_hot(label, num_classes=2)
            sample.targets = label.float()

        return sample


