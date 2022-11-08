# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn.functional as F
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset

class GirasolesSensorDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("girasoles_sensor", config, dataset_type, imdb_file_index, *args, **kwargs)
        self.model_type = config.model_type
        self.max_seq_length = config.max_seq_length
        self.BOS_IDX = config.max_pred_num
        self.max_pred_num = config.max_pred_num
        self.answer_key = config.answer_key
        np.random.seed(0)
        self.cls_token = np.random.normal(0, 1, (1, config.hidden_size)).astype(np.float32)
        np.random.seed(1)
        self.sep_token = np.random.normal(0, 1, (1, config.hidden_size)).astype(np.float32)
        np.random.seed(seed=None)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()   

        current_sample = self.add_sample_details(sample_info, current_sample)
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

    def add_sample_details(self, sample_info, sample):
        # Add cls token into sample (normal distribution)
        sample.cls_token = torch.from_numpy(self.cls_token)
        sample.cls_mask = torch.tensor([1], dtype=torch.float32)
        
        # Load data from corresponding keys and prepare the pad
        included_keys = ['temp2', 'hr2', 'magnitude', 'yaxis', 'demographic']
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
        # self.answer_keys = ["aki_labels", "usg_labels", "temp"]
        core_temp = [37.384662667451906, 1.6970803973260453] # mean & std
        window_size = 10

        if self.answer_key != "temp":
            raw_label = list(sample_info[self.answer_key][-self.max_seq_length:])
            train_loss_mask = [1] * len(raw_label)
        else:
            raw_label = []
            temp_sample = sample_info[self.answer_key]
            orig_temp = temp_sample * core_temp[1] +  core_temp[0]
            for i in range(orig_temp.shape[0]-window_size):
                temp_window = orig_temp[i+1:i+1+window_size]
                temp_bool = temp_window > 38.0
                temp_occr = temp_bool.astype(int).sum()
                if temp_occr > 0:
                    raw_label.append(1)
                else:
                    raw_label.append(0)
            train_loss_mask = [1] * (len(raw_label)-window_size) + [0] * window_size

        while len(raw_label) < self.max_seq_length:
            raw_label.append(raw_label[-1])
            train_loss_mask.append(0)

        sample.train_loss_mask = torch.tensor(train_loss_mask, dtype=torch.float32)

        train_prev_inds = torch.zeros(self.max_seq_length, dtype=torch.int64)
        train_prev_inds[0] = self.BOS_IDX
        train_prev_inds[1:] = torch.tensor(raw_label[:-1], dtype=torch.int64)
        sample.train_prev_inds = train_prev_inds

        if self.model_type == "onetime":
            if self.answer_key == "aki_labels":
                label = torch.tensor(raw_label[-1], dtype=torch.int64)
            else:
                mean_usg = np.mean(raw_label)
                value = 0 if mean_usg < 1.030 else 1
                label = torch.tensor(value, dtype=torch.int64)
            label = F.one_hot(label, num_classes=self.max_pred_num)
            sample.targets = label.float()
        else:
            label = torch.tensor(raw_label, dtype=torch.int64)
            label = F.one_hot(label, num_classes=self.max_pred_num)
            sample.targets = label.float()

        return sample