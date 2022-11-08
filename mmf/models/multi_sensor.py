# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer, ConvNet, Flatten
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)


logger = logging.getLogger(__name__)


@registry.register_model("multi_sensor")
class MultiSensor(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = registry.get("config").datasets.split(",")[0]

    @classmethod
    def config_path(cls):
        return "configs/models/multi_sensor/defaults.yaml"

    def build(self):
        if self.config.fusion_type == "early":
            self.gru = nn.GRU(input_size=self.config.data_in_dim.total, **self.config.gru)

            classifier_config = deepcopy(self.config.classifier)
            self.classifier = ClassifierLayer(
                classifier_config.type, **classifier_config.params
            )
        elif self.config.fusion_type == "late":
            if self.dataset_name == "physionet_sepsis":
                
                self.gru_vital = nn.GRU(input_size=self.config.data_in_dim.vital, **self.config.gru)
                self.gru_lab = nn.GRU(input_size=self.config.data_in_dim.laboratory, **self.config.gru)
                self.linear_dem = nn.Linear(self.config.data_in_dim.demographic, self.config.hidden_size)

                self.linear_cat = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size)

                classifier_config = deepcopy(self.config.classifier)
                self.classifier = ClassifierLayer(
                    classifier_config.type, **classifier_config.params
                )
            elif self.dataset_name == "girasoles_sensor":

                self.gru_temp = nn.GRU(input_size=self.config.data_in_dim.double_vital, **self.config.gru)
                self.gru_hr = nn.GRU(input_size=self.config.data_in_dim.double_vital, **self.config.gru)
                self.gru_magnitude = nn.GRU(input_size=self.config.data_in_dim.single_vital, **self.config.gru)
                self.gru_yaxis = nn.GRU(input_size=self.config.data_in_dim.single_vital, **self.config.gru)
                self.linear_dem = nn.Linear(self.config.data_in_dim.demographic, self.config.hidden_size)

                self.linear_cat = nn.Linear(self.config.hidden_size * 5, self.config.hidden_size)

                classifier_config = deepcopy(self.config.classifier)
                self.classifier = ClassifierLayer(
                    classifier_config.type, **classifier_config.params
                )


    def forward(self, sample_list):
        if self.config.fusion_type == "early":
            if self.dataset_name == "physionet_sepsis":
                vital = sample_list.vital
                laboratory = sample_list.laboratory
                demographic = sample_list.demographic
                concat = torch.cat([vital, laboratory, demographic], dim=-1)
            elif self.dataset_name == "girasoles_sensor":
                temp2 = sample_list.temp2
                hr2 = sample_list.hr2
                magnitude = sample_list.magnitude
                yaxis = sample_list.yaxis
                demographic = sample_list.demographic
                concat = torch.cat([temp2, hr2, magnitude, yaxis, demographic], dim=-1)

            self.gru.flatten_parameters()
            if self.config.model_type == "onetime":
                _, hidden = self.gru(concat)
                scores = self.classifier(hidden[0])
            else:
                hidden, _ = self.gru(concat)
                scores = self.classifier(hidden)
            return {"scores": scores}

        elif self.config.fusion_type == "late":
            if self.dataset_name == "physionet_sepsis":
                vital = sample_list.vital
                laboratory = sample_list.laboratory
                demographic = sample_list.demographic

                self.gru_vital.flatten_parameters()
                self.gru_lab.flatten_parameters()

                if self.config.model_type == "onetime":
                    _, hidden_vital = self.gru_vital(vital)
                    _, hidden_lab = self.gru_lab(laboratory)
                    hidden_dem = self.linear_dem(demographic)
                    hidden_cat = torch.cat([hidden_vital[0], hidden_lab[0], hidden_dem[:,0]], dim=-1)
                    concat = self.linear_cat(hidden_cat)
                    scores = self.classifier(concat)
                else:
                    hidden_vital, _ = self.gru_vital(vital)
                    hidden_lab, _ = self.gru_lab(laboratory)
                    hidden_dem = self.linear_dem(demographic)
                    hidden_cat = torch.cat([hidden_vital, hidden_lab, hidden_dem], dim=-1)
                    concat = self.linear_cat(hidden_cat)
                    scores = self.classifier(concat)

                return {"scores": scores}    

            elif self.dataset_name == "girasoles_sensor":
                self.gru_temp.flatten_parameters()
                self.gru_hr.flatten_parameters()
                self.gru_magnitude.flatten_parameters()
                self.gru_yaxis.flatten_parameters()

                temp2 = sample_list.temp2
                hr2 = sample_list.hr2
                magnitude = sample_list.magnitude
                yaxis = sample_list.yaxis
                demographic = sample_list.demographic   

                if self.config.model_type == "onetime":
                    _, hidden_temp = self.gru_temp(temp2)
                    _, hidden_hr = self.gru_hr(hr2)
                    _, hidden_mag = self.gru_magnitude(magnitude)
                    _, hidden_yaxis = self.gru_yaxis(yaxis)
                    hidden_dem = self.linear_dem(demographic)
                    hidden_cat = torch.cat([hidden_temp[0], hidden_hr[0], hidden_mag[0], hidden_yaxis[0], hidden_dem[:,0]], dim=-1)
                    concat = self.linear_cat(hidden_cat)
                    scores = self.classifier(concat)
                else:
                    hidden_temp, _ = self.gru_temp(temp2)
                    hidden_hr, _ = self.gru_hr(hr2)
                    hidden_mag, _ = self.gru_magnitude(magnitude)
                    hidden_yaxis, _ = self.gru_yaxis(yaxis)
                    hidden_dem = self.linear_dem(demographic)
                    hidden_cat = torch.cat([hidden_temp, hidden_hr, hidden_mag, hidden_yaxis, hidden_dem], dim=-1)
                    concat = self.linear_cat(hidden_cat)
                    scores = self.classifier(concat)

                return {"scores": scores}         
