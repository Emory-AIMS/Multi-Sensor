# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.girasoles_sensor.dataset import GirasolesSensorDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("girasoles_sensor")
class GirasolesSensorBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="girasoles_sensor", dataset_class=GirasolesSensorDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/girasoles_sensor/defaults.yaml"