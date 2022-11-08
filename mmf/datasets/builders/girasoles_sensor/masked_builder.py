# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.girasoles_sensor.masked_dataset import MaskedGirasolesSensorDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("masked_girasoles_sensor")
class MaskedGirasolesSensorBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="masked_girasoles_sensor", dataset_class=MaskedGirasolesSensorDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/girasoles_sensor/masked.yaml"