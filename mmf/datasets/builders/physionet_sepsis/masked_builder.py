# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.physionet_sepsis.masked_dataset import MaskedPhysionetSepsisDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("masked_physionet_sepsis")
class MaskedPhysionetSepsisBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="masked_physionet_sepsis", dataset_class=MaskedPhysionetSepsisDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/physionet_sepsis/masked.yaml"