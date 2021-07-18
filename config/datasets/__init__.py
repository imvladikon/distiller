from config.datasets.gong_hard_labels import GongHardDatasetConfig
from config.datasets.gong_soft_labels import GongSoftDatasetConfig
from config.datasets.gong_with_weak_labeled import GongWithWeakLabeledConfig
from config.datasets.jigsaw_info import JigsawDatasetConfig
from config.datasets.utils import read_data

DATASETS_CONFIG_INFO = {
    "jigsaw": JigsawDatasetConfig,
    "gong_hard_labels": GongHardDatasetConfig,
    "gong_soft_labels": GongSoftDatasetConfig,
    "gong_with_weak_labeled": GongWithWeakLabeledConfig
}


class DataFactory:

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    @classmethod
    def create_from_config(cls, config_name, *args, **kwargs):
        ds_info = DATASETS_CONFIG_INFO[config_name]

        ds = read_data(
            train_filename=ds_info.train_filename,
            val_filename=ds_info.val_filename,
            tokenizer=kwargs.get("tokenizer", None),
            max_seq_length=kwargs.get("max_length", 512),
            train_size=kwargs.get("train_size", -1),
            val_size=kwargs.get("val_size", -1),
            map_label_columns=dict(zip(ds_info.labels, ds_info.labels_columns)),
            text_column=ds_info.text_column,
            val_split_size=ds_info.val_split_size,
            print_label_dist=True,
            transform_label_fn=ds_info.transform_label_fn,
            processing_data_fn=ds_info.processing_data_fn,
        )
        return cls(dataset=ds, config=ds_info)
