from config.datasets.gong_hard_labels import GongHardDatasetConfig
from config.datasets.gong_soft_labels import GongSoftDatasetConfig
from config.datasets.jigsaw_info import JigsawDatasetConfig
from utils.dataloader import read_data

DATASETS_CONFIG_INFO = {
    "jigsaw": JigsawDatasetConfig,
    "gong_hard_labels": GongHardDatasetConfig,
    "gong_soft_labels": GongSoftDatasetConfig
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
            tokenizer=kwargs["tokenizer"],
            max_seq_length=kwargs["max_length"],
            train_size=kwargs["train_size"],
            val_size=kwargs["val_size"],
            map_label_columns=dict(zip(ds_info.labels, ds_info.labels_columns)),
            text_column=ds_info.text_column,
            val_split_size=ds_info.val_split_size,
            print_label_dist=True,
            transform_label_fn=ds_info.transform_label_fn,
            processing_data_fn=ds_info.processing_data_fn,
        )
        return cls(dataset=ds, config=ds_info)
