from dataclasses import dataclass

from config.datasets.base_dataset_config import BaseDatasetConfig
from const import ROOT_DIR


class JigsawDatasetConfig(BaseDatasetConfig):
    train_filename = str(ROOT_DIR / "data" / "jigsaw" / "train.csv")
    test_filename = str(ROOT_DIR / "data" / "jigsaw" / "test.csv")

    val_split_size = 0.2
    load_test_from_file = False

    labels = [
        'toxic', 'severe_toxic', 'obscene', 'threat',
        'insult', 'identity_hate'
    ]

    labels_columns = labels

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}

    text_column = "comment_text"