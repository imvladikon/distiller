from config.datasets.base_dataset_config import BaseDatasetConfig
from const import ROOT_DIR
from functools import partial


def transform_label_fn_(l, label2id):
    return [int(k in l) for k in label2id]


class GongHardDatasetConfig(BaseDatasetConfig):
    train_filename = str(ROOT_DIR / "data" / "0" / "train.csv")
    val_filename = str(ROOT_DIR / "data" / "0" / "test.csv")
    test_filename = str(ROOT_DIR / "data" / "0" / "test.csv")

    load_test_from_file = True

    labels = [
        "reject",
        "scheduling",
        "pricing",
        "pi",
        "ooo",
        "payments",
        "contract",
        "reply"
    ]
    labels_columns = labels

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}

    text_column = "document_text"

    transform_label_fn = partial(transform_label_fn_, label2id=label2id)
    processing_data_fn = None
