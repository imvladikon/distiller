from functools import partial

from config.datasets.base_dataset_config import BaseDatasetConfig
from const import ROOT_DIR


def transform_label_fn_(ll, threshold):
    return [int(l > threshold) for l in ll]


class GongSoftDatasetConfig(BaseDatasetConfig):
    train_filename = str(ROOT_DIR / "data" / "0" / "train_weak_label_bin_email_id.csv")
    val_filename = str(ROOT_DIR / "data" / "0" / "test.csv")
    test_filename = str(ROOT_DIR / "data" / "0" / "test.csv")

    load_test_from_file = True
    threshold = 0.8

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
    labels_columns = [f"proba_{col}" for col in labels]

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}

    text_column = "document_text"

    transform_label_fn = partial(transform_label_fn_, threshold=threshold)
    processing_data_fn = None
