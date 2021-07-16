from functools import partial

from config.datasets.base_dataset_config import BaseDatasetConfig
from const import ROOT_DIR


def transform_label_fn_(ll, threshold):
    return [int(l > threshold) for l in ll]


class GongSoftDatasetConfig(BaseDatasetConfig):
    train_filename = str(ROOT_DIR / "data" / "0" / "train_weak_label_bin_email_id.csv")

    """
    test file doesn't have a probabilites, then we skip it
    """
    # val_filename = str(ROOT_DIR / "data" / "0" / "test.csv")
    # test_filename = str(ROOT_DIR / "data" / "0" / "test.csv")

    load_test_from_file = False
    threshold = 0.8
    val_split_size = 0.2

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

if __name__ == '__main__':
    from config.datasets import DataFactory

    ds = DataFactory.create_from_config("gong_soft_labels")
    dataset_info = ds.config
    ds = ds.dataset
