from config.datasets.base_dataset_config import BaseDatasetConfig
from const import ROOT_DIR
from functools import partial


def transform_label_fn_(ll):
    """
    transform_label_fn_([-1,0,0,0,1,-1]) = [0,0,0,0,1,0]
    Args:
        threshold:
    Returns:
    """
    return [int(l > 0) for l in ll]


class GongWithWeakLabeledConfig(BaseDatasetConfig):
    train_filename = str(ROOT_DIR / "data" / "train_test_labeled" / "train.csv")
    val_filename = str(ROOT_DIR / "data" / "train_test_labeled" / "test.csv")
    test_filename = str(ROOT_DIR / "data" / "train_test_labeled" / "test.csv")

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

    transform_label_fn = transform_label_fn_
    processing_data_fn = None

if __name__ == '__main__':
    from config.datasets import DataFactory

    ds = DataFactory.create_from_config("gong_with_weak_labeled")
    dataset_info = ds.config
    ds = ds.dataset