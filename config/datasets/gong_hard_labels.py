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
    labels_columns = [f"was_label_{col}" for col in labels]

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}

    text_column = "document_text"

    transform_label_fn = transform_label_fn_
    processing_data_fn = None

if __name__ == '__main__':
    from config.datasets import DataFactory

    ds = DataFactory.create_from_config("gong_hard_labels")
    dataset_info = ds.config
    ds = ds.dataset