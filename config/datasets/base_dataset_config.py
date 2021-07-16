from typing import List, Dict


class BaseDatasetConfig:
    train_filename: str = ""
    val_filename: str = ""
    test_filename: str = ""
    val_split_size: float = 0.
    load_test_from_file: bool = True
    labels: List = []
    labels_columns: List = []
    id2label: Dict = dict()
    label2id: Dict = dict()
    text_column: str = ""
    transform_label_fn = None
    processing_data_fn = None
