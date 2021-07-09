import pandas as pd
from ast import literal_eval

def load_dataset(name="email"):
    train_df = pd.read_csv("data/0/train.csv").reset_index(drop=True)
    test_df = pd.read_csv("data/0/test.csv").reset_index(drop=True)
    train_df["label_text"] = train_df.label_text.map(literal_eval)
    test_df["label_text"] = test_df.label_text.map(literal_eval)
    return ""