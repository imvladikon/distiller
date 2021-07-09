from pathlib import Path

import torch

from const import ROOT_DIR
from utils import dotdict

chekpoint_path = ROOT_DIR / 'models' / 'bert-base-uncased'
chekpoint_path = r'bert-base-uncased'
BERT_PRETRAINED_PATH = Path(chekpoint_path)

TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", 'PI', "OOO", "Payments", "Contract", "reply"]
TEST_LABELS = TRAIN_LABELS

model_name = 'reply'
suffix = '_x512'

DATA_PATH = ROOT_DIR / 'data' / '0'
BERT_FINETUNED_PATH = ROOT_DIR / 'models' / 'tuned' / f'tuned_bert{model_name}'

CLAS_DATA_PATH = DATA_PATH / 'class/output'
CLAS_DATA_PATH.mkdir(exist_ok=True, parents=True)

args = {
    "do_train": True,
    "do_eval": True,
    "one_cycle_train": True,
    "train_format_with_proba": False,
    # "input_fname": 'train_sel+cur+aug.csv',
    "train_size": -1,
    "val_size": -1,
    "data_dir": DATA_PATH,
    "task_name": "email_reject",
    "bert_tokenizer": Path(chekpoint_path),
    "bert_model_dir": BERT_PRETRAINED_PATH,
    "output_model_dir": BERT_FINETUNED_PATH,
    "data_output_dir": CLAS_DATA_PATH,
    "max_seq_length": 512,
    "do_lower_case": True,
    "train_batch_size": 24,
    "eval_batch_size": 12,
    "n_threads": 4,
    "learning_rate": 3e-5,
    "num_train_epochs": 3,
    "warmup_linear": False,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 888,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": True,
    "fp16": False,
    "loss_scale": 128,
    'model_name': model_name
}
args = dotdict(args)
n_gpu = torch.cuda.device_count()
args.n_gpu = n_gpu
