import os
import torch
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random_state = 200

labels = ['contract',
          'ooo',
          'payments',
          'pi',
          'pricing',
          'reject',
          'reply',
          'scheduling']

id2label = {k: v for k, v in enumerate(labels)}
label2id = {v: k for k, v in id2label.items()}