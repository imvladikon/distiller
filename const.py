import os
import torch
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random_state = 200