from itertools import chain

from catalyst.callbacks import AccuracyCallback
from catalyst.contrib.datasets import MNIST
import torch
from torch import nn
from torch.utils.data import DataLoader

from distillation.runners import EndToEndDistilRunner
from utils.data import TorchvisionDatasetWrapper as Wrp




class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int = 28 * 28,
        num_classes: int = 10,
        num_layers: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        layers = []
        self.flatten = nn.Flatten()
        self.num_layers = num_layers
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layers.append(nn.Linear(inp_dim, hidden_dim))
            elif layer_idx == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, num_classes))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, inp, output_hidden_states: bool = False, return_dict: bool = False):
        cur_hidden = self.flatten(inp)
        if output_hidden_states:
            hiddens = []

        for layer_idx, layer in enumerate(self.layers):
            cur_hidden = layer(cur_hidden)
            if output_hidden_states:  #  accumulate hidden states
                hiddens.append(cur_hidden)

            if layer_idx != self.num_layers - 1:  # last layer case
                cur_hidden = torch.relu(cur_hidden)

        logits = cur_hidden
        if return_dict:
            output = {"logits": logits}
            if output_hidden_states:
                output["hidden_states"] = tuple(hiddens)
            return output

        if output_hidden_states:
            return logits, tuple(hiddens)

        return logits


teacher = MLP(num_layers=4)
student = MLP(num_layers=3)

datasets = {
    "train": Wrp(MNIST("./data", train=True, download=True)),
    "valid": Wrp(MNIST("./data", train=False)),
}

loaders = {
    dl_key: DataLoader(dataset, shuffle=dl_key == "train", batch_size=32)
    for dl_key, dataset in datasets.item()
}

optimizer = torch.optim.Adam(chain(teacher.parameters(), student.parameters()))

runner = EndToEndDistilRunner(hidden_state_loss="mse", num_train_teacher_epochs=5)

runner.train(
    model={"teacher": teacher, "student": student},
    loaders=loaders,
    optimizer=optimizer,
    num_epochs=4,
    callbacks=[AccuracyCallback(input_key="logits", target_key="targets")],
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    logdir="./logs",
)
