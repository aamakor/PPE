"""Copy this file to my_experiment.py and customize the constants/model below.

From its directory: ppe run my_experiment:build
DATA_DIR=None runs generated data. Otherwise read train.npz, val.npz, test.npz,
each containing inputs [samples, INPUT_DIM] and targets [samples, objective_count].
For other data formats, replace make_loaders() with your own PyTorch datasets.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ppe import PPE, PPEConfig, Objectives, TerminalInteraction


# Keep names, class counts, model outputs and target columns in the same order.
OBJECTIVE_NAMES = ("task A", "task B", "task C")
CLASS_COUNTS = (2, 3, 2)
INPUT_DIM = 8
BATCH_SIZE = 8
DATA_DIR = None  # Or Path("data") containing your three .npz split files.
OUTPUT_DIR = "runs/my-dataset"  # Choose a fresh directory for each new run.


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, class_counts):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.heads = nn.ModuleList(nn.Linear(16, count) for count in class_counts)

    def forward(self, inputs):
        shared = self.shared(inputs)
        return [head(shared) for head in self.heads]


def make_loaders():
    def make_loader(n, shuffle=False):
        inputs = torch.randn(n, INPUT_DIM)
        labels = torch.stack([torch.randint(c, (n,)) for c in CLASS_COUNTS], dim=1)
        return DataLoader(TensorDataset(inputs, labels), batch_size=BATCH_SIZE, shuffle=shuffle)

    if DATA_DIR is None:
        return make_loader(64, True), make_loader(32), make_loader(32)

    loaders = []
    for split in ("train", "val", "test"):
        with np.load(Path(DATA_DIR) / f"{split}.npz", allow_pickle=False) as data:
            inputs = torch.as_tensor(data["inputs"], dtype=torch.float32)
            targets = torch.as_tensor(data["targets"], dtype=torch.long)
        dataset = TensorDataset(inputs, targets)
        loaders.append(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=split == "train"))
    return tuple(loaders)


def build():
    """Return PPE, three loaders and terminal interaction; the CLI calls fit()."""
    torch.manual_seed(24)
    model = MultiTaskModel(INPUT_DIM, CLASS_COUNTS).to("cpu")

    train, val, test = make_loaders()
    config = PPEConfig(OUTPUT_DIR, num_init=5, num_pred=10, num_corr=15,
                       step_size=0.01, cycles=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    corrector = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    trainer = PPE(model, Objectives.classification(OBJECTIVE_NAMES), optimizer, corrector, config,
                  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_init, eta_min=1e-5),
                  corrector_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(corrector, T_max=config.num_corr, eta_min=1e-5))
    return trainer, (train, val, test), TerminalInteraction()


if __name__ == "__main__":
    trainer, loaders, interaction = build()
    print(trainer.fit(*loaders, interaction=interaction))
