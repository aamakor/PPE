"""Small, download-free three-objective quadratic training example.

This exercises the deep-learning core, not the separate published DTLZ solver.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from . import PPE, PPEConfig, Objectives, ScriptedInteraction


class QuadraticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position = torch.nn.Parameter(torch.tensor([0.2, 0.1]))

    def forward(self, inputs):
        point = self.position.expand(len(inputs), -1)
        return point, point, point


def make_demo(output_dir, *, cycles=1, device="cpu"):
    model = QuadraticModel().to(device)
    anchors = torch.tensor([[1., 0.], [-0.5, 0.8660254], [-0.5, -0.8660254]])
    # Separate splits, identical deterministic objective landscape.
    def loader(count):
        return DataLoader(TensorDataset(torch.zeros(count, 1), anchors.expand(count, -1, -1).clone()), batch_size=4)
    objectives = Objectives(("Obj A", "Obj B", "Obj C"),
                            lambda i, output, y: (output - y[:, i]).square().mean())
    config = PPEConfig(output_dir, num_init=2, num_pred=3, num_corr=2,
                       num_minres=10, step_size=0.05, cycles=cycles)
    trainer = PPE(model, objectives, torch.optim.SGD(model.parameters(), lr=0.01),
                  torch.optim.SGD(model.parameters(), lr=0.001), config)
    interaction = ScriptedInteraction([[-1, 0, 0]] * max(1, cycles),
                                     accept_fronts=[True] * max(1, cycles))
    return trainer, (loader(16), loader(8), loader(8)), interaction
