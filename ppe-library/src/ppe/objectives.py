"""User-defined per-objective losses; no dataset imports."""
from dataclasses import dataclass
from typing import Callable
import math
import torch
import torch.nn.functional as F


def move_targets(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_targets(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_targets(v, device) for v in value)
    return value


@dataclass(frozen=True)
class Objectives:
    """loss(index, output_i, targets) -> differentiable mean scalar Tensor.

    Model outputs must be an ordered sequence, one output per objective.
    Optional metric uses the same arguments and returns a scalar (higher is better).
    Metrics are reported; validation losses drive model selection.
    """
    names: tuple[str, ...]
    loss: Callable
    score: Callable | None = None

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))
        if len(self.names) < 3:
            raise ValueError("This preserved training core requires at least 3 objectives; see the user guide's supported limits.")
        if len(set(self.names)) != len(self.names) or any(not n for n in self.names):
            raise ValueError("Objective names must be nonempty and unique.")

    @property
    def n_objectives(self):
        return len(self.names)

    def task_loss(self, index, output, targets):
        loss = self.loss(index, output, targets)
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("Each objective loss must be a scalar tensor with mean batch reduction.")
        if not torch.isfinite(loss).item():
            raise ValueError(f"Nonfinite loss for {self.names[index]}.")
        return loss

    def metric(self, index, output, targets):
        return float(self.score(index, output, targets)) if self.score else math.nan

    @classmethod
    def classification(cls, names):
        """Cross-entropy and top-1 contract: labels [batch, objectives]."""
        return cls(tuple(names),
                   lambda i, output, y: F.cross_entropy(output, y[:, i].long()),
                   lambda i, output, y: (output.topk(1, dim=1, largest=True, sorted=True)[1].squeeze(1)
                                         == y[:, i].long()).float().mean())
