"""PPE filesystem, reporting and timing services."""
from dataclasses import dataclass, field
from pathlib import Path
import uuid
from .recovery import atomic_json, atomic_save
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from .interaction import validate_preference


def get_random_batch(trainloader):
    # Same random-sample policy as Data/mnist_data.py; preserve user's collator.
    sampler = RandomSampler(trainloader.dataset)
    loader = DataLoader(trainloader.dataset, batch_size=trainloader.batch_size,
                        sampler=sampler, collate_fn=trainloader.collate_fn)
    return next(iter(loader))


class CheckpointModel(torch.nn.Module):
    """Save and load the supplied model using plain state dictionaries."""
    def __init__(self, model, n_objectives):
        super().__init__()
        self.model = model
        self.n_objectives = n_objectives

    def forward(self, inputs):
        if not torch.is_tensor(inputs) or inputs.ndim < 1:
            raise ValueError("Inputs must be a batched tensor; adapt structured inputs in your dataset/model.")
        outputs = self.model(inputs)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != self.n_objectives:
            raise ValueError("Model must return a list/tuple with one output per objective.")
        return outputs

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def save_model(self, path):
        atomic_save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=next(self.parameters()).device,
                                               weights_only=True))


@dataclass
class RunResult:
    output_dir: Path
    status: str = "completed"
    reason: str | None = None
    cycles: list[dict] = field(default_factory=list)
    initial: dict | None = None
    objective_names: list[str] = field(default_factory=list)


class Runtime:
    def __init__(self, config, interaction, device, on_cycle=None, on_initial=None):
        self.config = config
        self.interaction = interaction
        self.device = torch.device(device)
        self.resume = config.resume
        self.results_dir = Path(config.output_dir).resolve() / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._result = RunResult(Path(config.output_dir).resolve())
        self.on_cycle = on_cycle
        self.on_initial = on_initial

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def keep_running(self, count):
        return self.config.cycles is None or count < self.config.cycles

    def preference(self, n):
        return self.recovery.decision('preference', lambda: validate_preference(self.interaction.preference(n), n))

    def step_size(self):
        return self.recovery.decision('step_size', self._step_size)

    def _step_size(self):
        value = float(self.interaction.step_size())
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Step size must be positive and finite.")
        return value

    def accept_front(self):
        return self.recovery.decision('accept_front', self.interaction.accept_front)

    def action(self, pref, max_runs):
        return self.recovery.decision('action', lambda: self._action(pref, max_runs))

    def _action(self, pref, max_runs):
        action, value, steps = self.interaction.action(pref, max_runs)
        if action not in ("change_pref", "rerun_predictor"):
            raise ValueError("Unknown predictor action.")
        if action == "rerun_predictor" and (type(steps) is not int or steps < 1):
            raise ValueError("Predictor runs must be a positive integer.")
        return action, validate_preference(value, len(pref)), steps

    def record_cycle(self, count, pref, predictor, corrector, test_predictor, test_corrector):
        record = {"cycle": count, **{k: np.asarray(v).tolist() for k, v in {
            "preference": pref, "predictor": predictor, "corrector": corrector,
            "test_predictor": test_predictor, "test_corrector": test_corrector}.items()}}
        self._result.cycles.append(record)
        self.result()

    def notify_cycle(self):
        if self.on_cycle:
            self.on_cycle(self._result.cycles[-1])

    def record_initial(self, validation, test, objective_names):
        self._result.objective_names = list(objective_names)
        record = {"validation": np.asarray(validation).tolist(), "test": np.asarray(test).tolist()}
        self._result.initial = record
        self.result()

    def notify_initial(self):
        if self.on_initial and self._result.initial:
            self.on_initial({**self._result.initial, "objective_names": self._result.objective_names})

    def save_boundary(self, state):
        self.recovery.save(state)

    def restore_boundary(self):
        return self.recovery.restore()

    def archive_point(self, model, phase, losses, alpha, **position):
        directory = self._result.output_dir / 'checkpoints/archive'
        directory.mkdir(parents=True, exist_ok=True)
        stem = f'{phase}_{uuid.uuid4().hex}'
        checkpoint = directory / f'{stem}.pth'
        model.save_model(checkpoint)
        weights = alpha.detach().cpu().numpy() if torch.is_tensor(alpha) else np.asarray(alpha)
        record = {'phase': phase, **position, 'validation': np.asarray(losses).tolist(),
                  'alpha': weights.tolist(), 'checkpoint': checkpoint.name}
        atomic_json(record, directory / f'{stem}.json')
        return checkpoint

    def result(self):
        payload = {"status": self._result.status, "reason": self._result.reason,
                   "cycles": self._result.cycles, "initial": self._result.initial,
                   "objective_names": self._result.objective_names}
        atomic_json(payload, self.results_dir / "navigation.json")
        return self._result
