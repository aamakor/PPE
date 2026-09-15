"""PPE training API by Augustina C. Amakor."""
from dataclasses import dataclass
from pathlib import Path
import math
import torch
from .interaction import NavigationStopped, TerminalInteraction
from .runtime import CheckpointModel, Runtime


@dataclass(frozen=True)
class PPEConfig:
    output_dir: str | Path
    num_init: int = 500
    num_pred: int = 10
    num_corr: int = 15
    num_minres: int = 100
    step_size: float = 0.01
    cycles: int | None = None
    resume: bool = False
    run_name: str = "custom"

    def __post_init__(self):
        for name in ("num_init", "num_pred", "num_corr", "num_minres"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_pred < 2:
            raise ValueError("The logging/early-stopping loop requires num_pred >= 2.")
        if self.cycles is not None and (type(self.cycles) is not int or self.cycles < 0):
            raise ValueError("cycles must be a nonnegative integer or None.")
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be positive and finite.")
        if not self.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.run_name):
            raise ValueError("run_name may contain letters, digits, underscores and hyphens.")


class PPE:
    """Call fit with map-style loaders yielding (input_tensor, targets).

    Place the model on its device BEFORE constructing optimizers and this object.
    Both optimizers must own exactly this model's parameters. Model parameters must
    all require gradients and support the second derivatives used by the predictor.
    """
    def __init__(self, model, objectives, optimizer, corrector_optimizer, config,
                 *, scheduler=None, corrector_scheduler=None):
        self.model = model
        self.objectives = objectives
        self.optimizer = optimizer
        self.corrector_optimizer = corrector_optimizer
        self.config = config
        self.scheduler = scheduler
        self.corrector_scheduler = corrector_scheduler
        params = list(model.parameters())
        if not params or any(not p.requires_grad for p in params):
            raise ValueError("PPE requires a nonempty, fully trainable model.")
        if len({p.device for p in params}) != 1:
            raise ValueError("Model parameters must be on a single device.")
        for opt in (optimizer, corrector_optimizer):
            owned = [p for group in opt.param_groups for p in group["params"]]
            if len(owned) != len(params) or {id(p) for p in owned} != {id(p) for p in params}:
                raise ValueError("Each optimizer must own exactly this model's parameters.")
        for sched, opt in ((scheduler, optimizer), (corrector_scheduler, corrector_optimizer)):
            if sched is not None and sched.optimizer is not opt:
                raise ValueError("Scheduler must belong to the corresponding optimizer.")

    def fit(self, train_loader, val_loader, test_loader, *, interaction=None, on_cycle=None, on_initial=None):
        from ._training import run_training
        for name, loader in (("training", train_loader), ("validation", val_loader), ("test", test_loader)):
            if len(loader) == 0:
                raise ValueError(f"{name} loader is empty.")
        if self.config.cycles != 0 and len(train_loader) < 4:
            raise ValueError("Predictor requires >= 4 training batches (quarter-loader Jacobian rule).")
        if train_loader.batch_size is None:
            raise ValueError("Provide a DataLoader with batch_size and a map-style dataset.")
        path = Path(self.config.output_dir).resolve() / "checkpoints"
        if path.exists() and any(path.iterdir()) and not self.config.resume:
            raise FileExistsError("Use a fresh output_dir or resume=True to continue this run.")
        path.mkdir(parents=True, exist_ok=True)
        device = next(self.model.parameters()).device
        runtime = Runtime(self.config, interaction if interaction is not None else TerminalInteraction(), device, on_cycle, on_initial)
        from .recovery import Recovery
        runtime.recovery = Recovery(runtime, self.model,
                                    (self.optimizer, self.corrector_optimizer),
                                    (self.scheduler, self.corrector_scheduler),
                                    (train_loader, val_loader, test_loader), self.objectives.names)
        try:
            return run_training(
                model=CheckpointModel(self.model, self.objectives.n_objectives), lr=self.config.step_size,
                nobj=self.objectives.n_objectives, optimizer=self.optimizer,
                optimizer_c=self.corrector_optimizer, num_init=self.config.num_init,
                path=path, lr_scheduler=self.scheduler, lr_scheduler_c=self.corrector_scheduler,
                criterion=self.objectives, num_pred=self.config.num_pred,
                trainloader=train_loader, valloader=val_loader, testloader=test_loader,
                device=device, type=self.config.run_name, num_corr=self.config.num_corr,
                num_minres=self.config.num_minres, runtime=runtime)
        except NavigationStopped as exc:
            runtime._result.status = "stopped"
            runtime._result.reason = str(exc)
            return runtime.result()
