"""Durable navigation boundaries and recorded decisions for cycle recovery."""
import copy
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch


def _atomic_write(path, write):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(value, path):
    _atomic_write(path, lambda stream: torch.save(value, stream))


def atomic_json(value, path):
    data = (json.dumps(value, indent=2) + '\n').encode('utf-8')
    _atomic_write(path, lambda stream: stream.write(data))


class Recovery:
    def __init__(self, runtime, model, optimizers, schedulers, loaders, objective_names):
        self.runtime = runtime
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.generators = []
        for loader in loaders:
            for owner in (loader, loader.sampler, getattr(loader, 'batch_sampler', None)):
                generator = getattr(owner, 'generator', None)
                if generator is not None and all(generator is not g for g in self.generators):
                    self.generators.append(generator)
        self.path = Path(runtime.config.output_dir).resolve() / 'checkpoints/recovery.pt'
        self.signature = {k: v for k, v in vars(runtime.config).items()
                          if k not in ('resume', 'cycles', 'output_dir')}
        self.signature['objectives'] = list(objective_names)
        self.signature['optimizers'] = [self._kind(o) for o in optimizers]
        self.signature['schedulers'] = [self._kind(s) for s in schedulers]
        self.signature['loaders'] = [(len(l.dataset), len(l), l.batch_size,
                                      self._kind(l.sampler)) for l in loaders]
        self.signature['generators'] = len(self.generators)
        self.payload = None
        self.cursor = 0

    @staticmethod
    def _kind(value):
        return None if value is None else f'{type(value).__module__}.{type(value).__qualname__}'

    def random_state(self):
        return {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'generators': [g.get_state() for g in self.generators]}

    def restore_random(self, state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if state['cuda']:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
        for generator, saved in zip(self.generators, state['generators']):
            generator.set_state(saved.cpu())

    def save(self, state):
        # Phase aliases are working files. Preserve the boundary versions for replay.
        aliases = {p.name: torch.load(p, map_location='cpu', weights_only=True)
                   for p in self.path.parent.glob('model_*.pth')}
        self.payload = copy.deepcopy({
            'version': 1, 'signature': self.signature, 'state': state,
            'model': self.model.state_dict(),
            'training': [m.training for m in self.model.modules()],
            'gradients': [p.grad for p in self.model.parameters()],
            'optimizers': [o.state_dict() for o in self.optimizers],
            'schedulers': [s.state_dict() if s is not None else None for s in self.schedulers],
            'random': self.random_state(), 'result': vars(self.runtime._result),
            'aliases': aliases, 'decisions': []})
        atomic_save(self.payload, self.path)
        self.cursor = 0

    def restore(self):
        if not self.runtime.resume:
            return None
        if not self.path.exists():
            raise FileNotFoundError(f'No recovery checkpoint at {self.path}. Use a new output_dir for a new run.')
        # Recovery files are local PPE run artifacts containing NumPy/Python state.
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        if payload.get('version') != 1 or payload['signature'] != self.signature:
            raise ValueError('Recovery configuration differs. Recreate the same model, objectives, loaders, optimizers, schedulers and training settings; cycles may change.')
        self.payload = payload
        self.model.load_state_dict(payload['model'])
        for module, mode in zip(self.model.modules(), payload['training']):
            module.training = mode
        for parameter, gradient in zip(self.model.parameters(), payload['gradients']):
            parameter.grad = None if gradient is None else gradient.to(parameter.device)
        for optimizer, state in zip(self.optimizers, payload['optimizers']):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(self.schedulers, payload['schedulers']):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        for name, weights in payload['aliases'].items():
            atomic_save(weights, self.path.parent / name)
        for name, value in payload['result'].items():
            setattr(self.runtime._result, name, value)
        self.runtime._result.output_dir = self.path.parent.parent
        self.runtime._result.status = 'completed'
        self.runtime._result.reason = None
        self.restore_random(payload['random'])
        self.cursor = 0
        return copy.deepcopy(payload['state'])

    def decision(self, name, callback):
        journal = self.payload['decisions']
        if self.cursor < len(journal):
            entry = journal[self.cursor]
            if entry['name'] != name:
                raise RuntimeError('Recovery decision sequence changed. Use the same data and deterministic training setup.')
            value = copy.deepcopy(entry['value'])
            self.restore_random(entry['random'])
        else:
            value = callback()
            journal.append({'name': name, 'value': copy.deepcopy(value), 'random': self.random_state()})
            atomic_save(self.payload, self.path)
        self.cursor += 1
        return value
