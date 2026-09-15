"""Check snapshot identity and interrupted training against uninterrupted runs."""
import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ppe import ScriptedInteraction
from ppe.demo import make_demo
from ppe import _training
from ppe._numerics import evaluate
from ppe.runtime import CheckpointModel
from ppe.recovery import atomic_save


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_failed_checkpoint_write_keeps_previous_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recovery.pt'
            atomic_save({'step': 1}, path)
            before = path.read_bytes()
            def fail(value, stream):
                stream.write(b'incomplete checkpoint')
                raise OSError('simulated write failure')
            with patch('ppe.recovery.torch.save', side_effect=fail):
                with self.assertRaisesRegex(OSError, 'simulated write failure'):
                    atomic_save({'step': 2}, path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def make(self, directory, resume=False, cycles=1):
        random.seed(21)
        np.random.seed(21)
        torch.manual_seed(21)
        trainer, loaders, decisions = make_demo(directory, cycles=cycles)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.01, momentum=.8)
        trainer.corrector_optimizer = torch.optim.Adam(trainer.model.parameters(), lr=.001)
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 2, gamma=.95)
        trainer.corrector_scheduler = torch.optim.lr_scheduler.StepLR(trainer.corrector_optimizer, 2, gamma=.95)
        trainer.config = replace(trainer.config, resume=resume)
        for i, loader in enumerate(loaders):
            loader.generator = torch.Generator().manual_seed(100 + i)
        return trainer, loaders, decisions

    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_archive_weights_reproduce_every_loss(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer, loaders, decisions = self.make(directory)
            trainer.fit(*loaders, interaction=decisions)
            records = list((Path(directory) / 'checkpoints/archive').glob('*.json'))
            self.assertGreaterEqual(len(records), 5)
            filenames = []
            for path in records:
                record = json.loads(path.read_text())
                filenames.append(record['checkpoint'])
                trainer.model.load_state_dict(torch.load(path.parent / record['checkpoint'], weights_only=True))
                losses, _ = evaluate(CheckpointModel(trainer.model, 3), loaders[1], 3, trainer.objectives)
                np.testing.assert_allclose(losses, record['validation'], rtol=0, atol=0)
            self.assertEqual(len(filenames), len(set(filenames)))

    def test_interrupted_initialization_predictor_and_corrector(self):
        for phase in ('initialization', 'predictor', 'corrector'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as baseline, tempfile.TemporaryDirectory() as interrupted, contextlib.redirect_stdout(io.StringIO()):
                expected, loaders, decisions = self.make(baseline)
                result = expected.fit(*loaders, interaction=decisions)
                expected_rng = (random.random(), np.random.rand(), torch.rand(3))
                trainer, loaders, decisions = self.make(interrupted)
                calls = 0
                name = 'predictor_step' if phase == 'predictor' else 'mgda_optimize'
                function = getattr(_training, name)
                stop_at = {'initialization': 3, 'predictor': 2, 'corrector': 10}[phase]
                def crash(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == stop_at:
                        raise RuntimeError('simulated interruption')
                    return function(*args, **kwargs)
                with patch.object(_training, name, side_effect=crash):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        trainer.fit(*loaders, interaction=decisions)
                archive = Path(interrupted) / 'checkpoints/archive'
                hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.glob('*.pth')}
                resumed, loaders, decisions = self.make(interrupted, resume=True)
                if phase != 'initialization':
                    decisions = ScriptedInteraction([], accept_fronts=[True])
                actual = resumed.fit(*loaders, interaction=decisions)
                self.assertEqual(actual.status, 'completed')
                self.assertEqual(actual.cycles, result.cycles)
                for a, b in ((expected.model.state_dict(), resumed.model.state_dict()),
                             (expected.optimizer.state_dict(), resumed.optimizer.state_dict()),
                             (expected.corrector_optimizer.state_dict(), resumed.corrector_optimizer.state_dict()),
                             (expected.scheduler.state_dict(), resumed.scheduler.state_dict()),
                             (expected.corrector_scheduler.state_dict(), resumed.corrector_scheduler.state_dict()),
                             (expected_rng, (random.random(), np.random.rand(), torch.rand(3)))):
                    self.assert_nested_equal(a, b)
                for p, digest in hashes.items():
                    self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)

    def test_completed_cycle_continues_with_history(self):
        with tempfile.TemporaryDirectory() as baseline, tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            expected, loaders, decisions = self.make(baseline, cycles=2)
            result = expected.fit(*loaders, interaction=decisions)
            trainer, loaders, decisions = self.make(directory)
            first = trainer.fit(*loaders, interaction=decisions)
            resumed, loaders, decisions = self.make(directory, resume=True, cycles=2)
            actual = resumed.fit(*loaders, interaction=decisions)
            self.assertEqual(len(first.cycles), 1)
            self.assertEqual(len(actual.cycles), 2)
            self.assertEqual(actual.cycles, result.cycles)
            self.assert_nested_equal(expected.model.state_dict(), resumed.model.state_dict())
            self.assert_nested_equal(expected.corrector_optimizer.state_dict(), resumed.corrector_optimizer.state_dict())

    def test_missing_or_incompatible_recovery_rejected(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer, loaders, decisions = self.make(directory, resume=True)
            with self.assertRaisesRegex(FileNotFoundError, 'No recovery'):
                trainer.fit(*loaders, interaction=decisions)
            trainer.config = replace(trainer.config, resume=False, cycles=0)
            trainer.fit(*loaders, interaction=decisions)
            trainer.config = replace(trainer.config, resume=True, num_pred=4)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                trainer.fit(*loaders, interaction=decisions)


if __name__ == '__main__':
    unittest.main()
