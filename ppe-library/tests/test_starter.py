"""Exercise the same file visitors download with a real five-objective dataset."""
import contextlib
from dataclasses import replace
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class StarterIntegration(unittest.TestCase):
    def test_console_entry_loads_visitor_file_from_working_directory(self):
        root = Path(__file__).resolve().parents[1]
        starter = (root / "docs/assets/custom_dataset.py").read_text()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            project = folder / "visitor-project"
            project.mkdir()
            (project / "my_experiment.py").write_text(starter + '''
from dataclasses import replace
original_build = build
def build():
    trainer, loaders, interaction = original_build()
    trainer.config = replace(trainer.config, num_init=1, cycles=0)
    return trainer, loaders, interaction
''')
            # Mimic pip's console script outside the visitor's directory.
            entry = folder / "console_entry.py"
            entry.write_text("from ppe.cli import main\nmain()\n")
            env = {**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"}
            run = subprocess.run([sys.executable, str(entry), "run", "my_experiment:build"],
                                 cwd=project, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((project / "runs/my-dataset/results/navigation.json").read_text())
            self.assertEqual(len(result["initial"]["validation"]), 3)

    def test_five_objectives_from_three_dataset_files(self):
        path = Path(__file__).resolve().parents[1] / "docs/assets/custom_dataset.py"
        spec = importlib.util.spec_from_file_location("visitor_experiment", path)
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        experiment.OBJECTIVE_NAMES = ("A", "B", "C", "D", "E")
        experiment.CLASS_COUNTS = (2, 3, 2, 4, 2)
        rng = np.random.default_rng(81)
        with tempfile.TemporaryDirectory() as folder:
            experiment.DATA_DIR = Path(folder)
            experiment.OUTPUT_DIR = Path(folder) / "run"
            expected = {}
            for split, n in (("train", 32), ("val", 16), ("test", 8)):
                inputs = rng.normal(size=(n, experiment.INPUT_DIM)).astype("float32")
                targets = np.stack([rng.integers(c, size=n) for c in experiment.CLASS_COUNTS], axis=1)
                np.savez(Path(folder) / f"{split}.npz", inputs=inputs, targets=targets)
                expected[split] = (inputs, targets)
            trainer, loaders, interaction = experiment.build()
            self.assertEqual(trainer.objectives.n_objectives, 5)
            for split, loader in zip(("train", "val", "test"), loaders):
                x, y = loader.dataset.tensors
                np.testing.assert_array_equal(x.numpy(), expected[split][0])
                np.testing.assert_array_equal(y.numpy(), expected[split][1])
            x, y = next(iter(loaders[0]))
            outputs = trainer.model(x)
            self.assertEqual([tuple(out.shape) for out in outputs],
                             [(8, count) for count in experiment.CLASS_COUNTS])
            self.assertEqual(len(y[0]), 5)
            trainer.config = replace(trainer.config, cycles=0, num_init=1)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = trainer.fit(*loaders, interaction=interaction)
            self.assertEqual(len(result.initial["validation"]), 5)
            self.assertTrue(np.isfinite(result.initial["validation"]).all())
            self.assertEqual(tuple(result.objective_names), experiment.OBJECTIVE_NAMES)
            self.assertTrue((result.output_dir / "results/navigation.json").is_file())


if __name__ == "__main__":
    unittest.main()
