import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ppe import PPE, PPEConfig, Objectives, ScriptedInteraction, TerminalInteraction, get_preference
from ppe import _numerics as core
from ppe.demo import make_demo
from ppe.runtime import CheckpointModel

PACKAGE = Path(__file__).resolve().parents[1]
RESEARCH = PACKAGE.parent / "src/function.py"


@unittest.skipUnless(RESEARCH.exists(), "Research parity checks require the original repository")
class ResearchParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load original definitions, excluding imports and their dataset/seed side effects.
        source = ast.parse(RESEARCH.read_text())
        cls.original_ast = {n.name: n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        definitions = list(cls.original_ast.values())
        namespace = dict(vars(core), device=torch.device("cpu"))
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(RESEARCH), "exec"), namespace)
        cls.original = namespace

    def test_unchanged_mathematical_definitions(self):
        current = ast.parse(Path(core.__file__).read_text())
        current = {n.name: n for n in current.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ("_min_norm_element_from2", "_min_norm_2d", "_projection2simplex", "_next_point",
                     "find_min_norm_element", "compute_alpha", "flatten_grads", "task_grads_shared",
                     "row_normalize_G", "beta_basis", "solve_theta_nnls", "proj_general",
                     "check_combinations", "assign_grad", "is_dominated", "is_dom", "is_not_dominated",
                     "find_min_mean", "save_or_copy"):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(self.original_ast[name]), ast.dump(current[name]))

    def test_source_files_unchanged(self):
        for name, digest in json.loads((PACKAGE / "source_manifest.json").read_text()).items():
            self.assertEqual(hashlib.sha256((PACKAGE.parent / name).read_bytes()).hexdigest(), digest)

    def test_classification_gradients_hvp_and_update(self):
        torch.manual_seed(12)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 6)
            def forward(self, x):
                return self.layer(x).split(2, dim=1)
        original, adapted = Model(), Model()
        adapted.load_state_dict(original.state_dict())
        x = torch.randn(8, 2)
        y = torch.randint(0, 2, (8, 3))
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        loss = torch.nn.CrossEntropyLoss()
        objectives = Objectives.classification(("a", "b", "c"))
        for model, criterion, function in ((original, loss, self.original["compute_grads"]),
                                            (adapted, objectives, core.compute_grads)):
            grads = function(model, criterion, loader, iter(loader))
            if model is original:
                reference = grads
            else:
                torch.testing.assert_close(grads, reference, rtol=0, atol=0)
        alpha = core.compute_alpha(reference)
        operators = [self.original["HVPLinearOperator"](loader, original, loss),
                     core.HVPLinearOperator(loader, adapted, objectives)]
        products = []
        for op in operators:
            with op.init(alpha) as linear:
                products.append(linear @ np.ones(op.shape[0]))
        np.testing.assert_array_equal(*products)
        for model, criterion, fn in ((original, loss, self.original["mgda_optimize"]),
                                    (adapted, objectives, core.mgda_optimize)):
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            fn(model, x, y, criterion, core.compute_alpha, opt)
        for left, right in zip(original.parameters(), adapted.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        expected = self.original["evaluate"](original, loader, 3, loss)
        actual = core.evaluate(adapted, loader, 3, objectives)
        np.testing.assert_array_equal(actual, expected)
        # Retain torch.topk's tie behavior, which can differ from argmax.
        tied = torch.zeros(8, 4)
        expected_top1 = self.original["topk_accuracies"](tied, y[:, 0])[0]
        self.assertEqual(objectives.metric(0, tied, y), expected_top1)

    def test_archive_arity_preserves_selection(self):
        entries = [(np.array(v), str(i), np.array([0.2, 0.3, 0.5]), 0.9)
                   for i, v in enumerate(([1, 2, 3], [2, 3, 4], [2, 1, 3], [1, 2, 3]))]
        for pref in (None, np.array([-1., -1., 0.])):
            expected = self.original["prune_archive"](entries, typePred=pref is not None, pref=pref)
            actual = core.prune_archive([e[:3] for e in entries], typePred=pref is not None, pref=pref)
            self.assertEqual([e[1] for e in expected], [e[1] for e in actual])


class PackageIntegration(unittest.TestCase):
    def test_preference_entry_formats(self):
        for n, responses, expected in (
            (3, ["13", "0.7 0.3"], [-0.7, 0, -0.3]),
            (5, ["24", "0.2 0.8"], [0, -0.2, 0, -0.8, 0]),
            (12, ["1, 10", "0.6 0.4"], [-0.6] + [0] * 8 + [-0.4, 0, 0]),
        ):
            with self.subTest(n=n), patch("builtins.input", side_effect=responses):
                np.testing.assert_array_equal(get_preference(n), expected)
        with patch("builtins.input", return_value="-0.7 0 -0.3"):
            np.testing.assert_array_equal(
                TerminalInteraction(preference_format="signed").preference(3), [-0.7, 0, -0.3])

    def test_invalid_preference_reprompts(self):
        with patch("builtins.input", side_effect=["4", "13", "0.5", "13", "0.7 0.3"]), \
                contextlib.redirect_stdout(io.StringIO()):
            np.testing.assert_array_equal(get_preference(3), [-0.7, 0, -0.3])

    def test_predictor_action_can_keep_or_replace_preference(self):
        pref = np.array([-1., 0., 0.])
        with patch("builtins.input", side_effect=["1", "y", "20"]), \
                contextlib.redirect_stdout(io.StringIO()):
            action, actual, budget = TerminalInteraction().action(pref, 10)
            self.assertEqual(action, "rerun_predictor")
            self.assertIs(actual, pref)
            self.assertEqual(budget, 20)
        with patch("builtins.input", side_effect=["0", "n", "0.7 0.3 0"]), \
                contextlib.redirect_stdout(io.StringIO()):
            action, actual, budget = TerminalInteraction().action(pref, 10)
            self.assertEqual(action, "change_pref")
            np.testing.assert_array_equal(actual, [-0.7, -0.3, 0])
            self.assertIsNone(budget)

    def test_default_prompts_after_initialization_and_completed_cycle(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer, loaders, _ = make_demo(directory)
            self.assertIsNone(PPEConfig(directory).cycles)
            trainer.config = replace(trainer.config, cycles=PPEConfig(directory).cycles)
            prompts = []
            answers = iter(["1", "1", "q"])

            def answer(prompt):
                prompts.append(prompt)
                # These are written only after initialization has completed.
                self.assertTrue((Path(directory) / "checkpoints/alpha_2.npy").exists())
                self.assertTrue((Path(directory) / "results/info_cen_init.txt").exists())
                if len(prompts) == 3:
                    saved = json.loads((Path(directory) / "results/navigation.json").read_text())
                    self.assertEqual(len(saved["cycles"]), 1)
                return next(answers)

            with patch("builtins.input", side_effect=answer):
                result = trainer.fit(*loaders)
            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.reason, "Stopped by user.")
            self.assertEqual(len(result.cycles), 1)
            self.assertIn("Select objectives", prompts[0])
            self.assertIn("Enter 1 preference values", prompts[1])
            self.assertIn("Select objectives", prompts[2])

    def test_full_cpu_cycle_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer, loaders, decisions = make_demo(directory)
            result = trainer.fit(*loaders, interaction=decisions)
            self.assertEqual(result.status, "completed")
            self.assertEqual(len(result.cycles), 1)
            self.assertLess(result.cycles[0]["predictor"][0], 0.325)
            result_file = Path(directory) / "results/navigation.json"
            self.assertEqual(json.loads(result_file.read_text())["cycles"], result.cycles)
            checkpoint = Path(directory) / "checkpoints/model_corr_2_custom.pth"
            state = torch.load(checkpoint, weights_only=True)
            for name, tensor in trainer.model.state_dict().items():
                torch.testing.assert_close(state[name], tensor)
            self.assertTrue((Path(directory) / "results/first_result_cen0.pkl").exists())
            with self.assertRaises(FileExistsError):
                trainer.fit(*loaders, interaction=decisions)

    def test_script_exhaustion_and_initial_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer, loaders, _ = make_demo(directory)
            result = trainer.fit(*loaders, interaction=ScriptedInteraction([]))
            self.assertEqual(result.status, "stopped")
            checkpoint = Path(directory) / "checkpoints/model_init_2_custom.pth"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            from dataclasses import replace
            trainer.config = replace(trainer.config, resume=True, cycles=0)
            result = trainer.fit(*loaders)
            self.assertEqual(result.status, "completed")
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)

    def test_input_contract(self):
        with self.assertRaises(ValueError):
            Objectives.classification(("a", "b"))
        with self.assertRaises(ValueError):
            PPEConfig("unused", num_pred=1)
        for values in ([1, 0, 0], [0, 0, 0], [-1, 0, 0]):
            np.testing.assert_array_equal(ScriptedInteraction([values]).preference(3), values)
        for values in ([1, 0], [float("nan"), 0, 0], [float("inf"), 0, 0]):
            with self.assertRaises(ValueError):
                ScriptedInteraction([values]).preference(3)
        with tempfile.TemporaryDirectory() as directory:
            trainer, loaders, _ = make_demo(directory)
            small = DataLoader(loaders[0].dataset, batch_size=16)
            with self.assertRaisesRegex(ValueError, ">= 4"):
                trainer.fit(small, *loaders[1:])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
