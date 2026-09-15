import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from ppe.demo import make_demo
from ppe.plotting import NavigationPlot, plot_navigation
from ppe import TerminalInteraction


def record():
    return {"objective_names": ["a", "b", "c", "d", "e"],
            "initial": {"validation": [5, 4, 3, 2, 1], "test": [6, 5, 4, 3, 2]},
            "cycles": [
                {"cycle": 1, "preference": [-1, 0, 0, 0, 0],
                 "predictor": [4, 5, 3, 2, 1], "corrector": [3, 4, 2, 1, 0],
                 "test_predictor": [5, 6, 4, 3, 2], "test_corrector": [4, 5, 3, 2, 1]},
                {"cycle": 2, "preference": [0, -1, 0, 0, 0],
                 "predictor": [4, 3, 2, 1, 0], "corrector": [3, 2, 1, 0, 0],
                 "test_predictor": [5, 4, 3, 2, 1], "test_corrector": [4, 3, 2, 1, 1]}]}


class PlottingTests(unittest.TestCase):
    def test_3d_phase_connections_and_axis_selection(self):
        payload = record()
        for objectives, split in (((1, 2, 3), "validation"), ((1, 3, 5), "test")):
            with self.subTest(objectives=objectives):
                plot = plot_navigation(payload, objectives=objectives, split=split)
                self.addCleanup(plot.close)
                self.assertEqual(plot.ax.name, "3d")
                self.assertEqual([line.get_color() for line in plot.ax.lines], ["blue", "red", "blue", "red"])
                prefix = "test_" if split == "test" else ""
                points = [payload["initial"][split]]
                for cycle in payload["cycles"]:
                    points += [cycle[prefix + "predictor"], cycle[prefix + "corrector"]]
                projected = np.array(points)[:, np.array(objectives) - 1]
                for i, line in enumerate(plot.ax.lines):
                    np.testing.assert_array_equal(np.array(line.get_data_3d()).T, projected[i:i + 2])
                self.assertIn(payload["objective_names"][objectives[2] - 1], plot.ax.get_zlabel())

    def test_initial_only_and_old_results(self):
        payload = record()
        initial_only = {**payload, "cycles": []}
        plot = plot_navigation(initial_only)
        self.addCleanup(plot.close)
        self.assertEqual(len(plot.ax.collections), 1)
        self.assertEqual(len(plot.ax.lines), 0)
        old = {"cycles": payload["cycles"]}
        with self.assertWarnsRegex(UserWarning, "no initial point"):
            plot = plot_navigation(old)
        self.addCleanup(plot.close)
        self.assertEqual(len(plot.ax.lines), 3)
        restored = plot_navigation(old, initial=payload["initial"]["validation"])
        self.addCleanup(restored.close)
        self.assertEqual(len(restored.ax.lines), 4)

    def test_explicit_2d_and_bad_axes(self):
        plot = plot_navigation(record(), objectives=(1, 5))
        self.addCleanup(plot.close)
        self.assertEqual(plot.ax.name, "rectilinear")
        for indices in ((1,), (1, 1, 2), (0, 1, 2), (1, 2, 6)):
            with self.assertRaises(ValueError):
                plot_navigation(record(), objectives=indices)

    def test_callbacks_preserve_training_and_save_initial_before_prompt(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            reference, loaders, decisions = make_demo(Path(folder) / "reference")
            expected = reference.fit(*loaders, interaction=decisions)
            actual, loaders, decisions = make_demo(Path(folder) / "plotted")
            plot = NavigationPlot()
            self.addCleanup(plot.close)
            seen = []

            def initial_callback(initial):
                self.assertEqual(len(plot.ax.lines), 0)
                seen.append("initial")
                plot.on_initial(initial)

            original_preference = decisions.preference
            def preference(n):
                self.assertEqual(seen, ["initial"])
                return original_preference(n)
            decisions.preference = preference
            result = actual.fit(*loaders, interaction=decisions,
                                on_initial=initial_callback, on_cycle=plot.on_cycle)
            self.assertEqual(expected.cycles, result.cycles)
            for a, b in zip(reference.model.parameters(), actual.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            saved = json.loads((result.output_dir / "results/navigation.json").read_text())
            self.assertEqual(saved["initial"], result.initial)
            self.assertEqual(saved["objective_names"], list(actual.objectives.names))
            self.assertEqual(len(plot.ax.lines), 2)
            plot.save(Path(folder) / "trajectory.png")
            self.assertGreater((Path(folder) / "trajectory.png").stat().st_size, 1000)

    def test_terminal_wait_pumps_gui_events(self):
        plot = NavigationPlot()
        self.addCleanup(plot.close)
        plot.live = True  # Test event pumping without requiring a display server.
        release = threading.Event()
        def answer(prompt):
            if not release.wait(2):
                raise RuntimeError("The GUI event loop was not serviced.")
            return "q"
        with patch("builtins.input", side_effect=answer), \
                patch.object(plot.figure.canvas, "start_event_loop", side_effect=lambda _: release.set()) as pump:
            self.assertEqual(plot.read_input("prompt"), "q")
            pump.assert_called()


if __name__ == "__main__":
    unittest.main()
