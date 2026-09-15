"""Explicit decision interfaces for interactive and scripted navigation."""
from collections import deque
from typing import Protocol
import numpy as np


class NavigationStopped(Exception):
    """A caller stopped navigation or a script needs another decision."""


class Interaction(Protocol):
    def preference(self, n_objectives: int): ...
    def step_size(self) -> float: ...
    def accept_front(self) -> bool: ...
    def action(self, preference, max_runs: int): ...


def validate_preference(value, n):
    pref = np.asarray(value, dtype=float)
    if pref.shape != (n,) or not np.isfinite(pref).all():
        raise ValueError(f"Preference must contain {n} finite values.")
    #if (pref > 0).any() or not (pref < 0).any():
    #    raise ValueError("Use negative weights for objectives to reduce, zero for unselected objectives.")
    return pref


class TerminalInteraction:
    """Objective-selection/value prompts; q stops at any prompt.

    Set preference_format="signed" to enter a full signed vector instead.
    """
    def __init__(self, *, preference_format="objectives", input_fn=None):
        if preference_format not in ("objectives", "signed"):
            raise ValueError("preference_format must be 'objectives' or 'signed'.")
        self.preference_format = preference_format
        self.input_fn = input_fn

    def _read(self, prompt):
        try:
            reader = self.input_fn if self.input_fn is not None else input
            value = reader(prompt + " [q to stop]: ").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise NavigationStopped("Terminal session stopped.") from exc
        if value.lower() == "q":
            raise NavigationStopped("Stopped by user.")
        return value

    def preference(self, n_objectives):
        while True:
            try:
                if self.preference_format == "signed":
                    return validate_preference([float(v) for v in self._read(
                        f"Enter {n_objectives} signed preference weights").split()], n_objectives)
                prompt = ("Select objectives (e.g. 1, 24, 135)" if n_objectives < 10 else
                          "Select objective numbers separated by spaces or commas (e.g. 1, 10)")
                raw = self._read(prompt)
                if n_objectives >= 10 or "," in raw or " " in raw:
                    numbers = raw.replace(",", " ").split()
                else:
                    numbers = list(raw)
                indices = [int(number) - 1 for number in numbers]
                if not indices or any(i < 0 or i >= n_objectives for i in indices):
                    raise ValueError(f"Select objective numbers from 1 to {n_objectives}.")
                if len(set(indices)) != len(indices):
                    raise ValueError("Select each objective only once.")
                values = [float(v) for v in self._read(
                    f"Enter {len(indices)} preference values").split()]
                if len(values) != len(indices):
                    raise ValueError(f"Expected {len(indices)} values, got {len(values)}")
                #if not np.isfinite(values).all() or any(value < 0 for value in values):
                #    raise ValueError("Enter nonnegative finite preference values.")
                pref = np.zeros(n_objectives)
                for index, value in zip(indices, values):
                    pref[index] = -value
                return validate_preference(pref, n_objectives)
            except ValueError as exc:
                print(exc)

    def step_size(self):
        while True:
            try:
                value = float(self._read("Enter a new positive predictor step size"))
                if np.isfinite(value) and value > 0:
                    return value
            except ValueError:
                pass
            print("Enter a positive finite number.")

    def accept_front(self):
        while True:
            value = self._read("Continue with the new front? y/n").lower()
            if value in ("y", "yes", "n", "no"):
                return value in ("y", "yes")

    def action(self, preference, max_runs):
        print("\nCurrent preference:", preference)
        print("Current max predictor runs:", max_runs)
        while True:
            action = self._read("0: change preference; 1: rerun predictor")
            if action in ("0", "1"):
                break
        while True:
            keep = self._read("Keep current preference? (y/n)").lower()
            if keep in ("y", "n"):
                break
        pref = preference
        if keep == "n":
            while True:
                try:
                    values = [float(v) for v in self._read(
                        f"Enter {len(preference)} new preference values (nonnegative magnitudes)").split()]
                    pref = validate_preference(-np.asarray(values), len(preference))
                    break
                except ValueError as exc:
                    print(exc)
        if action == "0":
            return "change_pref", pref, None
        while True:
            try:
                steps = int(self._read("Maximum predictor runs"))
                if steps > 0:
                    return "rerun_predictor", pref, steps
            except ValueError:
                pass


def get_preference(nobj=3):
    """Prompt for selected objectives and weights, through the PPE interface.

    For three objectives, selecting '13' and entering '0.7 0.3' returns
    array([-0.7, 0.0, -0.3]). Matches the default terminal interaction prompts.
    """
    return TerminalInteraction().preference(nobj)


class ScriptedInteraction:
    """Separate queues for each decision; exhaustion stops, never invents consent.

    preferences=[[-1, 0, 0]], step_sizes=[0.01], accept_fronts=[True],
    actions=[('rerun_predictor', [-1, 0, 0], 10)].
    A rejected direction may request another preference within the same cycle.
    """
    def __init__(self, preferences, *, step_sizes=(), accept_fronts=(), actions=()):
        self.preferences = deque(preferences)
        self.step_sizes = deque(step_sizes)
        self.accept_fronts = deque(accept_fronts)
        self.actions = deque(actions)

    def _next(self, queue, name):
        if not queue:
            raise NavigationStopped(f"Script needs a {name} decision.")
        return queue.popleft()

    def preference(self, n_objectives):
        return validate_preference(self._next(self.preferences, "preference"), n_objectives)

    def step_size(self):
        return self._next(self.step_sizes, "step size")

    def accept_front(self):
        value = self._next(self.accept_fronts, "accept front")
        if not isinstance(value, bool):
            raise ValueError("accept_fronts must contain booleans.")
        return value

    def action(self, preference, max_runs):
        return self._next(self.actions, "predictor action")
