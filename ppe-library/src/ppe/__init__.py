"""Preference Pareto Exploration for user-defined PyTorch datasets."""
from .api import PPE, PPEConfig
from .objectives import Objectives
from .interaction import Interaction, TerminalInteraction, ScriptedInteraction, NavigationStopped, get_preference
from .runtime import RunResult

__version__ = "0.1.0a1"
__all__ = ["PPE", "PPEConfig", "Objectives", "Interaction", "TerminalInteraction",
           "ScriptedInteraction", "NavigationStopped", "RunResult", "get_preference"]
