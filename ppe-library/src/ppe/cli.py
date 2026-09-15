"""Download-free demo and a factory entry point for user datasets."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="PPE: Preference Pareto Exploration")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the small quadratic example without downloading data")
    demo.add_argument("--output", default="runs/quadratic")
    demo.add_argument("--cycles", type=int, default=None,
                      help="Completed cycles (default: 1 scripted, unlimited interactive)")
    demo.add_argument("--interactive", action="store_true",
                      help="Ask for objectives and preferences after initialization and each cycle")
    demo.add_argument("--device", default="cpu")
    custom = commands.add_parser("run", help="Run a factory from an importable Python module")
    custom.add_argument("factory", help="module:function returning (PPE, (train, val, test), interaction)")
    for command in (demo, custom):
        command.add_argument("--plot", action="store_true", help="Show live 3D navigation (tries QtAgg, TkAgg, then WebAgg)")
        command.add_argument("--plot-backend", help="Override automatic backend selection, e.g. WebAgg, TkAgg or QtAgg")
        command.add_argument("--webagg-port", type=int, default=None, help="Browser server port (default 8988; 0 selects an available port)")
        command.add_argument("--plot-objectives", type=int, nargs=3, default=[1, 2, 3],
                             metavar=("X", "Y", "Z"), help="Three one-based objective indices")
    args = parser.parse_args()
    if (args.plot_backend is not None or args.webagg_port is not None) and not args.plot:
        parser.error("--plot-backend and --webagg-port require --plot.")
    if args.command == "demo":
        from .demo import make_demo
        trainer, loaders, interaction = make_demo(
            args.output, cycles=args.cycles if args.cycles is not None else 1, device=args.device)
        if args.interactive:
            from dataclasses import replace
            from .interaction import TerminalInteraction
            trainer.config = replace(trainer.config, cycles=args.cycles)
            interaction = TerminalInteraction()
    else:
        module, separator, name = args.factory.partition(":")
        if not separator:
            parser.error("Factory must use module:function syntax.")
        # Console entry points start in the installation's bin directory, unlike
        # python -m. Make the visitor's experiment and sibling modules importable.
        sys.path.insert(0, str(Path.cwd()))
        trainer, loaders, interaction = getattr(importlib.import_module(module), name)()
    plot = None
    callbacks = {}
    if args.plot:
        from .plotting import NavigationPlot
        from .interaction import TerminalInteraction
        try:
            plot = NavigationPlot(objectives=args.plot_objectives, live=True,
                                  backend=args.plot_backend, webagg_port=args.webagg_port)
            plot._labels(trainer.objectives.names)
        except (ImportError, ValueError, OSError, RuntimeError) as exc:
            if plot is not None:
                plot.close()
            parser.error(str(exc))
        if interaction is None or isinstance(interaction, TerminalInteraction):
            interaction = TerminalInteraction(
                preference_format=getattr(interaction, "preference_format", "objectives"),
                input_fn=plot.read_input)
        callbacks = {"on_initial": plot.on_initial, "on_cycle": plot.on_cycle}
    try:
        result = trainer.fit(*loaders, interaction=interaction, **callbacks)
        if plot:
            plot.save(result.output_dir / "results/navigation-3d.png")
            print("Training finished. The final plot remains available." if plot.is_webagg else
                  "Training finished. Close the plot window to exit.")
            plot.show()
    finally:
        if plot:
            plot.close()
    print(json.dumps({"status": result.status, "reason": result.reason,
                      "cycles": len(result.cycles), "output_dir": str(result.output_dir)}))


if __name__ == "__main__":
    main()
