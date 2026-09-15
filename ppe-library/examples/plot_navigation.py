"""Plot initial → predictor → corrector navigation in 3D by default.

From the repository root, after installing './ppe-library[plot]':
python ppe-library/examples/plot_navigation.py RUN/results/navigation.json --output trajectory-3d.png
Add --show for a rotatable GUI, --objectives 1 3 5 for another 3D projection,
or --objectives 1 2 for an explicit 2D view. See docs/plotting.md for live training.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot PPE navigation in 3D (or explicitly select two objectives)")
    parser.add_argument("result", help="Saved navigation.json")
    parser.add_argument("--objectives", type=int, nargs="+", default=[1, 2, 3], metavar="INDEX")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--initial", type=float, nargs="+", help="Full initial loss vector for older result files")
    parser.add_argument("--output", default="trajectory-3d.png")
    parser.add_argument("--show", action="store_true", help="Open a rotatable plot in a GUI backend")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    from ppe.plotting import plot_navigation
    try:
        record = json.loads(Path(args.result).read_text())
        plot = plot_navigation(record, objectives=args.objectives, split=args.split, initial=args.initial)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    try:
        plot.save(args.output)
        print(f"Saved {args.output}")
        if args.show:
            plot.show()
    finally:
        plot.close()


if __name__ == "__main__":
    main()
