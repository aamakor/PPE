"""Optional 3D navigation plots; importing PPE does not import Matplotlib."""
from pathlib import Path
import queue
import threading
import warnings
import numpy as np


def _create_figure(matplotlib, plt, *, live, backend):
    if backend is not None:
        matplotlib.use(backend)
    if not live or backend is not None:
        return plt.figure(figsize=(10, 7))

    # Creating a figure checks both the GUI dependency and access to a display.
    for candidate in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(candidate)
            return plt.figure(figsize=(10, 7))
        except Exception:
            # GUI startup errors include ImportError, TclError and RuntimeError.
            # A failed desktop startup should allow the next backend to try.
            continue
    try:
        matplotlib.use("WebAgg")
        figure = plt.figure(figsize=(10, 7))
    except ImportError as exc:
        raise ImportError("QtAgg and TkAgg are unavailable. Install 'ppe-navigation[plot]' "
                          "to enable the automatic WebAgg fallback.") from exc
    print("QtAgg and TkAgg are unavailable; using WebAgg for live navigation.", flush=True)
    return figure


class NavigationPlot:
    """Initial point and linked predictor/corrector phases in objective space.

    Default axes are objectives 1, 2 and 3 (one-based). Select any three for more
    objectives, or pass two indices for an explicit 2D view. live=True tries
    QtAgg, then TkAgg, then WebAgg unless backend is explicitly supplied.
    Use read_input with TerminalInteraction
    to service display events while a terminal prompt waits for input.
    """
    def __init__(self, *, objectives=(1, 2, 3), split="validation", live=False,
                 backend=None, webagg_port=None):
        self.objectives = tuple(objectives)
        if len(self.objectives) not in (2, 3) or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("Choose two or three different objective indices.")
        if any(type(i) is not int or i < 1 for i in self.objectives):
            raise ValueError("Objective indices are one-based positive integers.")
        if split not in ("validation", "test"):
            raise ValueError("split must be 'validation' or 'test'.")
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        self.split = split
        self.live = live
        self._web_server = None
        self.webagg_port = matplotlib.rcParams["webagg.port"] if webagg_port is None else webagg_port
        if type(self.webagg_port) is not int or not 0 <= self.webagg_port <= 65535:
            raise ValueError("webagg_port must be an integer from 0 to 65535 (0 selects an available port).")
        self.figure = _create_figure(matplotlib, plt, live=live, backend=backend)
        self.is_webagg = matplotlib.get_backend().lower() == "webagg"
        self.ax = self.figure.add_subplot(111, projection="3d" if len(self.objectives) == 3 else None)
        self.figure.subplots_adjust(left=0.08, right=0.8, bottom=0.13, top=0.9)
        if live and not self.is_webagg and self.figure.canvas.required_interactive_framework is None:
            plt.close(self.figure)
            raise ValueError("Live navigation needs WebAgg or a desktop GUI backend such as TkAgg or QtAgg. "
                             "For browser plotting use --plot-backend WebAgg. "
                             "For a headless run, save JSON and plot it later without --show.")
        self.previous = None
        self._predictor_label = False
        self._corrector_label = False
        self._labels()
        if live and self.is_webagg:
            try:
                self._ensure_web_server()
            except Exception:
                plt.close(self.figure)
                raise

    @property
    def url(self):
        return self._web_server.url if self._web_server else None

    def _ensure_web_server(self):
        if self._web_server is None:
            from ._webagg import WebAggServer
            self._web_server = WebAggServer(self.figure, self.webagg_port)

    def _pump_events(self, seconds):
        if self.is_webagg:
            self._ensure_web_server()
            self._web_server.pump(seconds)
        else:
            self.figure.canvas.start_event_loop(seconds)

    def _labels(self, names=None):
        if names and max(self.objectives) > len(names):
            raise ValueError("Selected objective index exceeds the objective count.")
        labels = [f"{names[i - 1] if names else f'Objective {i}'} ({self.split} loss)"
                  for i in self.objectives]
        self.ax.set_xlabel(labels[0], labelpad=12)
        self.ax.set_ylabel(labels[1], labelpad=12)
        if len(labels) == 3:
            self.ax.set_zlabel(labels[2], labelpad=12)

    def _point(self, values):
        vector = np.asarray(values, dtype=float)
        if vector.ndim != 1 or max(self.objectives) > len(vector) or not np.isfinite(vector).all():
            raise ValueError("Loss vectors must be finite and contain every selected objective.")
        return vector[np.array(self.objectives) - 1]

    def on_initial(self, record):
        self._labels(record.get("objective_names"))
        self.previous = self._point(record[self.split])
        options = {"depthshade": False} if len(self.objectives) == 3 else {}
        self.ax.scatter(*self.previous, color="black", s=60, label="Initial point", **options)
        self.ax.set_title("Initial point — choose a preference")
        self._refresh()

    def on_cycle(self, record):
        prefix = "test_" if self.split == "test" else ""
        predictor = self._point(record[prefix + "predictor"])
        corrector = self._point(record[prefix + "corrector"])
        if self.previous is not None:
            self._segment(self.previous, predictor, "blue",
                          "Predictor" if not self._predictor_label else "_nolegend_")
            self._predictor_label = True
        self._segment(predictor, corrector, "red",
                      "Corrector" if not self._corrector_label else "_nolegend_")
        self._corrector_label = True
        cycle = record["cycle"]
        self.ax.text(*corrector, f"  C{cycle}", fontsize=9)
        pref = np.asarray(record["preference"])
        self.ax.set_title(f"Cycle {cycle} — preference {np.array2string(pref, precision=3)}")
        self.previous = corrector.copy()
        self._refresh()

    def _segment(self, start, end, color, label):
        points = np.stack((start, end))
        self.ax.plot(*points.T, "-o", color=color, linewidth=2, markersize=4, label=label)

    def _refresh(self):
        self.ax.legend(loc="best")
        if self.live and self.plt.fignum_exists(self.figure.number):
            if not self.is_webagg:
                self.plt.show(block=False)
            self.figure.canvas.draw_idle()
            self._pump_events(0.05)

    def read_input(self, prompt):
        """Read terminal input while processing GUI events on the main thread."""
        if not self.live or not self.plt.fignum_exists(self.figure.number):
            return input(prompt)
        answers = queue.Queue()

        def read():
            try:
                answers.put((True, input(prompt)))
            except BaseException as exc:
                answers.put((False, exc))

        threading.Thread(target=read, daemon=True).start()
        while True:
            try:
                success, answer = answers.get_nowait()
            except queue.Empty:
                self._pump_events(0.05)
                continue
            if not success:
                raise answer
            return answer

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = [self.ax.xaxis.label, self.ax.yaxis.label]
        if len(self.objectives) == 3:
            labels.append(self.ax.zaxis.label)
        self.figure.canvas.draw()
        self.figure.savefig(path, dpi=160, bbox_inches="tight", bbox_extra_artists=labels)

    def show(self):
        if self.is_webagg:
            self._ensure_web_server()
            was_live = self.live
            self.live = True
            try:
                self.read_input("WebAgg remains available. Press Enter here to finish: ")
            except (EOFError, KeyboardInterrupt):
                pass
            finally:
                self.live = was_live
        else:
            self.plt.show(block=True)

    def close(self):
        if self._web_server:
            self._web_server.close()
        self.plt.close(self.figure)


def plot_navigation(record, *, objectives=(1, 2, 3), split="validation", initial=None):
    """Plot saved JSON, including older files without an initial point.

    initial optionally supplies the full initial loss vector for the selected split.
    Older files without it omit only the unavailable first predictor segment.
    """
    initial_record = record.get("initial")
    if initial is not None:
        initial_record = {split: initial}
    if initial_record is None and not record.get("cycles"):
        raise ValueError("No initial point or completed cycles are available to plot.")
    plot = NavigationPlot(objectives=objectives, split=split)
    try:
        plot._labels(record.get("objective_names"))
        if initial_record is not None:
            plot.on_initial({**initial_record, "objective_names": record.get("objective_names")})
        else:
            warnings.warn("This older result has no initial point; the first predictor segment is omitted. "
                          "Supply --initial with the full initial loss vector to include it.", stacklevel=2)
        for cycle in record.get("cycles", []):
            plot.on_cycle(cycle)
        return plot
    except Exception:
        plot.close()
        raise
