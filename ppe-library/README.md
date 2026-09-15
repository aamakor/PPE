# PPE · Preference Pareto Exploration

PPE is an installable predictor–corrector training library by Augustina C. Amakor.
Bring a PyTorch model, three data loaders, and your objective losses. No MultiMNIST,
UCI, downloaded checkpoint, or repository-specific dataset import is required.

The algorithm, implementation, Python API, and documentation website are developed
and maintained by Augustina C. Amakor. See the [user guide](docs/guide.html)
for supported behavior and dataset customization.

## Install and run

From the PPE repository root, preferably in a Python 3.11 virtual environment:

```bash
python -m pip install -e ./ppe-library
ppe demo --interactive --output runs/interactive-demo
```

For a scripted one-cycle example that does not ask for input:

```bash
ppe demo --output runs/first-demo
```

With `--interactive`, the terminal first asks which objectives to select, then their preference values,
as in the PPE interface. For example, select `13` and enter `0.7 0.3`
to construct `[-0.7, 0, -0.3]`. Enter `q` at a prompt to finish.

Only the interactive command asks these questions. Installing with `-e` keeps
your local checkout's updates available to the installed package.

The alpha package targets Python 3.11–3.12 and NumPy 1.26.x.

The demo runs initialization, prediction, and correction on a tiny three-objective
quadratic regression problem on CPU. It downloads nothing. Use a fresh output path
for each run. `ppe demo --cycles 0 --output runs/init-only` runs only initialization.
The distribution name is `ppe-navigation`; the import name is `ppe`. This package
has not been published to PyPI: bare `pip install ppe-navigation` is not a release instruction.

## Use your dataset

Copy [custom_dataset.py](examples/custom_dataset.py) to **your own project** as
`my_experiment.py`. This is the file you customize; do not edit PPE's `demo.py`.
The starter supplies a complete `build()` function returning:

```python
return trainer, (train_loader, val_loader, test_loader), TerminalInteraction()
```

1. Set the starter's `OBJECTIVE_NAMES`, `CLASS_COUNTS`, and `INPUT_DIM`.
2. Replace `make_loaders()` with your PyTorch datasets and train/validation/test
   loaders, or set `DATA_DIR` to a folder with `train.npz`, `val.npz`, and `test.npz`.
   Each file uses `inputs` and `targets` arrays. Leave `DATA_DIR = None` to try
   generated data first.
3. Adapt `MultiTaskModel`. Return one output per objective in the names' order.
4. Use `Objectives.classification(names)` for classification, or
   `Objectives(names, loss)` with your differentiable scalar mean losses.
5. Choose device, optimizers and `PPEConfig` in `build()`, including a fresh
   output directory. The CLI calls `fit()`; do not also call it inside `build()`.

From the directory containing your file:

```bash
ppe run my_experiment:build
```

Custom factories already request preferences; `--interactive` is only needed for
`ppe demo`. You can also run the starter directly with `python my_experiment.py`.
See the [dataset guide](docs/guide.html#dataset) for data shapes, the optional NumPy
file layout, and non-classification objectives.

## Five or more objectives

PPE derives the count from the objective names. In the downloaded starter, set:

```python
OBJECTIVE_NAMES = ("task A", "task B", "task C", "task D", "task E")
CLASS_COUNTS = (2, 3, 2, 4, 2)
OUTPUT_DIR = "runs/five-objectives"
```

The starter creates five heads and five classification losses. Supply five target
columns with your own data; generated data adapts automatically. Keep names,
outputs, losses and targets aligned. A custom loss receives a zero-based objective
index. There is no separate `nobj` setting to update.

For a live projection of objectives 1, 3 and 5:

```bash
ppe run my_experiment:build --plot --plot-objectives 1 3 5
```

Install the plotting extra below before using `--plot`. All five objectives remain
part of training; selecting three plot axes does not remove objectives. See the
[five-objective guide](docs/guide.html#many-objectives) for preferences and contracts.

Normal training prompts after initialization and again after each completed
predictor–corrector cycle. `PPEConfig.cycles` defaults to `None`, preserving ongoing
interaction; set `cycles=1` to request just one cycle. The standalone
`from ppe import get_preference` exposes the same objective/value prompts.
The optional `TerminalInteraction(preference_format="signed")` accepts full signed
vectors. Scripted decisions and custom interfaces are described in the [guide](docs/guide.html).

## What happens after initialization?

1. PPE evaluates the initial point, then asks for objective numbers and their values.
2. The predictor uses that preference; its existing checks may request another
   preference, a step size, or a predictor rerun.
3. The corrector runs. If the front-change condition is reached, PPE asks
   whether to continue with the new front.
4. After a completed cycle, PPE requests preferences again until you enter `q`
   or reach an explicitly configured cycle limit.

Example for three objectives:

```text
Select objectives (e.g. 1, 24, 135) [q to stop]: 13
Enter 2 preference values [q to stop]: 0.7 0.3
```

The terminal negates each entered value at the selected index. Current input
validation requires the correct vector length and finite values; it accepts
either sign. An internal negative entry selects an objective for reduction.
With ten or more objectives, use space- or comma-separated indices, such as
`1, 10`. Full signed vectors supplied to `ScriptedInteraction` or
`TerminalInteraction(preference_format="signed")` are used without negation.

| Interface or setting | Behavior |
| --- | --- |
| `trainer.fit(...)` with default configuration | Prompts after initialization and each completed cycle. |
| `ppe demo --interactive` | Interactive quadratic example; continues until stopped. |
| `ppe demo` | Scripted quadratic example; one completed cycle. |
| `cycles=None` | Ongoing navigation; the library default. |
| `cycles=1` / `--cycles 1` | Finish after one completed cycle. |
| `cycles=0` / `--cycles 0` | Initialization only; no preference prompt. |

For a standalone prompt:

```python
from ppe import get_preference, NavigationStopped

try:
    pref = get_preference(3)
except NavigationStopped:
    pref = None  # The user entered q or closed the terminal input.
```

Calling `trainer.fit(...)` handles that stop internally and returns a
`RunResult` with `status="stopped"` and a reason. Continue a saved run using
`resume=True`, as described below.

## Archive snapshots and resume

Each archived point has a unique `.pth` model snapshot and matching `.json`
record in `checkpoints/archive/`. The record contains validation losses, alpha
weights, phase, and the snapshot filename. Load a model snapshot with
`torch.load(path, weights_only=True)` followed by `model.load_state_dict(...)`.

To continue a run, recreate the same trainer and loaders, then set:

```python
from dataclasses import replace

trainer, loaders, interaction = build()
trainer.config = replace(trainer.config, resume=True, cycles=5)
result = trainer.fit(*loaders, interaction=interaction)
```

`checkpoints/recovery.pt` restores the model, both optimizers, schedulers, random
state, loader generators, completed history, and navigation state. An interrupted
initialization or cycle replays from its saved start using recorded decisions.
Use the same data and deterministic setup for reproducible replay. `cycles=5`
means five cycles in total, including completed cycles; `None` continues interactively.
See the [resume guide](docs/guide.html#resume) for the full workflow.

## 3D navigation plots

Live plotting tries **QtAgg, then TkAgg**, and automatically starts **WebAgg only
if neither desktop backend can create a figure**:

```bash
python -m pip install -e './ppe-library[plot]'
ppe demo --interactive --plot --output runs/auto-demo
```

Use `--plot` to select the display automatically. Both `plot` and `web` extras
include the fallback dependencies. Enter preferences in the training terminal and
follow its prompts to finish.

The initial point is black, predictor segments are blue, and corrector segments
are red. The plot updates after initialization and each completed cycle. For a
saved run, the plotting example now uses **three axes by default**:

```bash
python ppe-library/examples/plot_navigation.py runs/interactive-3d/results/navigation.json --output trajectory-3d.png
```

See the [plotting guide](docs/plotting.md) for live Python callbacks, headless
export, other objective projections, and older JSON files lacking an initial point.

## User documentation

The [website](docs/index.html) and [user guide](docs/guide.html) explain installation,
dataset adaptation, custom objectives, preferences, results, and supported behavior.
Training runs in your Python environment; the website provides documentation and
downloadable examples.

## Existing experiments, toy examples and data

The PPE repository includes separate mathematical examples and DTLZ 1–7
scripts in [`toyEx_code`](https://github.com/aamakor/PPE/tree/master/toyEx_code).
Follow the setup instructions supplied with each example. The included quadratic
demo demonstrates the training API.

The repository README supplies the [Zenodo dataset DOI](https://doi.org/10.5281/zenodo.20623056)
for MultiMNIST and UCI Census experiments. Use these datasets to run the supplied
experiments, or connect your own data through the Python API.

## Cite

Citation supplied by the repository README:

```bibtex
@inbook{Amakor2026,
  title = {Interactive Pareto Navigation for Deep Multi-task Learning},
  ISBN = {9783032376671},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-032-37667-1_37},
  DOI = {10.1007/978-3-032-37667-1_37},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
  publisher = {Springer Nature Switzerland},
  author = {Amakor,  Augustina C. and Sonntag,  Konstantin and Peitz,  Sebastian},
  year = {2026},
  month = Sept,
  pages = {652–668}
}
```

MIT license, copyright © 2026 Augustina C. Amakor.
