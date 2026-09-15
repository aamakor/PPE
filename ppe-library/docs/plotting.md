# Plot your PPE navigation

The plot shows a **black initial point**, **blue predictor segments**, and
**red corrector segments**. Each predictor starts from the previous corrector.
The title shows the current cycle and preference; axes use your objective names.

## Plot during training

Install the plotting dependencies from the source checkout root:

```bash
python -m pip install -e './ppe-library[plot]'
```

From the directory containing your experiment file:

```bash
ppe run my_experiment:build --plot
```

The initial point appears before the first preference prompt. The view updates
after each completed predictor–corrector cycle. Enter preferences in the training
terminal. QtAgg is tried first, then TkAgg; WebAgg is selected automatically only
when neither desktop backend is usable. Follow the terminal prompts to finish.

The CLI saves `results/navigation-3d.png` and `results/navigation.json` in your
configured output directory. Rejected trial steps are not plotted.

## More than three objectives

The number of trained objectives comes from your `Objectives` names and losses.
Plot axes are only a projection. For a five-objective problem:

```bash
ppe run my_experiment:build --plot --plot-objectives 1 3 5
```

All five objectives still participate in training and appear in the recorded loss
vectors. The plot displays only objectives 1, 3 and 5, using one-based indices.
Without this option, the default axes are 1, 2 and 3. See the
[five-objective setup](guide.html#many-objectives) for model and target alignment.

## Plot a saved run

Download [plot_navigation.py](assets/plot_navigation.py) to your project directory:

```bash
python plot_navigation.py RUN/results/navigation.json --output trajectory.png

# Another projection from a problem with at least five objectives:
python plot_navigation.py RUN/results/navigation.json --objectives 1 3 5

# Display test losses instead of validation losses:
python plot_navigation.py RUN/results/navigation.json --split test

# Explicit 2D projection:
python plot_navigation.py RUN/results/navigation.json --objectives 1 2
```

These commands save figures without requiring a display. The default is a 3D
validation-loss view. Test-loss plots do not change the validation-based training
decisions. For interactive inspection on a configured display, add `--show`.

Current result files include the initial point, objective names and completed
cycles. Older files without the initial point omit the first connecting segment
and report that omission. To restore it, pass `--initial` followed by your **full**
initial loss vector for the selected split, including objectives not displayed.

## Use plots from Python

After constructing your trainer and data loaders as described in the
[user guide](guide.html#dataset):

```python
from ppe import TerminalInteraction
from ppe.plotting import NavigationPlot

plot = NavigationPlot(objectives=(1, 2, 3), live=True)
try:
    result = trainer.fit(
        train_loader, val_loader, test_loader,
        interaction=TerminalInteraction(input_fn=plot.read_input),
        on_initial=plot.on_initial,
        on_cycle=plot.on_cycle,
    )
    plot.save(result.output_dir / "results/navigation-3d.png")
    plot.show()
finally:
    plot.close()
```

Keep both callbacks to include the initial point and all completed cycles.
`plot.read_input` allows the display to remain responsive while preferences are
entered. Use this live interface in a normal Python script. To only save a figure,
set `live=False`, omit `input_fn=plot.read_input`, and omit `plot.show()`.

Plotting does not change optimizer updates, constraints or checkpoint selection.
