# PPE
This respository contains the Preference Pareto Exploration algorithm using the predictor-corrector approach of Continuation method described in our paper "**Interactive Pareto navigation for deep multi-task learning**"

<table style="width: 50%; border: none;">
  <tr style="border: none;">
    <td style="width: 33.33%; border: none; text-align: center;">
      <img src="./toyEx_code/images/video_dtlz1.gif" style="width: 100%;" alt="Plot 1">
    </td>
    <td style="width: 33.33%; border: none; text-align: center;">
      <img src="./toyEx_code/images/video_dtlz3.gif" style="width: 100%;" alt="Plot 2">
    </td>
    <td style="width: 33.33%; border: none; text-align: center;">
      <img src="./toyEx_code/images/video_dtlz5.gif" style="width: 100%;" alt="Plot 3">
    </td>
  </tr>
</table>


## Installable library and interactive navigation

PPE’s algorithm, implementation, Python API, and documentation website are
developed and maintained by Augustina C. Amakor.

Use [`ppe-library/`](ppe-library/README.md) to supply your own PyTorch model,
data loaders and objective losses without editing the PPE training code.
From this repository's root, install it and try the interactive CPU example:

```bash
python -m pip install -e ./ppe-library
ppe demo --interactive --output runs/interactive-demo
```

After initialization finishes, select objectives (for example, `13`), then enter
their preference values (`0.7 0.3`). PPE constructs `[-0.7, 0, -0.3]`, runs the
predictor and corrector, and asks for preferences again. Enter `q` at a prompt to
finish. Normal library training uses `cycles=None` by default for ongoing
interaction. The plain `ppe demo` command runs one scripted cycle without prompts.

The public `get_preference(nobj)` function exposes the objective/value prompts;
`trainer.fit(...)` already requests preferences at the relevant training stages.
See the [public user guide](ppe-library/docs/guide.html) for installation, dataset
adaptation, custom losses, preferences, and supported behavior. Copy the
[classification starter](ppe-library/examples/custom_dataset.py) to your own
`my_experiment.py`, customize it, and run `ppe run my_experiment:build` from its
directory. [Five or more objectives](ppe-library/docs/guide.html#many-objectives)
use the same interface: align objective names, model outputs, losses and targets.

For a live 3D plot, install `'./ppe-library[plot]'` and add `--plot` to the
command. PPE tries QtAgg, then TkAgg, and starts WebAgg automatically only if
neither desktop backend is usable. Choose displayed objectives with
`--plot-objectives 1 3 5`; training still uses every objective. The
[plotting guide](ppe-library/docs/plotting.md) covers interpreting and exporting
trajectories. Use a fresh output directory for each independent run.

## General descriptions
| Name | Type | Description |
|----------|----------|----------|
| Data    | folder     | Contains all experimental data used for testing the PPE together with all all loading and transformation applied.      |
| model_path   | folder    | models saved during training.     |
|ParetoMTL | folder | Contains the ParetoMTL framework used for comparison.  |
|plots | folder | Contains the .py files for visualizing the results of the multitask datasets saved in the ```Results``` folder  |
| Results    | folder     | Saved pickle files containing optimal points for later visualization.     |
| src  | folder  | Contains functions and models used for the PPE framework for multitask problems.  | 
| ppe-library | folder | Installable training library, dataset adapters, examples and documentation website. |
| toyEx_code | folder  | The PPE implementation for toy examples such as the DTLZ 1-7 and other mathematical functions.   |
|main.py | script | Run to implement the PPE for included DL problems such as 3-task MultiMNIST and 3 & 5 -task UCI census income problem.|
| ws.py | script | Contains the weighted sum implementation.|

- Download the multitask datasets, i.e., the MultiMNIST, the UCI Census 3 and 5-task datasets, into the **Data** folder by clicking here [:arrow_right: DATASET](https://doi.org/10.5281/zenodo.20623056)


## Packages
```
* Python 3.11.5
* Torch 2.2.2
* numpy 1.26.4  
```


## Toy examples (``` toyEx_code```)

This folder contains the various toy examples shown both in the main paper and also the supplementary material of the PPE paper. Each file in this folder contains the mathematical or benchmark toy problems in a notebook or .py file. 

To visualize the interactivity and Pareto navigation, the .py files should be run. For final visualization, run the ```plot_dtlz.py``` or ```plot_toy2.py``` for the benchmark or mathematical problem, respectively.


## Experiment
To run the PPE framework for the multitask datasets in ```Data```, simply run the ```main.py``` and  interactively supply the preference weights to navigate to new Pareto optimal points. 
- For example: ```main.py  --dtype UCI --num_obj 3``` loads and runs the PPE for the 3-tasks UCI Census income dataset with the already saved model for the initial optimal solution 
*(To retrain the initial point from scratch see ```util.py``` in the src folder on where to comment out)*.

For personal use, simply add your model and dataset into the ```model.py``` and ```dataLoader.py```, and ensure both your new model and data are called in the ```main.py```. Interactive visualization during navigation is only available in 3D. For more than three objectives, final visualization of all objectives can be done by loading the results and using the ```plot_uciplus.py```.

