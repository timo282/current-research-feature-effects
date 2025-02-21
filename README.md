# Current Research Project: Understanding Implications of Dataset Choice for Feature Effect Estimation - A Simulation-Based Investigation through Error Decomposition

Interpretable Machine Learning Research Project at LMU Munich.

The choice of dataset for estimating feature effects - whether
to use training data, validation data, or even employ cross-validation -
is an important fundamental question in Interpretable ML that remains largely
unexplored.
Through a comprehensive simulation study, we investigate this question for
Partial Dependence Plots (PDP) and Accumulated Local Effects (ALE), decomposing
their error into bias and variance components.

This repository contains the source code implementing the simulation study,
the latex code for the paper, as well as the experiment results with corresponding
analyses.

## Repository Structure

- *analyses*: jupyter notebooks containing analyses and plots of the experiment results
    - *ablation_study_mc_error.ipynb*: analyses corresponding to the ablation study regarding dataset size impact on Monte Carlo variance
    - *ablation_study_variance_decomposition.ipynb*: analyses corresponding to the ablation study estimating Monte Carlo variances to enable variance decomposition
    - *groundtruth_effects.ipynb*: visualizations of the the ground truth feature effects for each dataset
    - *main_study_analysis.ipynb*: analyses corresponding to the main simulation study
    - *model_performance_analysis.ipynb*: evaluation of model performances and overview of model hyperparameters
-  *configs*: configurations of the experiments
    - *ablation_study_mc.ini*: configuration for the ablation study on dataset size impact on Monte Carlo variance
    - *ablation_study_variance_decomposition.ini*: configuration for the ablation study estimating Monte Carlo variances to enable variance decomposition
    - *datasets.yaml*: dataset configurations for all experiments
    - *main_study.ini*: configuration for the main experiment
    - *models_ablation_study.yaml*: model configurations for the ablation study on Monte Carlo variance estimation for variance decomposition
    - *models.yaml*: model configurations for the main experiment
- *current_research_feature_effects*: 'library' with implementations of functions and classes used in the experiment scripts
    - detailed documentation is provided in the code
- *experiments*: results of the experiments
    - *ablation_study_mc_error*: ablation study results for dataset size impact on Monte Carlo variance
    - *ablation_study_variance_decomposition_new*: ablation study results of Monte Carlo variances for variance decomposition
    - *main_study_parallel*: main experiment results
- *paper*: latex and other files necessary to compile the paper describing the entire theoretical background, approach, and results
- *scripts*: python scripts for the experiments
    - *ablation_study_mc.py*: script for ablation study to investigate dataset size impact on Monte Carlo variance
    - *ablation_study_variance_decomposition.py*: script to estimate Monte Carlo variances for variance decomposition
    - *main.py*: script for main simulation study

## Installation
To execute experiments in this project, it is recommended to follow these steps:

1.) Clone or fork this repository

2.) Install pipx:
```
pip install --user pipx
```

3.) Install Poetry:
```
pipx install poetry
```

*3a.) Optionally (if you want the env-folder to be created in your project)*:
```
poetry config virtualenvs.in-project true
```

4.) Install this project:
```
poetry install
```

*4a.) If the specified Python version (3.11 for this project) is not found:*

If the required Python version is not installed on your device, install Python from the official [python.org](https://www.python.org/downloads) website.

Then run
```
poetry env use <path-to-your-python-version>
```
and install the project:
```
poetry install
```

## Usage

### Run Main Study

Adapt the configs/main_study.ini with your paths, navigate to the root folder of this repository and run:

```
poetry run python scripts/main_study.py --config configs/main_study.ini
```

This will automatically run the main study based on the parameters provided in the `configs/main_study.ini.ini`.
It will create a new subdirectory in `experiments` based on the simulation name you specified in the config,
copy the config-file to this directory, and store all simulation results (model results, tuning trials, and feature effects results in database-files).

Random seeds are used throughout the entire simulation to make all results reproducible. Datasets created in each simulation run are not stored, but can simply be recreated afterwards if needed by using the number of the simulation run as seed for the dataset generation.

### Run Ablation Study on MC Variance Estimation for Variance Decomposition

Adapt the ablation_study_variance_decomposition.ini with your paths, navigate to the root folder of this repository and run:

```
poetry run python scripts/ablation_study_variance_decomposition.py --config configs/ablation_study_variance_decomposition.ini
```

Results can also be found in the `experiments` directory under the corresponding experiment.

### Run Ablation Study on Dataset Size Impact on MC Variance

Adapt the ablation_study_mc.ini with your paths, navigate to the root folder of this repository and run:

```
poetry run python scripts/ablation_study_mc.py --config configs/ablation_study_mc.ini
```

Results can also be found in the `experiments` directory under the corresponding experiment.
