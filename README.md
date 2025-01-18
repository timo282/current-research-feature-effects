# Current Research: Should We Compute Feature Effects on Training or Validation Data?

Explainable AI / Interpretable ML Research Project at LMU Munich.

// Intro / Description

## Installation
To install and use/contribute to this project, it is recommended to follow the following steps:

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

Navigate to the root folder of this repository and run:

```
poetry run python scripts/main_study.py --config configs/main_study.ini
```

This will automatically run a simulation based on the parameters provided in the `configs/main_study.ini.ini` config file. It will create a new subdirectory in `experiments` based on the simulation name you specified in the config, it will copy the config-file to this directory, and store all simulation results (trained models as joblib-files, model results, tuning trials, and feature effects results in database-files).

Random seeds are used throughout the entire simulation to make all results reproducible. Datasets created in each simulation run are not stored, but can simply be recreated afterwards if needed by using the number of the simulation run as seed for the dataset generation.

### Run Ablation Study on MC Variance

Navigate to the root folder of this repository and run:

```
poetry run python scripts/ablation_study_mc.py --config configs/ablation_study_mc.ini
```

Results can be found in the `experiments` directory under the corresponding experiment.


## Repository Structure

tbd

## Contact

tbd
