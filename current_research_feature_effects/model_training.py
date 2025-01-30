from typing_extensions import Literal, Dict
from pathlib import Path
from configparser import ConfigParser
import os
import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error

from current_research_feature_effects.mappings import suggested_hps_for_model
from current_research_feature_effects.data_generating.data_generation import generate_data, Groundtruth


def _objective(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    trial: optuna.trial.Trial,
    metric: str,
) -> float:
    hyperparams = suggested_hps_for_model(model, trial)
    model.set_params(**hyperparams)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    if metric == "neg_mean_squared_error":
        score = -mean_squared_error(y_val, y_pred)
    elif metric == "neg_mean_absolute_error":
        score = -mean_absolute_error(y_val, y_pred)
    else:
        raise ValueError(f"Invalid metric {metric}.")

    return score


def optimize(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    metric: str,
    direction: Literal["maximize", "minimize"],
    study_name: str,
    storage_name: Path,
) -> optuna.study.Study:
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        sampler=sampler,
        storage=storage_name,
        study_name=study_name,
        direction=direction,
        load_if_exists=False,
    )

    def objective(trial: optuna.trial.Trial):
        return _objective(model, X_train, y_train, X_val, y_val, trial, metric)

    study.optimize(objective, n_trials=n_trials, catch=(np.linalg.LinAlgError, RuntimeError))

    return study


def initialize_model(
    model_dict: Dict, model_str: str, groundtruth: Groundtruth, n_train: int, snr: float, config: ConfigParser
) -> BaseEstimator:
    """Initialize model with hyperparameters (tuned or default).

    This function initializes a model with hyperparameters. If the hyperparameters are set to "to_tune" in the model
    dictionary, the function will either load the best hyperparameters from a tuning study or run a new tuning study
    if the study does not exist. The function will then set the hyperparameters of the model and return it.

    Parameters
    ----------
    model_dict : Dict
        Dictionary containing the model, and the hyperparameters to tune per dataset and dataset size.
    model_str : str
        Name of the model.
    groundtruth : Groundtruth
        Groundtruth object of the dataset.
    n_train : int
        Number of training samples.
    snr : float
        Signal-to-noise ratio.
    config : ConfigParser
        Config containing simulation metadata.

    Returns
    -------
    BaseEstimator
        Initialized model.
    """
    model: BaseEstimator = model_dict["model"]
    if model_dict["model_params"][str(groundtruth)][n_train] == "to_tune":
        tuning_studies_dir = config.get("storage", "tuning_studies_folder")
        os.makedirs(Path(os.getcwd()) / str(groundtruth) / tuning_studies_dir, exist_ok=True)
        study_name = f"{model_str}_{n_train}_{int(snr)}"
        storage_name = f"sqlite:///{str(groundtruth)}/{tuning_studies_dir}/{model_str}_{n_train}_{int(snr)}.db"
        try:
            model_params = optuna.load_study(
                study_name=study_name,
                storage=storage_name,
            ).best_params
        except KeyError as e:
            if str(e) == "'Record does not exist.'":
                X_tuning_train, y_tuning_train, X_tuning_val, y_tuning_val = generate_data(
                    groundtruth=groundtruth,
                    n_train=n_train,
                    n_test=config.getint("simulation_metadata", "n_tuning_val"),
                    snr=snr,
                    seed=config.getint("simulation_metadata", "tuning_data_seed"),
                )
                model_params = optimize(
                    model=model,
                    X_train=X_tuning_train,
                    y_train=y_tuning_train,
                    X_val=X_tuning_val,
                    y_val=y_tuning_val,
                    n_trials=config.getint("simulation_metadata", "n_tuning_trials"),
                    metric=config.get("simulation_metadata", "tuning_metric"),
                    direction=config.get("simulation_metadata", "tuning_direction"),
                    study_name=study_name,
                    storage_name=storage_name,
                ).best_params
            else:
                raise
    else:
        model_params = model_dict["model_params"][str(groundtruth)][n_train]

    model.set_params(**model_params)

    return model
