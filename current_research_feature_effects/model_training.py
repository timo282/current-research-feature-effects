from typing_extensions import Literal
from pathlib import Path
import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error

from current_research_feature_effects.mappings import suggested_hps_for_model


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
    storage_name: Path
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

    study.optimize(objective, n_trials=n_trials)

    return study
