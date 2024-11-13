from typing_extensions import Literal
import os
from pathlib import Path
import uuid
import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from current_research_feature_effects.mappings import suggested_hps_for_model


def _objective(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    trial: optuna.trial.Trial,
    cv: int,
    metric: str,
) -> float:
    hyperparams = suggested_hps_for_model(model, trial)
    model.set_params(**hyperparams)

    score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=1).mean()

    return score


def optimize(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int,
    cv: int,
    metric: str,
    direction: Literal["maximize", "minimize"],
    tuning_studies_folder: str,
    model_name: str,
) -> optuna.study.Study:
    model_str = model.__class__.__name__
    sampler = optuna.samplers.TPESampler(seed=42)
    os.makedirs(Path(os.getcwd()) / tuning_studies_folder, exist_ok=True)
    study = optuna.create_study(
        sampler=sampler,
        storage=f"sqlite:///{tuning_studies_folder}/{model_str}.db",
        study_name=f"{model_name}_{uuid.uuid4().hex}",
        direction=direction,
        load_if_exists=False,
    )

    def objective(trial: optuna.trial.Trial):
        return _objective(model, X_train, y_train, trial, cv, metric)

    study.optimize(objective, n_trials=n_trials)

    return study


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int,
    cv: int,
    metric: str,
    direction: Literal["maximize", "minimize"],
    tuning_studies_folder: str,
    model_name: str,
) -> BaseEstimator:

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    study = optimize(
        model=model,
        X_train=X_train,
        y_train=y_train,
        n_trials=n_trials,
        cv=cv,
        metric=metric,
        direction=direction,
        tuning_studies_folder=tuning_studies_folder,
        model_name=model_name,
    )

    hyperparams = study.best_params
    model.set_params(**hyperparams)

    model.fit(X_train, y_train)

    return model
