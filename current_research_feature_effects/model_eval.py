"""
This module contains functions for evaluating model performance.
"""

from typing_extensions import Dict, List
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def eval_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, List[float]]:
    """
    Evaluate model on training and test set using MSE, MAE, and R2.

    Parameters
    ----------
    model : BaseEstimator
        Trained model.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training target.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test target.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary containing model evaluation metrics.
    """
    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, model.predict(X_test))
    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    mae_test = mean_absolute_error(y_test, model.predict(X_test))
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))

    metric_dict = {
        "mse_train": [mse_train],
        "mse_test": [mse_test],
        "mae_train": [mae_train],
        "mae_test": [mae_test],
        "r2_train": [r2_train],
        "r2_test": [r2_test],
    }

    return metric_dict


def empty_dict() -> Dict:
    """
    Return empty dictionary with keys for model evaluation metrics.
    """
    metric_dict = {
        "mse_train": [np.nan],
        "mse_test": [np.nan],
        "mae_train": [np.nan],
        "mae_test": [np.nan],
        "r2_train": [np.nan],
        "r2_test": [np.nan],
    }
    return metric_dict
