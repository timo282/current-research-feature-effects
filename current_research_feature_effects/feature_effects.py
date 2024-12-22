import warnings
from typing_extensions import List, Dict, Callable
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from PyALE import ale


def _partial_dependence(estimator: BaseEstimator, X: np.ndarray, features: List[int], grid: List[np.ndarray]) -> Dict:
    """
    Compute partial dependence of features in a dataset based on a provided grid of values.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted model to compute partial dependence for.
    X : np.ndarray
        Dataset to compute partial dependence for.
    features : List[int]
        Indices of features to compute partial dependence for.
    grid : List[np.ndarray]
        Grid of values to compute partial dependence for.

    Returns
    -------
    Dict
        Dictionary containing the grid values and the computed partial dependence values.
    """
    if not isinstance(features, list):
        features = [features]

    grid_values = grid if isinstance(grid, list) else [grid]
    pdp_values = np.zeros(len(grid_values[0]))

    for i, value in enumerate(grid_values[0]):
        X_temp = X.copy()
        X_temp[:, features[0]] = value

        predictions = estimator.predict(X_temp)
        pdp_values[i] = np.mean(predictions)

    return {"grid_values": grid_values[0], "effect": pdp_values}


def compute_pdps(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: List[str],
    grid_values: List[np.ndarray],
    center_curves: bool = False,
) -> List[Dict]:
    """
    Compute partial dependence plots for a given model and dataset.

    Parameters
    ----------
    model : BaseEstimator
        Fitted model to compute partial dependence for.
    X : np.ndarray
        Dataset to compute partial dependence for.
    feature_names : List[str]
        Names of features to compute partial dependence for.
    grid_values : List[np.ndarray]
        Grid of values to compute partial dependence for.
    center_curves : bool, optional
        Whether to center the PDP curves by substracting their mean or not, by default False.

    Returns
    -------
    List[Dict]
        List of dictionaries containing the feature name, grid values and computed partial dependence values.
    """
    pdp = []
    for feature, f_name, grid in zip(range(X.shape[1]), feature_names, grid_values):
        pdp_feature = _partial_dependence(
            estimator=model,
            X=X,
            features=[feature],
            grid=[grid],
        )

        if center_curves:
            pdp_feature["effect"] -= np.mean(pdp_feature["effect"])

        pdp.append(
            {
                "feature": f_name,
                "grid_values": pdp_feature["grid_values"],
                "effect": pdp_feature["effect"],
            }
        )

    return pdp


def compute_ales(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: List[str],
    grid_intervals: int,
    center_curves: bool = False,
) -> List[Dict]:
    """
    Compute accumulated local effects for a given model and dataset.

    Parameters
    ----------
    model : BaseEstimator
        Fitted model to compute accumulated local effects for.
    X : np.ndarray
        Dataset to compute accumulated local effects for.
    feature_names : List[str]
        Names of features to compute accumulated local effects for.
    grid_intervals : int
        Number of intervals to compute accumulated local effects for.
    center_curves : bool, optional
        Whether to center the ALE curves by substracting their mean or not, by default False.

    Returns
    -------
    List[Dict]
        List of dictionaries containing the feature name, grid values and computed accumulated local effects values.
    """
    ales = []
    X_df = pd.DataFrame(X, columns=feature_names)
    for feature in X_df.columns:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            ale_feature = ale(
                X=X_df,
                model=model,
                feature=[feature],
                grid_size=grid_intervals,
                plot=False,
                include_CI=False,
            )

        if center_curves:
            ale_feature["eff"] -= np.mean(ale_feature["eff"])

        ales.append(
            {
                "feature": feature,
                "grid_values": ale_feature.index.values,
                "effect": ale_feature["eff"].values,
            }
        )

    return ales


def compare_effects(
    effects_groundtruth: List[Dict],
    effects_model: List[Dict],
    metric: Callable,
) -> pd.DataFrame:
    comparison = {"metric": metric.__name__}
    for i, effects_model_feature in enumerate(effects_model):
        effects_groundtruth_feature = effects_groundtruth[i]
        if effects_groundtruth_feature["feature"] != effects_model_feature["feature"]:
            raise ValueError("Features in groundtruth and model effects do not match")
        if not np.array_equal(
            effects_groundtruth_feature["grid_values"],
            effects_model_feature["grid_values"],
        ):
            raise ValueError("Grid values in groundtruth and model effects do not match")

        comparison[effects_model_feature["feature"]] = metric(
            effects_groundtruth_feature["effect"],
            effects_model_feature["effect"],
        )

    return pd.DataFrame(comparison, index=[0])
