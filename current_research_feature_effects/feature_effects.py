from typing_extensions import List, Dict, Callable, Tuple
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold


def _partial_dependence(estimator: BaseEstimator, X: np.ndarray, feature: int, grid: np.ndarray) -> Dict:
    """
    Compute partial dependence of features in a dataset based on a provided grid of values.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted model to compute partial dependence for.
    X : np.ndarray
        Dataset to compute partial dependence for.
    feature : int
        Index of feature to compute partial dependence for.
    grid : np.ndarray
        Grid of values to compute partial dependence for.

    Returns
    -------
    Dict
        Dictionary containing the grid values and the computed partial dependence values.
    """
    pdp_values = np.zeros(len(grid))

    for i, value in enumerate(grid):
        X_temp = X.copy()
        X_temp[:, feature] = value

        predictions = estimator.predict(X_temp)
        pdp_values[i] = np.mean(predictions)

    return {"grid_values": grid, "effect": pdp_values}


def _accumulated_local_effects(
    estimator: BaseEstimator, X: pd.DataFrame, feature: str, grid: np.ndarray
) -> pd.DataFrame:
    """Compute the accumulated local effect of a numeric continuous feature.

    This function is a customized version of the `aleplot_1D_continuous()` function
    from the PyALE library. The original function can be found at:

    https://github.com/DanaJomar/PyALE/blob/master/PyALE/_src/ALE_1D.py

    (version 1.2.0)

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted model to compute accumulated local effects for.
    X : pd.DataFrame
        Dataset to compute accumulated local effects for.
    feature : str
        Name of the feature to compute accumulated local effects for.
    grid : np.ndarray
        Grid of values to compute accumulated local effects for.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the grid values and the computed accumulated local effects values.

    """
    # modification: use custom grid
    bins = np.unique(grid)
    feat_cut = pd.cut(X[feature], bins, include_lowest=True)

    bin_codes = feat_cut.cat.codes

    X1 = X.copy()
    X2 = X.copy()
    X1[feature] = [bins[i] for i in bin_codes]
    X2[feature] = [bins[i + 1] for i in bin_codes]
    try:
        y_1 = estimator.predict(X1).ravel()
        y_2 = estimator.predict(X2).ravel()
    except Exception:
        raise Exception("Please check that your model is fitted, and accepts X as input.")

    delta_df = pd.DataFrame({feature: bins[bin_codes + 1], "Delta": y_2 - y_1})

    # modification to also include empty bins:
    all_grid_points = pd.DataFrame({feature: bins})  # include all points
    # Compute means and sizes for each grid point, ensuring all points are preserved
    grouped = delta_df.groupby([feature], observed=False)
    means = grouped.Delta.mean().reindex(all_grid_points[feature]).fillna(0)
    sizes = grouped.Delta.size().reindex(all_grid_points[feature]).fillna(0)
    res_df = pd.DataFrame({"eff": means, "size": sizes})

    res_df["eff"] = res_df["eff"].cumsum()
    res_df.loc[min(bins), :] = 0
    # subtract the total average of a moving average of size 2
    mean_mv_avg = ((res_df["eff"] + res_df["eff"].shift(1, fill_value=0)) / 2 * res_df["size"]).sum() / res_df[
        "size"
    ].sum()
    res_df = res_df.sort_index().assign(eff=res_df["eff"] - mean_mv_avg)

    return res_df


def get_modified_grids(
    base_grids: List[np.ndarray], Xs: List[np.ndarray], feature_names: List[str]
) -> Tuple[np.ndarray]:
    """
    Get modified grids based on the minimum and maximum values of the datasets.

    Parameters
    ----------
    base_grids : List[np.ndarray]
        List of base grids to be modified.
    Xs : List[np.ndarray]
        List of datasets to compute the minimum and maximum values from.
    feature_names : List[str]
        Names of features to compute the minimum and maximum values from.

    Returns
    -------
    Tuple[np.ndarray]
        Tuple of modified grids based on the minimum and maximum values of the datasets,
        length of the tuple is equal to the number of datasets.
    """
    mins = np.array([[np.min(X[:, i]) for i in range(X.shape[1])] for X in Xs]).T
    maxs = np.array([[np.max(X[:, i]) for i in range(X.shape[1])] for X in Xs]).T

    common_mins = mins.max(axis=1)
    common_maxs = maxs.min(axis=1)

    dataset_grids = [[] for _ in range(len(Xs))]
    for i in range(len(feature_names)):
        base_grid = base_grids[i]
        filtered_grid = base_grid[(base_grid > common_mins[i]) & (base_grid < common_maxs[i])]

        for j in range(len(Xs)):
            specific_grid = np.concatenate([[mins[i, j]], filtered_grid, [maxs[i, j]]])
            dataset_grids[j].append(specific_grid)

    return tuple(dataset_grids)


def compute_pdps(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: List[str],
    grid_values: List[np.ndarray],
    center_curves: bool = False,
    remove_first_last: bool = False,
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
    remove_first_last : bool, optional
        Whether to remove the first and last grid values from the PDP curves before centering and
        returning or not, by default False.

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
            feature=feature,
            grid=grid,
        )

        if remove_first_last:
            pdp_feature["effect"] = pdp_feature["effect"][1:-1]
            pdp_feature["grid_values"] = pdp_feature["grid_values"][1:-1]

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
    grid_values: List[np.ndarray],
    center_curves: bool = False,
    remove_first_last: bool = False,
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
    grid_values : List[np.ndarray]
        Grid of values to compute accumulated local effects for.
    center_curves : bool, optional
        Whether to center the ALE curves by substracting their mean or not, by default False.
    remove_first_last : bool, optional
        Whether to remove the first and last grid values from the ALE curves before centering and
        returning or not, by default False.

    Returns
    -------
    List[Dict]
        List of dictionaries containing the feature name, grid values and computed accumulated local effects values.
    """
    ales = []
    X_df = pd.DataFrame(X, columns=feature_names)
    for feature, grid in zip(feature_names, grid_values):
        ale_feature = _accumulated_local_effects(
            estimator=model,
            X=X_df,
            feature=feature,
            grid=grid,
        )

        if remove_first_last:
            ale_feature = ale_feature.iloc[1:-1]

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


def compute_cv_feature_effect(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    feature_names: List[str],
    cv_grids: List[List[np.ndarray]],
    effect_fn: Callable,
    center_curves: bool = False,
    remove_first_last: bool = False,
) -> List[Dict]:
    """
    Compute feature effects using cross-validation.

    For each fold, fits the model on training data and computes feature effects on validation data.
    The effects across folds are then averaged. Grid points must be consistent across folds
    for each feature.

    Parameters
    ----------
    model : BaseEstimator
        Model to compute feature effects for.
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    cv : KFold
        Cross-validation splitter.
    feature_names : List[str]
        Names of the features.
    cv_grids : List[List[np.ndarray]]
        Grid points for each feature for each fold.
    effect_fn : Callable
        Function to compute feature effects. Must take arguments:
        (model, X, feature_names, grid_values, center_curves, remove_first_last)
    center_curves : bool, default=False
        Whether to center the effect curves around zero.
    remove_first_last : bool, default=False
        Whether to remove first and last grid points from effects.

    Returns
    -------
    List[Dict]
        List of dictionaries, one per feature, containing:
        - 'feature': Feature name
        - 'grid_values': Grid points for this feature
        - 'effect': Averaged effect values across folds

    Raises
    ------
    ValueError
        If grid points are not consistent across folds for any feature.
    """
    effects = []
    for (train_index, test_index), cv_grid in zip(cv.split(X=X, y=y), cv_grids):
        X_train, X_val = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]
        model_fold = deepcopy(model)
        model_fold.fit(X_train, y_train)
        effect_fold = effect_fn(model_fold, X_val, feature_names, cv_grid, center_curves, remove_first_last)
        effects.append(effect_fold)

    averaged_effects = []
    for feature in feature_names:
        grids = [item["grid_values"] for sublist in effects for item in sublist if item["feature"] == feature]
        all_same = all(np.array_equal(grid, grids[0]) for grid in grids)

        if not all_same:
            raise ValueError(f"Grid values not consistent for feature {feature}")

        effect_folds = [item["effect"] for sublist in effects for item in sublist if item["feature"] == feature]
        averaged_effect = {"feature": feature, "grid_values": grids[0], "effect": np.mean(effect_folds, axis=0)}

        averaged_effects.append(averaged_effect)

    return averaged_effects


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
