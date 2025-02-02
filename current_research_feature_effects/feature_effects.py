from typing_extensions import List, Dict, Callable, Literal, Union
from copy import deepcopy
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from current_research_feature_effects.data_generating.data_generation import Groundtruth


EffectType = Literal["pdp", "ale"]


@dataclass
class FeatureEffect:
    """
    Class to store and manipulate feature effects (PDP or ALE).
    """

    effect_type: EffectType
    features: Dict[str, Dict[str, np.ndarray]]  # {feature_name: {'grid': array, 'effect': array}}

    def __init__(
        self,
        effect_type: EffectType,
        feature_effects: List[Dict],  # List of dicts with 'feature', 'grid_values', 'effect' keys
    ):
        """
        Initialize from list of feature effect dictionaries.

        Parameters
        ----------
        effect_type : EffectType
            Type of effect (PDP or ALE).
        feature_effects : List[Dict]
            List of dictionaries containing feature effect information.
        """
        self.effect_type = effect_type
        self.features = {
            effect["feature"]: {"grid": np.array(effect["grid_values"]), "effect": np.array(effect["effect"])}
            for effect in feature_effects
        }

    def __repr__(self) -> str:
        features_str = ", ".join(self.features.keys())
        return f"FeatureEffect(type={self.effect_type}, features=[{features_str}])"

    def _validate_operation(self, other: "FeatureEffect") -> None:
        """Validate that two FeatureEffects can be combined."""
        if not isinstance(other, FeatureEffect):
            raise TypeError(f"Unsupported operand type: {type(other)}")

        if self.effect_type != other.effect_type:
            raise ValueError(f"Cannot combine different effect types: {self.effect_type} and {other.effect_type}")

        if set(self.features.keys()) != set(other.features.keys()):
            raise ValueError("Feature sets must match for arithmetic operations")

        for feature in self.features:
            if not np.array_equal(self.features[feature]["grid"], other.features[feature]["grid"]):
                raise ValueError(f"Grid values must match for feature {feature}")

    def _apply_operation(self, other: Union["FeatureEffect", float, int], op) -> "FeatureEffect":
        """Apply operation to effect values."""
        if isinstance(other, (float, int)):
            # Scalar operation
            new_effects = [
                {
                    "feature": feature,
                    "grid_values": self.features[feature]["grid"],
                    "effect": op(self.features[feature]["effect"], other),
                }
                for feature in self.features
            ]
        else:
            # FeatureEffect operation
            self._validate_operation(other)
            new_effects = [
                {
                    "feature": feature,
                    "grid_values": self.features[feature]["grid"],
                    "effect": op(self.features[feature]["effect"], other.features[feature]["effect"]),
                }
                for feature in self.features
            ]

        return FeatureEffect(self.effect_type, new_effects)

    def __add__(self, other: Union["FeatureEffect", float, int]) -> "FeatureEffect":
        return self._apply_operation(other, np.add)

    def __sub__(self, other: Union["FeatureEffect", float, int]) -> "FeatureEffect":
        return self._apply_operation(other, np.subtract)

    def __mul__(self, other: Union["FeatureEffect", float, int]) -> "FeatureEffect":
        return self._apply_operation(other, np.multiply)

    def __truediv__(self, other: Union["FeatureEffect", float, int]) -> "FeatureEffect":
        if isinstance(other, (float, int)) and other == 0:
            raise ZeroDivisionError("Division by zero")
        return self._apply_operation(other, np.divide)

    def __pow__(self, power: float) -> "FeatureEffect":
        return self._apply_operation(power, np.power)

    def __radd__(self, other: Union["FeatureEffect", float, int]) -> "FeatureEffect":
        return self.__add__(other)

    def __rsub__(self, other: Union["FeatureEffect", float, int]) -> "FeatureEffect":
        return self._apply_operation(other, lambda x, y: y - x)

    def __rmul__(self, other: Union["FeatureEffect", float, int]) -> "FeatureEffect":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union["FeatureEffect", float, int]) -> "FeatureEffect":
        return self._apply_operation(other, lambda x, y: y / x)

    def mean(self) -> Dict[str, float]:
        """Aggregate the feature effects into one number per feature by taking the mean."""
        return {feature: np.mean(self.features[feature]["effect"]) for feature in self.features}

    def to_list(self) -> List[Dict]:
        """Convert back to list of dictionaries format."""
        return [
            {
                "feature": feature,
                "grid_values": self.features[feature]["grid"],
                "effect": self.features[feature]["effect"],
            }
            for feature in self.features
        ]


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
        y_1 = estimator.predict(X1.to_numpy()).ravel()
        y_2 = estimator.predict(X2.to_numpy()).ravel()
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


# def get_modified_grids(
#     base_grids: List[np.ndarray], Xs: List[np.ndarray], feature_names: List[str]
# ) -> Tuple[np.ndarray]:
#     """
#     Get modified grids based on the minimum and maximum values of the datasets.

#     Parameters
#     ----------
#     base_grids : List[np.ndarray]
#         List of base grids to be modified.
#     Xs : List[np.ndarray]
#         List of datasets to compute the minimum and maximum values from.
#     feature_names : List[str]
#         Names of features to compute the minimum and maximum values from.

#     Returns
#     -------
#     Tuple[np.ndarray]
#         Tuple of modified grids based on the minimum and maximum values of the datasets,
#         length of the tuple is equal to the number of datasets.
#     """
#     mins = np.array([[np.min(X[:, i]) for i in range(X.shape[1])] for X in Xs]).T
#     maxs = np.array([[np.max(X[:, i]) for i in range(X.shape[1])] for X in Xs]).T

#     common_mins = mins.max(axis=1)
#     common_maxs = maxs.min(axis=1)

#     dataset_grids = [[] for _ in range(len(Xs))]
#     for i in range(len(feature_names)):
#         base_grid = base_grids[i]
#         filtered_grid = base_grid[(base_grid > common_mins[i]) & (base_grid < common_maxs[i])]

#         for j in range(len(Xs)):
#             specific_grid = np.concatenate([[mins[i, j]], filtered_grid, [maxs[i, j]]])
#             dataset_grids[j].append(specific_grid)

#     return tuple(dataset_grids)


def compute_pdps(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: List[str],
    grid_values: List[np.ndarray],
    center_curves: bool = False,
    remove_first_last: bool = False,
) -> FeatureEffect:
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
    FeatureEffect
        Object containing the computed partial dependence values.
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

    return FeatureEffect("pdp", pdp)


def compute_ales(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: List[str],
    grid_values: List[np.ndarray],
    center_curves: bool = False,
    remove_first_last: bool = False,
) -> FeatureEffect:
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
    FeatureEffect
        Object containing the computed accumulated local effects values.
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

    return FeatureEffect("ale", ales)


def compute_theoretical_effects(
    groundtruth: Groundtruth,
    effect: EffectType,
    feature_names: List[str],
    grid_values: List[np.ndarray],
    center_curves: bool = False,
    remove_first_last: bool = False,
) -> FeatureEffect:
    """
    Compute theoretical partial dependence plots for a given groundtruth
    and apply it to the grid_values.

    Parameters
    ----------
    groundtruth : Groundtruth
        Groundtruth object to compute theoretical partial dependence for.
    effect : Literal["pdp", "ale"]
        Type of effect to compute: partial dependence or accumulated local effects.
    feature_names : List[str]
        Names of features to compute theoretical partial dependence for.
    grid_values : List[np.ndarray]
        Grid of values to compute theoretical partial dependence for.
    center_curves : bool, optional
        Whether to center curves by substracting the mean (based on grid values) or not, by default False.
    remove_first_last : bool, optional
        Whether to remove the first and last grid values from the effects before centering and
        returning or not, by default False.

    Returns
    -------
    FeatureEffect
        Object containing the computed theoretical partial dependence values.
    """
    effects = []
    for f_name, grid in zip(feature_names, grid_values):

        if remove_first_last:
            grid = grid[1:-1]

        if effect == "pdp":
            feature_effect = groundtruth.get_theoretical_partial_dependence(feature=f_name)(grid)
        elif effect == "ale":
            feature_effect = groundtruth.get_theoretical_accumulated_local_effects(feature=f_name)(grid)
        else:
            raise ValueError("Effect type not supported")

        if center_curves:
            feature_effect -= np.mean(feature_effect)

        effects.append(
            {
                "feature": f_name,
                "grid_values": grid,
                "effect": feature_effect,
            }
        )

    return FeatureEffect(effect, effects)


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
    return_models: bool = False,
) -> FeatureEffect:
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
    return_models : bool, default=False
        Whether to return the fitted models for each fold.

    Returns
    -------
    FeatureEffect
        Object containing the averaged feature effects across folds.

    Raises
    ------
    ValueError
        If grid points are not consistent across folds for any feature.
    """
    effects = []
    models = []
    for (train_index, test_index), cv_grid in zip(cv.split(X=X, y=y), cv_grids):
        X_train, X_val = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]
        model_fold = deepcopy(model)
        model_fold.fit(X_train, y_train)
        effect_fold = effect_fn(model_fold, X_val, feature_names, cv_grid, center_curves, remove_first_last)
        effects.append(effect_fold)
        models.append(model_fold)

    averaged_effects = effects[0]
    for effect in effects[1:]:
        averaged_effects += effect

    averaged_effects /= len(effects)

    if return_models:
        return averaged_effects, [model_fold for model_fold in models]

    return averaged_effects


def compute_feature_effect_metrics(estimates: List[FeatureEffect], groundtruth: FeatureEffect):
    """
    Compute pointwise metrics to compare feature effects estimates with groundtruth.

    Parameters
    ----------
    estimates : List[FeatureEffect]
        List of feature effect estimates.
    groundtruth : FeatureEffect
        Groundtruth feature effect.

    Returns
    -------
    Dict
        Dictionary containing the computed metrics.
    """
    metrics = {}

    mean_estimate = sum(estimates) / len(estimates)

    for estimate in estimates:
        se = (estimate - groundtruth) ** 2
        sd = (estimate - mean_estimate) ** 2
        metrics["MSE"] = se if "MSE" not in metrics else metrics["MSE"] + se
        metrics["Variance"] = sd if "Variance" not in metrics else metrics["Variance"] + sd

    metrics["MSE"] /= len(estimates)
    metrics["Variance"] /= len(estimates) - 1
    metrics["Bias^2"] = (mean_estimate - groundtruth) ** 2

    return metrics


def compute_variance(estimates: List[FeatureEffect]) -> FeatureEffect:
    """
    Compute variance of feature effect estimates.

    Parameters
    ----------
    estimates : List[FeatureEffect]
        List of feature effect estimates.

    Returns
    -------
    FeatureEffect
        Feature effect object containing the computed variance.
    """
    mean_estimate = sum(estimates) / len(estimates)
    return sum((estimate - mean_estimate) ** 2 for estimate in estimates) / (len(estimates) - 1)
