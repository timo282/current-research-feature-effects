from configparser import ConfigParser
import warnings
from typing_extensions import List, Dict, Callable
import pandas as pd
import numpy as np
from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator
from PyALE import ale


def compute_pdps(
    model: BaseEstimator,
    X_train: np.ndarray,
    feature_names: List[str],
    config: ConfigParser,
) -> List[Dict]:
    pdp = []
    percentiles = (
        float(config.get("pdp", "percentiles").split(",")[0]),
        float(config.get("pdp", "percentiles").split(",")[1]),
    )
    grid_resolution = config.getint("pdp", "grid_resolution")
    for feature, f_name in zip(range(X_train.shape[1]), feature_names):
        pdp_feature = partial_dependence(
            estimator=model,
            X=X_train,
            features=[feature],
            kind="average",
            percentiles=percentiles,
            grid_resolution=grid_resolution,
        )
        pdp.append(
            {
                "feature": f_name,
                "grid_values": pdp_feature["grid_values"][0],
                "effect": pdp_feature["average"][0],
            }
        )

    return pdp


def compute_ales(
    model: BaseEstimator,
    X_train: np.ndarray,
    feature_names: List[str],
    config: ConfigParser,
) -> List[Dict]:
    ales = []
    grid_intervals = config.getint("ale", "grid_intervals")
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    for feature in X_train_df.columns:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            ale_feature = ale(
                X=X_train_df,
                model=model,
                feature=[feature],
                grid_size=grid_intervals,
                plot=False,
                include_CI=False,
            )
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
    center_curves: bool = False,
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

        groundtruth_effect = np.array(effects_groundtruth_feature["effect"])
        model_effect = np.array(effects_model_feature["effect"])

        # Center the effects by subtracting the mean if center_curves is True
        if center_curves:
            groundtruth_effect -= np.mean(groundtruth_effect)
            model_effect -= np.mean(model_effect)

        comparison[effects_model_feature["feature"]] = metric(
            effects_groundtruth_feature["effect"],
            effects_model_feature["effect"],
        )

    return pd.DataFrame(comparison, index=[0])
