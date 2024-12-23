from typing_extensions import List, Tuple, Literal
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin
from pygam import LinearGAM, s, l, te
from xgboost import XGBRegressor
import optuna

from current_research_feature_effects.data_generating.data_generation import Groundtruth
from current_research_feature_effects.data_generating.simple import (
    SimpleAdditiveGroundtruth,
    SimpleInteractionGroundtruth,
    SimpleCombinedGroundtruth,
)
from current_research_feature_effects.data_generating.friedman1 import Friedman1Groundtruth
from current_research_feature_effects.data_generating.physics_data import NewtonUniversalGravitationGroundtruth


def map_dataset_to_groundtruth(
    dataset: str,
    marginals: List[Tuple[Literal["normal", "uniform", "loguniform"], Tuple]],
    corr_matrix: np.array,
    feature_names: List[str] = None,
    name: str = None,
) -> Groundtruth:
    if dataset == "SimpleAdditiveGroundtruth":
        return SimpleAdditiveGroundtruth(
            marginal_distributions=marginals, correlation_matrix=corr_matrix, feature_names=feature_names, name=name
        )
    if dataset == "SimpleInteractionGroundtruth":
        return SimpleInteractionGroundtruth(
            marginal_distributions=marginals, correlation_matrix=corr_matrix, feature_names=feature_names, name=name
        )
    if dataset == "SimpleCombinedGroundtruth":
        return SimpleCombinedGroundtruth(
            marginal_distributions=marginals, correlation_matrix=corr_matrix, feature_names=feature_names, name=name
        )
    if dataset == "Friedman1Groundtruth":
        return Friedman1Groundtruth(
            marginal_distributions=marginals, correlation_matrix=corr_matrix, feature_names=feature_names, name=name
        )
    if dataset == "NewtonUniversalGravitationGroundtruth":
        return NewtonUniversalGravitationGroundtruth(
            marginal_distributions=marginals, correlation_matrix=corr_matrix, feature_names=feature_names, name=name
        )


def map_modelname_to_estimator(model_name: str) -> BaseEstimator:
    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=42)
    if model_name == "XGBoost-full":
        return XGBRegressor(random_state=42)
    if model_name == "XGBoost-f1-cor" or model_name == "XGBoost-2comb-cor":
        return XGBRegressor(random_state=42, interaction_constraints="[[0, 1]]")
    if model_name == "XGBoost-2add-cor":
        return XGBRegressor(random_state=42, interaction_constraints="[]")
    if model_name == "DecisionTree":
        return DecisionTreeRegressor(random_state=42)
    if model_name == "SVM-RBF":
        return SVR(kernel="rbf")
    if model_name == "ElasticNet":
        return ElasticNet(random_state=42, max_iter=10000)
    if model_name == "GAM-f1-cor":
        return GAM(te_features=[(0, 1)], s_features=[2], l_features=[3, 4])
    if model_name == "GAM-2add-cor":
        return GAM(s_features=[0, 1])
    if model_name == "GAM-2comb-cor":
        return GAM(te_features=[(0, 1)], s_features=[0, 1])
    if model_name == "GAM-4-full":
        return GAM(te_features=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], s_features=[0, 1, 2, 3])
    raise NotImplementedError("Base estimator not implemented yet")


def suggested_hps_for_model(model: BaseEstimator, trial: optuna.trial.Trial) -> dict:
    if isinstance(model, RandomForestRegressor):
        return _suggest_hps_rf(trial)
    if isinstance(model, XGBRegressor):
        return _suggest_hps_xgboost(trial)
    if isinstance(model, DecisionTreeRegressor):
        return _suggest_hps_tree(trial)
    if isinstance(model, SVR):
        return _suggest_hps_svm(trial)
    if isinstance(model, ElasticNet):
        return _suggest_hps_elasticnet(trial)
    if isinstance(model, GAM):
        return _suggest_hps_gam(trial)
    raise NotImplementedError("Base estimator not implemented yet")


def _suggest_hps_rf(trial: optuna.trial.Trial):
    # using the values from https://www.jmlr.org/papers/v20/18-444.html
    hyperparams = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1750),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.01, 0.5),
        "max_samples": trial.suggest_float("max_samples", 0.3, 0.975),
        # use 1 as max because of the small number of features in the dataset:
        "max_features": trial.suggest_float("max_features", 0.035, 1),
    }

    return hyperparams


def _suggest_hps_xgboost(trial: optuna.trial.Trial):
    # using the values from https://www.jmlr.org/papers/v20/18-444.html
    hyperparams = {
        "n_estimators": trial.suggest_int("n_estimators", 920, 4550),
        "max_depth": trial.suggest_int("max_depth", 5, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.355, log=True),
        "subsample": trial.suggest_float("subsample", 0.545, 0.958),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.295, 6.984, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.419, 0.864),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.335, 0.886),
        "lambda": trial.suggest_float("lambda", 0.008, 29.755, log=True),
        "alpha": trial.suggest_float("alpha", 0.002, 6.105, log=True),
    }

    return hyperparams


def _suggest_hps_tree(trial: optuna.trial.Trial):
    # using the values from https://www.jmlr.org/papers/v20/18-444.html
    hyperparams = {
        "max_depth": trial.suggest_int("max_depth", 12, 27),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 42),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0, 0.008),
    }

    return hyperparams


def _suggest_hps_svm(trial: optuna.trial.Trial):
    # using the values from https://www.jmlr.org/papers/v20/18-444.html
    hyperparams = {
        "C": trial.suggest_float("C", 0.002, 920, log=True),
        "gamma": trial.suggest_float("gamma", 0.003, 18, log=True),
    }

    return hyperparams


def _suggest_hps_elasticnet(trial: optuna.trial.Trial):
    # using the values from https://www.jmlr.org/papers/v20/18-444.html
    hyperparams = {
        "alpha": trial.suggest_float("alpha", 0.001, 0.147, log=True),  # lambd
        "l1_ratio": trial.suggest_float("l1_ratio", 0.009, 0.981),  # alpha
    }

    return hyperparams


def _suggest_hps_gam(trial: optuna.trial.Trial):
    hyperparams = {
        "n_splines": trial.suggest_int("n_splines", 5, 50),
        "lam": trial.suggest_float("lam", 1e-3, 1e3, log=True),
    }

    return hyperparams


class GAM(BaseEstimator, RegressorMixin):
    """
    GAM compatible with sklearn API.

    Example:
    ```
    gam = GAM(terms={"s": [0, 1], "l": [2, 3], "te": [(4, 5)]})
    gam.fit(X_train, y_train)
    gam.predict(X_test)
    ```
    """

    def __init__(
        self,
        s_features: List[int] | None = None,
        l_features: List[int] | None = None,
        te_features: List[Tuple[int]] | None = None,
        n_splines: int = 25,
        lam: float = 0.6,
    ):
        self.n_splines = n_splines
        self.lam = lam
        self.s_features = s_features
        self.l_features = l_features
        self.te_features = te_features
        self.terms = self._parse_terms(self.s_features, self.l_features, self.te_features)
        self.model = LinearGAM(self.terms)
        self._is_fitted__ = False

    def __sklearn_is_fitted__(self):
        return self._is_fitted__

    def _parse_terms(self, s_features, l_features, te_features):
        gam_term = None
        if s_features is not None:
            for feature in s_features:
                term = s(feature, n_splines=self.n_splines, lam=self.lam)
                gam_term = term if gam_term is None else gam_term + term
        if l_features is not None:
            for feature in l_features:
                term = l(feature)
                gam_term = term if gam_term is None else gam_term + term
        if te_features is not None:
            for features in te_features:
                term = te(*features, lam=self.lam)
                gam_term = term if gam_term is None else gam_term + term

        return gam_term

    def fit(self, X, y):
        self.model.fit(X, y)
        self._is_fitted__ = True
        return self

    def predict(self, X):
        return self.model.predict(X)
