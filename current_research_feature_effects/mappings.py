from typing_extensions import List, Tuple, Literal
from itertools import combinations
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from pygam import LinearGAM, s, te
from xgboost import XGBRegressor
import optuna

from current_research_feature_effects.data_generating.data_generation import Groundtruth
from current_research_feature_effects.data_generating.simple import (
    SimpleAdditiveGroundtruth,
    SimpleInteractionGroundtruth,
    SimpleCombinedGroundtruth,
)
from current_research_feature_effects.data_generating.friedman1 import Friedman1Groundtruth
from current_research_feature_effects.data_generating.physics_data import (
    NewtonUniversalGravitationGroundtruth,
    WaveInterferenceGroundtruth,
)


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
    if dataset == "WaveInterferenceGroundtruth":
        return WaveInterferenceGroundtruth(
            marginal_distributions=marginals, correlation_matrix=corr_matrix, feature_names=feature_names, name=name
        )


def map_modelname_to_estimator(model_name: str) -> BaseEstimator:
    if model_name == "XGBoost":
        return XGBRegressor(random_state=42)
    if model_name == "SVM-RBF":
        return SVR(kernel="rbf")
    if model_name == "LinearRegression":
        return LinearRegression()
    if model_name == "GAM":
        return GAM(interaction_order=2)
    raise NotImplementedError(f"Base estimator {model_name} not implemented.")


def suggested_hps_for_model(model: BaseEstimator, trial: optuna.trial.Trial) -> dict:
    if isinstance(model, XGBRegressor):
        return _suggest_hps_xgboost(trial)
    if isinstance(model, SVR):
        return _suggest_hps_svm(trial)
    if isinstance(model, GAM):
        return _suggest_hps_gam(trial)
    raise NotImplementedError(f"HPO for model {model} not implemented.")


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


def _suggest_hps_svm(trial: optuna.trial.Trial):
    # using the values from https://www.jmlr.org/papers/v20/18-444.html
    hyperparams = {
        "C": trial.suggest_float("C", 0.002, 920, log=True),
        "gamma": trial.suggest_float("gamma", 0.003, 18, log=True),
    }

    return hyperparams


def _suggest_hps_gam(trial: optuna.trial.Trial):
    hyperparams = {
        "n_bases": trial.suggest_int("n_bases", 5, 50),
        "lam": trial.suggest_float("lam", 1e-3, 1e3, log=True),
    }

    return hyperparams


class GAM(BaseEstimator, RegressorMixin):
    """
     GAM compatible with sklearn API that automatically creates spline terms
     and their interactions up to a specified order.

     Parameters
     ----------
     interaction_order : int, default=1
         Maximum order of feature interactions to include.
         1 means no interactions (only main effects)
         2 means pairwise interactions
         3 means up to three-way interactions, etc.
    n_bases : int, default=25
        Total number of basis functions to target for each term.
        For k-way interactions, uses n_bases^(1/k) splines per feature.
    lam : float, default=0.6
        Smoothing parameter.
    max_iter : int, default=250
        Maximum number of iterations for the solver.

     Example
     -------
     ```
     # Create GAM with all pairwise interactions
     gam = GAM(interaction_order=2)
     gam.fit(X_train, y_train)
     gam.predict(X_test)
     ```
    """

    def __init__(
        self,
        interaction_order: int = 1,
        n_bases: int = 25,
        lam: float = 0.6,
        max_iter: int = 1000,
    ):
        if interaction_order < 1:
            raise ValueError("interaction_order must be >= 1")
        if n_bases < 4:
            raise ValueError("n_bases must be >= 4")

        self.interaction_order = interaction_order
        self.n_bases = n_bases
        self.lam = lam
        self.max_iter = max_iter
        self._is_fitted__ = False

    def _get_n_splines(self, order: int) -> int:
        """
        Calculate number of splines per feature for a given interaction order.

        Uses the order-th root of n_bases to maintain consistent total basis
        functions across different interaction orders.
        """
        return max(4, round(self.n_bases ** (1 / order)))

    def _generate_terms(self, n_features: int):
        """Generate spline terms for all features and their interactions."""
        features = list(range(n_features))
        gam_term = None

        # Add main effects (spline terms for each feature)
        n_splines_main = self._get_n_splines(1)
        for feature in features:
            term = s(feature, n_splines=n_splines_main, lam=self.lam)
            gam_term = term if gam_term is None else gam_term + term

        # Add interaction terms if interaction_order > 1
        if self.interaction_order > 1:
            for order in range(2, min(self.interaction_order + 1, n_features + 1)):
                n_splines_interaction = self._get_n_splines(order)
                for features_subset in combinations(features, order):
                    term = te(*features_subset, n_splines=n_splines_interaction, lam=self.lam)
                    gam_term = term if gam_term is None else gam_term + term

        return gam_term

    def fit(self, X, y):
        """
        Fit the GAM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_features = X.shape[1]
        terms = self._generate_terms(n_features)
        self.model = LinearGAM(terms, max_iter=self.max_iter)
        self.model.fit(X, y)
        self._is_fitted__ = True

        return self

    def predict(self, X):
        """
        Make predictions using the fitted GAM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        if not self._is_fitted__:
            raise ValueError("Model must be fitted before making predictions.")
        return self.model.predict(X)

    def __sklearn_is_fitted__(self):
        """Required for sklearn compatibility."""
        return self._is_fitted__
