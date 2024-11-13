from typing import Callable, Literal
import pandas as pd
import numpy as np

from current_research_feature_effects.data_generating.data_generation import Groundtruth


class SimpleAdditiveGroundtruth(Groundtruth):
    """
    A simple additive groundtruth specified by the formula::
    `g(x) = x_1 + 0.5*x_2^2` and optionally additional noise features.
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X[:, 0] + 0.5 * X[:, 1] ** 2

    def get_theoretical_partial_dependence(self, feature: Literal["x_1", "x_2"]) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for SimpleAdditiveGroundtruth.")


class SimpleInteractionGroundtruth(Groundtruth):
    """
    A simple interaction groundtruth specified by the formula::
    `g(x) = x_1 * x_2$` and optionally additional noise features.
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X[:, 0] * X[:, 1]

    def get_theoretical_partial_dependence(self, feature: Literal["x_1", "x_2"]) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for SimpleInteractionGroundtruth.")


class SimpleCombinedGroundtruth(Groundtruth):
    """
    A simple combined additive and interaction groundtruth specified by the formula::
    `g(x) = x_1 + 0.5*x_2^2 + x_1*x_2` and optionally additional noise features.
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X[:, 0] + 0.5 * X[:, 1] ** 2 + X[:, 0] * X[:, 1]

    def get_theoretical_partial_dependence(self, feature: Literal["x_1", "x_2"]) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for SimpleCombinedGroundtruth.")
