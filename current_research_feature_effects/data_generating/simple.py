from typing import Callable
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

    def get_theoretical_partial_dependence(self, feature: str) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for SimpleAdditiveGroundtruth.")

    def get_theoretical_accumulated_local_effects(self, feature: str) -> Callable:
        raise NotImplementedError(
            "Theoretical accumulated local effects not implemented for SimpleAdditiveGroundtruth."
        )


class SimpleInteractionGroundtruth(Groundtruth):
    """
    A simple interaction groundtruth specified by the formula::
    `g(x) = x_1 * x_2` and optionally additional noise features.
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

    def get_theoretical_partial_dependence(self, feature: str) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for SimpleInteractionGroundtruth.")

    def get_theoretical_accumulated_local_effects(self, feature: str) -> Callable:
        raise NotImplementedError(
            "Theoretical accumulated local effects not implemented for SimpleInteractionGroundtruth."
        )


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

    def get_theoretical_partial_dependence(self, feature: str) -> Callable:
        marginal_distrs = [feature[0] for feature in self.marginal_distributions]
        if not all(distr == "normal" for distr in marginal_distrs):
            raise NotImplementedError("Only normal marginals are supported.")

        if feature == self.feature_names[0]:
            mu2 = self.marginal_distributions[1][1][0]
            sigmasq2 = self.marginal_distributions[1][1][1]

            def partial_dependence(x1):
                return (1 + mu2) * x1 + 0.5 * (sigmasq2 + mu2**2)

        elif feature == self.feature_names[1]:
            mu1 = self.marginal_distributions[0][1][0]

            def partial_dependence(x2):
                return 0.5 * x2**2 + mu1 * x2 + mu1

        else:

            def partial_dependence(x):
                return np.zeros_like(x)

        return partial_dependence

    def get_theoretical_accumulated_local_effects(self, feature: str) -> Callable:
        marginal_distrs = [feature[0] for feature in self.marginal_distributions]
        if not all(distr == "normal" for distr in marginal_distrs):
            raise NotImplementedError("Only normal marginals are supported.")

        mu1 = self.marginal_distributions[0][1][0]
        mu2 = self.marginal_distributions[1][1][0]
        sigmasq1 = self.marginal_distributions[0][1][1]
        sigmasq2 = self.marginal_distributions[1][1][1]
        rho = self.correlation_matrix[0, 1]
        beta12 = rho * sigmasq1 / sigmasq2
        beta21 = rho * sigmasq2 / sigmasq1

        if feature == self.feature_names[0]:

            def accumulated_local_effects(x1):
                return (beta21 / 2) * x1**2 + (1 + mu2 - beta21 * mu1) * x1

        elif feature == self.feature_names[1]:
            mu1 = self.marginal_distributions[0][1][0]

            def accumulated_local_effects(x2):
                return (0.5 + mu1 + beta12 / 2) * x2**2 - beta12 * mu2 * x2

        else:

            def accumulated_local_effects(x):
                return np.zeros_like(x)

        return accumulated_local_effects
