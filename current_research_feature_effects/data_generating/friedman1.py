"""
This module contains the groundtruth class for the Friedman1 function.
"""

from typing import Callable
import pandas as pd
import numpy as np
from scipy.special import sici

from current_research_feature_effects.data_generating.data_generation import Groundtruth


class Friedman1Groundtruth(Groundtruth):
    """
    A groundtruth class for the Friedman1 function, which is defined as::

        y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4].
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth (Friedman1) for each
        sample in X. The output `y` is created according to the formula:

        y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
        + 10 * X[:, 3] + 5 * X[:, 4].

        Parameters
        ----------
        X : array-like of shape (n_samples, 5)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]

    def get_theoretical_partial_dependence(self, feature: str) -> Callable:
        """Get the theoretical partial dependence function for a feature.
        Only uniform feature distribution is supported.

        Parameters
        ----------
        feature : str
            The feature for which to compute the partial dependence function.

        Returns
        -------
        Callable
            The theoretical partial dependence function for the feature.
        """
        if not all(distr[0] == "uniform" for distr in self.marginal_distributions):
            raise ValueError("Only uniform feature distribution is supported.")

        mu3 = (self.marginal_distributions[2][1][0] + self.marginal_distributions[2][1][1]) / 2
        mu4 = (self.marginal_distributions[3][1][0] + self.marginal_distributions[3][1][1]) / 2
        mu5 = (self.marginal_distributions[4][1][0] + self.marginal_distributions[4][1][1]) / 2
        sigmasq3 = (self.marginal_distributions[2][1][1] - self.marginal_distributions[2][1][0]) ** 2 / 12
        a1 = self.marginal_distributions[0][1][0] + 1e-15  # for numerical stability
        b1 = self.marginal_distributions[0][1][1]
        a2 = self.marginal_distributions[1][1][0] + 1e-15  # for numerical stability
        b2 = self.marginal_distributions[1][1][1]
        int12 = (
            10
            / np.pi
            * (
                sici(np.pi * b1 * a2)[1]
                - sici(np.pi * a1 * a2)[1]
                - sici(np.pi * b1 * b2)[1]
                + sici(np.pi * a1 * b2)[1]
            )
        )

        if feature == "x_1":
            a = self.marginal_distributions[0][1][0]
            b = self.marginal_distributions[0][1][1]

            def partial_dependence(x1):
                return (
                    10 / (np.pi * x1) * (np.cos(np.pi * x1 * a) - np.cos(np.pi * x1 * b))
                    + 20 * mu3**2
                    + 20 * sigmasq3
                    - 20 * mu3
                    + 5
                    + 10 * mu4
                    + 5 * mu5
                )

        elif feature == "x_2":
            a = self.marginal_distributions[1][1][0]
            b = self.marginal_distributions[1][1][1]

            def partial_dependence(x2):
                return (
                    10 / (np.pi * x2) * (np.cos(np.pi * x2 * a) - np.cos(np.pi * x2 * b))
                    + 20 * mu3**2
                    + 20 * sigmasq3
                    - 20 * mu3
                    + 5
                    + 10 * mu4
                    + 5 * mu5
                )

        elif feature == "x_3":

            def partial_dependence(x3):
                return int12 + 20 * (x3 - 0.5) ** 2 + 10 * mu4 + 5 * mu5

        elif feature == "x_4":

            def partial_dependence(x4):
                return int12 + 20 * mu3**2 + 20 * sigmasq3 - 20 * mu3 + 5 + 10 * x4 + 5 * mu5

        elif feature == "x_5":

            def partial_dependence(x5):
                return int12 + 20 * mu3**2 + 20 * sigmasq3 - 20 * mu3 + 5 + 10 * mu4 + 5 * x5

        else:

            def partial_dependence(x):
                return np.ones_like(x) * int12 + 20 * mu3**2 + 20 * sigmasq3 - 20 * mu3 + 5 + 10 * mu4 + 5 * mu5

        return partial_dependence

    def get_theoretical_accumulated_local_effects(self, feature: str) -> Callable:
        """Get the theoretical accumulated local effects function for a feature.
        Only uniform feature distribution is supported.

        Parameters
        ----------
        feature : str
            The feature for which to compute the accumulated local effects function.

        Returns
        -------
        Callable
            The theoretical accumulated local effects function for the feature.
        """
        marginal_distrs = [feature[0] for feature in self.marginal_distributions]
        if not all(distr == "uniform" for distr in marginal_distrs):
            raise NotImplementedError("Only uniform marginals are supported.")

        if not np.equal(self.correlation_matrix[:5, :5].astype(float), np.eye(5)).all():
            raise NotImplementedError("Only independent features are supported.")

        if feature == "x_1":
            a = self.marginal_distributions[0][1][0]
            b = self.marginal_distributions[0][1][1]

            def accumulated_local_effects(x1):
                return 10 / (b - a) * (-1 / (np.pi * x1) * (np.cos(np.pi * x1 * b) - np.cos(np.pi * x1 * a)))

        elif feature == "x_2":
            a = self.marginal_distributions[1][1][0]
            b = self.marginal_distributions[1][1][1]

            def accumulated_local_effects(x2):
                return 10 / (b - a) * (-1 / (np.pi * x2) * (np.cos(np.pi * x2 * b) - np.cos(np.pi * x2 * a)))

        elif feature == "x_3":
            a = self.marginal_distributions[2][1][0]
            b = self.marginal_distributions[2][1][1]

            def accumulated_local_effects(x3):
                return 20 * x3**2 - 20 * x3

        elif feature == "x_4":
            a = self.marginal_distributions[3][1][0]
            b = self.marginal_distributions[3][1][1]

            def accumulated_local_effects(x4):
                return 10 * x4

        elif feature == "x_5":
            a = self.marginal_distributions[4][1][0]
            b = self.marginal_distributions[4][1][1]

            def accumulated_local_effects(x5):
                return 5 * x5

        else:

            def accumulated_local_effects(x):
                return np.zeros_like(x)

        return accumulated_local_effects
