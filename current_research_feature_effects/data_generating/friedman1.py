from typing import Literal, Callable
import pandas as pd
import numpy as np
from scipy.integrate import quad

from current_research_feature_effects.data_generating.data_generation import Groundtruth


class Friedman1Groundtruth(Groundtruth):
    """
    A groundtruth class for the Friedman1 function, which is defined as::

        y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4].
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth (Friedman1) for each
        sample in X. The output `y` is created according to the formula::

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

    def get_theoretical_partial_dependence(self, feature: Literal["x_1", "x_2", "x_3", "x_4", "x_5"]) -> Callable:
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

        integrand12_over_x2 = lambda x2, x1: 10 * np.sin(np.pi * x1 * x2)  # noqa: E731
        integrand12_over_x1 = lambda x1, x2: 10 * np.sin(np.pi * x1 * x2)  # noqa: E731
        integrand3_over_x3 = lambda x3: 20 * (x3 - 0.5) ** 2  # noqa: E731

        outer_integral = lambda x1: quad(integrand12_over_x2, 0, 1, args=(x1,))[0]  # noqa: E731
        complete_integral_12, _ = quad(outer_integral, 0, 1)
        integral_3, _ = quad(integrand3_over_x3, 0, 1)

        if feature == "x_1":

            def partial_dependence(x1):
                integral_12, _ = quad(integrand12_over_x2, 0, 1, args=(x1,))
                return integral_12 + integral_3 + 10 * 0.5 + 5 * 0.5

        elif feature == "x_2":

            def partial_dependence(x2):
                integral_12, _ = quad(integrand12_over_x1, 0, 1, args=(x2,))
                return integral_12 + integral_3 + 10 * 0.5 + 5 * 0.5

        elif feature == "x_3":

            def partial_dependence(x3):
                return complete_integral_12 + 20 * (x3 - 0.5) ** 2 + 10 * 0.5 + 5 * 0.5

        elif feature == "x_4":

            def partial_dependence(x4):
                return complete_integral_12 + integral_3 + 10 * x4 + 2.5

        elif feature == "x_5":

            def partial_dependence(x5):
                return complete_integral_12 + integral_3 + 5 + 5 * x5

        return partial_dependence
