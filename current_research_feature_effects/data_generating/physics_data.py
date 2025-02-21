"""
This module contains classes for generating data based on physics laws.
Currently, the Feynman equations I.9.18 and I.29.16 are implemented.
"""

from typing import Callable
import numpy as np
import pandas as pd
from scipy import constants

from current_research_feature_effects.data_generating.data_generation import Groundtruth


class NewtonUniversalGravitationGroundtruth(Groundtruth):
    """
    Newton's law of universal gravitation groundtruth specified by the formula::
    `g(x) = G * (m1 * m2) / ((x2 - x1)**2 + (y2 -y1)**2 + (z2 - z1)**2)`
    where G is the gravitational constant
    (Feynman I.9.18).
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, 6)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (
            constants.G
            * X[:, 0]
            * X[:, 1]
            / ((X[:, 2] - X[:, 3]) ** 2 + (X[:, 4] - X[:, 5]) ** 2 + (X[:, 6] - X[:, 7]) ** 2)
        )

    def get_theoretical_partial_dependence(self, feature: str) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for NewtonUniversalGravitation.")

    def get_theoretical_accumulated_local_effects(self, feature: str) -> Callable:
        raise NotImplementedError(
            "Theoretical accumulated local effects not implemented for NewtonUniversalGravitation."
        )


class WaveInterferenceGroundtruth(Groundtruth):
    """
    Wave interference groundtruth specified by the formula:
    `x = sqrt(x1^2 + x2^2 + 2*x1*x2*cos(theta1 - theta2))`
    where x1, x2 are wavelengths and theta1, theta2 are angles
    (Feynman I.29.16).
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, 4)
            The input samples with columns [wavelength1, wavelength2, angle1, angle2].

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        x1, x2 = X[:, 0], X[:, 1]
        theta1, theta2 = X[:, 2], X[:, 3]

        return np.sqrt(x1**2 + x2**2 + 2 * x1 * x2 * np.cos(theta1 - theta2))

    def get_theoretical_partial_dependence(self, feature: str) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for WaveInterference.")

    def get_theoretical_accumulated_local_effects(self, feature: str) -> Callable:
        raise NotImplementedError("Theoretical accumulated local effects not implemented for WaveInterference.")
