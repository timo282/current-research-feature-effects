from typing import Literal, Callable
import numpy as np
import pandas as pd
from scipy import constants

from current_research_feature_effects.data_generating.data_generation import Groundtruth


class NewtonUniversalGravitationGroundtruth(Groundtruth):
    """
    Newton's law of universal gravitation groundtruth specified by the formula::
    `g(x) = G * (m1 * m2) / ((x2 - x1)**2 + (y2 -y1)**2 + (z2 - z1)**2)`
    where G is the gravitational constant.
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

    def get_theoretical_partial_dependence(
        self, feature: Literal["m_1", "m_2", "x_1", "x_2", "y_1", "y_2", "z_1", "z_2"]
    ) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for NewtonUniversalGravitation.")
