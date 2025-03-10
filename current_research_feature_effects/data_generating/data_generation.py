"""
This module contains classes and functions for generating synthetic data based on a groundtruth function.
"""

from abc import abstractmethod, ABC
from typing import Callable, Literal, List, Optional, Tuple
import warnings
from scipy.stats import norm, uniform, loguniform
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


class Groundtruth(ABC, BaseEstimator):
    """
    A wrapper class for a groundtruth function wrapped as fitted sklearn
    regression estimator adhering to the standard scikit-learn estimator
    interface.

    Attributes
    ----------
    _is_fitted__ : bool
        Indicates whether the estimator has been 'fitted'. This is a mock
        attribute and is set to True by default.
    _estimator_type : str
        Defines the type of the estimator as 'regressor'.
    marginal_distributions : List[Tuple[Literal["normal", "uniform"], Tuple]]
        Marginal distributions of the features. Each tuple contains the
        distribution type and its parameters. Supported distributions and
        parameters:
        - 'normal': (mean, std)
        - 'uniform': (low, high)

    correlation_matrix : np.ndarray
        Correlation matrix of the features. If None, features are independent.
    n_features : int
        Number of features.
    feature_names : List[str]
        Names of the features.
    """

    def __init__(
        self,
        marginal_distributions: List[Tuple[Literal["normal", "uniform", "loguniform"], Tuple]],
        correlation_matrix: np.ndarray,
        feature_names: List[str] = None,
        name: str = None,
    ):
        self._is_fitted__ = True
        self._estimator_type = "regressor"
        self._marginal_distributions = marginal_distributions
        self._correlation_matrix = correlation_matrix
        if correlation_matrix is not None and correlation_matrix.shape != (
            len(marginal_distributions),
            len(marginal_distributions),
        ):
            raise ValueError("Correlation matrix must be of shape (n_features, n_features).")
        self._n_features = len(marginal_distributions)
        self._feature_names = (
            feature_names if feature_names is not None else [f"x_{i + 1}" for i in range(self._n_features)]
        )
        self._name = (
            name
            if name is not None
            else (
                f"{self.__class__.__name__}({self.marginal_distributions}, {self.correlation_matrix.tolist()})".replace(
                    " ", ""
                )
                .replace('"', "")
                .replace("'", "")
            )
        )

    def __sklearn_is_fitted__(self):
        return self._is_fitted__

    @property
    def marginal_distributions(self) -> List[Tuple[Literal["normal", "uniform"], Tuple]]:
        """Marginal distributions of the main features."""
        return self._marginal_distributions

    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Correlation matrix of the features."""
        return self._correlation_matrix

    @property
    def n_features(self) -> int:
        """Total number of features."""
        return self._n_features

    @property
    def feature_names(self) -> List[str]:
        """Names of all features."""
        return self._feature_names

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return self._name

    def fit(self, X, y):
        """
        Mocks fit method for the groundtruth (does not perform any operation).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Not used.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Not used.
        """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """

    @abstractmethod
    def get_theoretical_partial_dependence(self, feature: str) -> Callable:
        """Get the theoretical partial dependence function for a feature.

        Parameters
        ----------
        feature : str
            The feature for which to compute the partial dependence function.

        Returns
        -------
        Callable
            The theoretical partial dependence function for the feature.
        """

    @abstractmethod
    def get_theoretical_accumulated_local_effects(self, feature: str) -> Callable:
        """Get the theoretical accumulated local effects function for a feature
           (up to constant terms).

        Parameters
        ----------
        feature : str
            The feature for which to compute the accumulated local effects function.

        Returns
        -------
        Callable
            The theoretical accumulated local effects function for the feature.
        """

    def get_theoretical_quantiles(self, feature: str, quantiles: np.ndarray) -> np.ndarray:
        """Get theoretical quantiles for a feature's marginal distribution.

        Parameters
        ----------
        feature : str
            The feature for which to compute the quantiles.
        quantiles : np.ndarray
            Array of quantile values between 0 and 1.

        Returns
        -------
        np.ndarray
            The theoretical quantile values for the feature.
        """
        if feature not in self.feature_names:
            raise ValueError(f"Feature {feature} not found in {self.feature_names}")

        if not np.all((quantiles >= 0) & (quantiles <= 1)):
            raise ValueError("Quantiles must be between 0 and 1")

        feature_idx = self.feature_names.index(feature)
        dist_type, params = self.marginal_distributions[feature_idx]

        if dist_type == "normal":
            mean, std = params
            return norm.ppf(quantiles, loc=mean, scale=std)
        elif dist_type == "uniform":
            low, high = params
            return uniform.ppf(quantiles, loc=low, scale=high - low)
        elif dist_type == "loguniform":
            low, high = params
            return loguniform.ppf(quantiles, low, high)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    def __str__(self):
        """Return dataset name as string."""
        return self.name


def _transform_to_target_distribution(
    data: np.ndarray, dist_type: Literal["normal", "uniform", "loguniform"], params: Tuple
) -> np.ndarray:
    """
    Transform standard normal data to a target distribution using inverse CDF.

    Parameters
    ----------
    data : np.ndarray
        The data to transform.
    dist_type : str
        The target distribution type. Supported distributions are 'normal', 'uniform', and 'loguniform'.
    params : tuple
        Tuple containing the distribution parameters.
        - For normal: (mean, std)
        - For uniform: (low, high)
        - For loguniform: (low_exp, high_exp) where bounds are 10^low_exp and 10^high_exp

    Returns
    -------
    np.ndarray
        The transformed data.
    """
    if dist_type == "normal":
        mean, std = params
        return norm.ppf(norm.cdf(data)) * std + mean
    elif dist_type == "uniform":
        low, high = params
        # Transform standard normal data to uniform
        warnings.warn(
            f"Correlations > 0 may not be preserved by the transformation (distribution: {dist_type}).", UserWarning
        )
        return uniform.ppf(norm.cdf(data), loc=low, scale=high - low)
    elif dist_type == "loguniform":
        low, high = params
        # Transform standard normal data to loguniform
        warnings.warn(
            f"Correlations > 0 may not be preserved by the transformation (distribution: {dist_type}).", UserWarning
        )
        return loguniform.ppf(norm.cdf(data), a=low, b=high)
    else:
        raise ValueError(f"Unsupported distribution type {dist_type}")


def _generate_samples(
    groundtruth: Groundtruth,
    n_samples: int,
    noise_sd: float,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for specified groundtruth and noise standard deviation.

    Parameters
    ----------
    groundtruth : Groundtruth
        A Groundtruth object from which to predict the response variable.
    n_samples : int
        Number of samples to generate.
    noise_sd : float
        Standard deviation of the Gaussian noise added to the output.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        The generated features matrix with shape (n_samples, groundtruth.n_features).
    y : np.ndarray
        The generated response variable with added noise.
    """
    generator = check_random_state(random_state)
    X = generator.normal(0, 1, size=(n_samples, groundtruth.n_features))

    # Apply the correlation matrix using Cholesky decomposition
    if groundtruth.correlation_matrix is not None and not np.array_equal(
        groundtruth.correlation_matrix, np.eye(groundtruth.n_features)
    ):
        L = np.linalg.cholesky(groundtruth.correlation_matrix)
        X = X @ L.T

    # Transform each column to the desired distribution
    for i in range(groundtruth.n_features):
        dist_type, params = groundtruth.marginal_distributions[i]
        X[:, i] = _transform_to_target_distribution(X[:, i], dist_type, params)

    y = groundtruth.predict(X) + noise_sd * generator.standard_normal(n_samples)

    return X, y


def generate_data(
    groundtruth: Groundtruth,
    n_train: int,
    n_test: int,
    snr: float,
    seed: int,
    n_val: int = None,
):
    """Generate data for training, validation (optional), and testing based on the specified groundtruth
    and signal-to-noise ratio for noise standard deviation.

    Parameters
    ----------
    groundtruth : Groundtruth
        Groundtruth object to generate the response variable.
    n_train : int
        Number of training samples to generate.
    n_test : int
        Number of test samples to generate.
    snr : float
        Signal-to-noise-ratio defining the amount of noise to add to the data.
    seed : int
        Random seed to use for reproducibility.
    n_val : int, optional
        Number of validation samples to generate, by default None. if None, no validation data is generated.

    Returns
    -------
    tuple
        If n_val is None:
            X_train, y_train, X_test, y_test : Training and test data
        If n_val is provided:
            X_train, y_train, X_val, y_val, X_test, y_test : Training, validation and test data
    """
    total_samples = n_train + n_test + (n_val if n_val is not None else 0)

    X, y = _generate_samples(
        groundtruth=groundtruth,
        n_samples=total_samples,
        noise_sd=_get_noise_sd_from_snr(snr, groundtruth),
        random_state=seed,
    )

    if n_val is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)

        return X_train, y_train, X_test, y_test

    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=n_test, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=n_val, random_state=42)

        return X_train, y_train, X_val, y_val, X_test, y_test


def _get_noise_sd_from_snr(snr: int, groundtruth: Groundtruth) -> float:
    """Calculate noise standard deviation from signal-to-noise ratio.

    Parameters
    ----------
    snr : int
        Signal-to-noise ratio.
    groundtruth : Groundtruth
        Groundtruth object to generate the response variable
        (only for range of the response variable for signal to noise ratio).

    Returns
    -------
    float
        Noise standard deviation.
    """
    _, y = _generate_samples(n_samples=100000, groundtruth=groundtruth, noise_sd=0.0)

    signal_std = np.std(y)
    noise_std = (signal_std / snr) if snr != 0 else 0.0

    return noise_std
