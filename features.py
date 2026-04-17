"""Feature transformations for linear models."""

import numpy as np
from typing import Optional


class PolynomialFeatures:
    """Polynomial feature expansion up to a given degree.

    Transforms [x1, x2] with degree=2 and cross_terms=True into:
    [1, x1, x2, x1^2, x2^2, x1*x2]
    """

    def __init__(self, degree: int = 2, cross_terms: bool = True):
        self.degree = degree
        self.cross_terms = cross_terms

    def fit(self, X: np.ndarray) -> "PolynomialFeatures":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        parts = [np.ones((n, 1)), X]

        for deg in range(2, self.degree + 1):
            parts.append(X ** deg)

        if self.cross_terms and self.degree >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    parts.append((X[:, i] * X[:, j])[:, None])

        return np.hstack(parts)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class RBFFeatures:
    """Radial basis function feature mapping.

    Selects random training points as centers and maps each input via:
    phi_j(x) = exp(-||x - c_j||^2 / (2 * sigma^2))
    """

    def __init__(self, n_centers: int = 50, sigma: float = 1.0, seed: int = 42):
        self.n_centers = n_centers
        self.sigma = sigma
        self.seed = seed
        self.centers: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RBFFeatures":
        rng = np.random.default_rng(self.seed)
        k = min(self.n_centers, X.shape[0])
        idx = rng.choice(X.shape[0], size=k, replace=False)
        self.centers = X[idx]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.centers is not None, "Call fit() before transform()."
        diffs = X[:, None, :] - self.centers[None, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        Phi = np.exp(-sq_dists / (2 * self.sigma ** 2))
        return np.hstack([np.ones((X.shape[0], 1)), Phi])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
