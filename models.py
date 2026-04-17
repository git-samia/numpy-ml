"""Linear models implemented from scratch using NumPy."""

import numpy as np


class LinearRegression:
    """Ordinary least squares regression.

    Minimizes (1/2n) * ||XW - Y||^2 via gradient descent.
    """

    def __init__(self, n_features: int, n_outputs: int = 1):
        self.W = np.zeros((n_features, n_outputs))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W

    def gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if Y.ndim == 1:
            Y = Y[:, None]
        return (X.T @ (self.predict(X) - Y)) / X.shape[0]


class RidgeRegression(LinearRegression):
    """L2-regularized linear regression.

    Adds (alpha/2) * ||W||_F^2 penalty to the squared-error loss,
    shrinking all weights toward zero to reduce overfitting.
    """

    def __init__(self, n_features: int, n_outputs: int = 1, alpha: float = 1.0):
        super().__init__(n_features, n_outputs)
        self.alpha = alpha

    def gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return super().gradient(X, Y) + self.alpha * self.W


class LassoRegression(LinearRegression):
    """L1-regularized linear regression.

    Uses the subgradient sign(W) for the non-differentiable L1 term.
    For better convergence, pair with ProximalGD instead of plain SGD.
    """

    def __init__(self, n_features: int, n_outputs: int = 1, alpha: float = 1.0):
        super().__init__(n_features, n_outputs)
        self.alpha = alpha

    def gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return super().gradient(X, Y) + self.alpha * np.sign(self.W)

    def loss_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Squared-error gradient only (no L1 term), used by ProximalGD."""
        return super().gradient(X, Y)


class SoftmaxClassifier:
    """Multinomial logistic regression with cross-entropy loss.

    Supports optional L1 or L2 regularization via the penalty parameter.
    """

    def __init__(self, n_features: int, n_classes: int,
                 alpha: float = 0.0, penalty: str = "none"):
        self.W = np.zeros((n_features, n_classes))
        self.n_classes = n_classes
        self.alpha = alpha
        self.penalty = penalty

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    def _one_hot(self, Y: np.ndarray, n: int) -> np.ndarray:
        if Y.ndim == 1 or Y.shape[1] == 1:
            Y_oh = np.zeros((n, self.n_classes))
            Y_oh[np.arange(n), Y.flatten().astype(int)] = 1.0
            return Y_oh
        return Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W).argmax(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W)

    def gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        G = (X.T @ (self._softmax(X @ self.W) - self._one_hot(Y, n))) / n
        if self.alpha > 0:
            if self.penalty == "l2":
                G += self.alpha * self.W
            elif self.penalty == "l1":
                G += self.alpha * np.sign(self.W)
        return G

    def loss_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Cross-entropy gradient without regularization, used by ProximalGD."""
        n = X.shape[0]
        return (X.T @ (self._softmax(X @ self.W) - self._one_hot(Y, n))) / n
