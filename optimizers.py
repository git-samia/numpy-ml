"""Gradient-based optimizers implemented from scratch."""

import numpy as np
from typing import Optional


class SGD:
    """Mini-batch stochastic gradient descent with fixed learning rate."""

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    def step(self, model, X: np.ndarray, Y: np.ndarray):
        model.W -= self.lr * model.gradient(X, Y)


class Adam:
    """Adam optimizer with bias-corrected first and second moment estimates.

    Reference: Kingma & Ba, 'Adam: A Method for Stochastic Optimization' (2014)
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def step(self, model, X: np.ndarray, Y: np.ndarray):
        if self.m is None or self.m.shape != model.W.shape:
            self.m = np.zeros_like(model.W)
            self.v = np.zeros_like(model.W)
            self.t = 0

        self.t += 1
        g = model.gradient(X, Y)

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        model.W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ProximalGD:
    """Proximal gradient descent for L1-regularized (lasso) models.

    Applies the smooth-loss gradient step followed by the
    soft-thresholding proximal operator for the L1 penalty.
    """

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    @staticmethod
    def _soft_threshold(W: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(W) * np.maximum(np.abs(W) - threshold, 0.0)

    def step(self, model, X: np.ndarray, Y: np.ndarray):
        W_tilde = model.W - self.lr * model.loss_gradient(X, Y)
        model.W = self._soft_threshold(W_tilde, self.lr * model.alpha)
