"""Training loop and evaluation metrics."""

import numpy as np
from typing import Optional, Dict, List, Callable


def mse(Y_pred: np.ndarray, Y: np.ndarray) -> float:
    """Mean squared error."""
    if Y.ndim == 1:
        Y = Y[:, None]
    return float(np.mean(np.sum((Y_pred - Y) ** 2, axis=1)))


def accuracy(Y_pred: np.ndarray, Y: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(Y_pred.flatten() == Y.flatten()))


class Trainer:
    """Mini-batch training loop with per-epoch metric tracking."""

    def __init__(self, model, optimizer, metric: Callable = mse,
                 epochs: int = 100, batch_size: int = 32,
                 shuffle: bool = True, seed: int = 0):
        self.model = model
        self.optimizer = optimizer
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.history: Dict[str, List[float]] = {"train": [], "val": []}

    def fit(self, X: np.ndarray, Y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            Y_val: Optional[np.ndarray] = None) -> "Trainer":
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]

        for _ in range(self.epochs):
            idx = rng.permutation(n) if self.shuffle else np.arange(n)

            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                self.optimizer.step(self.model, X[batch], Y[batch])

            self.history["train"].append(self.metric(self.model.predict(X), Y))
            if X_val is not None and Y_val is not None:
                self.history["val"].append(self.metric(self.model.predict(X_val), Y_val))

        return self
