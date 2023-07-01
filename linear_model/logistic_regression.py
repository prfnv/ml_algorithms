import logging
import numpy as np
import pandas as pd


_logger = logging.getLogger(__name__)


class MyLogReg:
    EPS = 1e-15
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: float = 0.1,
) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__} class: " + ", ".join(params)
    
    def _get_bias(self, X: pd.DataFrame = None) -> pd.DataFrame:
        features = X.copy()
        list_ones = np.ones(X.shape[0], dtype="int8")
        features.insert(loc=0, column="ones", value=list_ones)

        return features
    
    def fit(self, X: pd.DataFrame = None, y: pd.Series = None, verbose: int = 0) -> None:
        features = self._get_bias(X)
        self.weights = np.ones(features.shape[1])

        for i in range(self.n_iter):
            y_hat = 1 / (1 + np.exp(-features @ self.weights))
            log_loss = -1 * (y * np.log(y_hat + self.EPS) + (1 - y) * np.log(1 - y_hat + self.EPS)).sum() / features.shape[0]
            grad = (y_hat - y) @ features / features.shape[0]
            self.weights -= self.learning_rate * grad.to_numpy()

            if verbose and i % verbose == 0:
                _logger.info(f"{i if i else 'start'} | loss: {log_loss}")
                
    def get_coef(self):
        return np.mean(self.weights[1:])