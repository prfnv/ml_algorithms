import logging
import numpy as np
import pandas as pd


_logger = logging.getLogger(__name__)


class MyLineReg:
    def __init__(
        self,
        n_iter: int = None,
        learning_rate: float = None,
        metric: str = None,
        reg: str = None,
        l1_coef: float = 0.0,
        l2_coef: float = 0.0
) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def _get_bias(self, X: pd.DataFrame = None):
        features = X.copy()
        list_ones = np.ones(X.shape[0], dtype="int8")
        features.insert(loc=0, column="ones", value=list_ones)

        return features

    @staticmethod
    def _mae(y_true: np.array = None, y_pred: np.array = None):
        return np.average(np.abs(y_pred - y_true), axis=0)

    @staticmethod
    def _mse(y_true: np.array = None, y_pred: np.array = None):
        output_errors = np.average((y_true - y_pred) ** 2, axis=0)

        return np.average(output_errors)
    
    @staticmethod
    def _rmse(y_true: np.array = None, y_pred: np.array = None):
        output_errors = np.sqrt(np.average((y_true - y_pred) ** 2, axis=0))

        return np.average(output_errors)

    @staticmethod
    def _mape(y_true: np.array = None, y_pred: np.array = None):
        epsilon = np.finfo(np.float64).eps
        mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)

        return 100 * np.average(mape, axis=0)

    @staticmethod
    def _r2(y_true: np.array = None, y_pred: np.array = None):
        numerator = ((y_true - y_pred) ** 2).sum(axis=0)
        denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
        output_scores = 1 - (numerator / denominator)

        return output_scores
    
    def _regularization(
        self,
        reg: str = None,
        l1_coef: float = 0.0,
        l2_coef: float = 0.0,
        weights: np.array = None
    ):
        if reg == "l1":
            reg = l1_coef * np.sign(weights)
        elif reg == "l2":
            reg = l2_coef * 2 * weights
        else:
            reg = l1_coef * np.sign(weights) + l2_coef * 2 * weights
            
        return reg

    def fit(self, X: pd.DataFrame = None, y: pd.Series = None, verbose: int = 0) -> None:
        features = self._get_bias(X)
        self.weights = np.ones(features.shape[1], dtype="int8")

        for i in range(self.n_iter):
            y_hat = features @ self.weights
            loss = (y_hat - y)
            mse = ((loss.to_numpy() ** 2).sum() / loss.size)
            grad = loss @ features * 2 / features.shape[0]
            if self.reg is not None:
                grad += self._regularization(
                    reg=self.reg,
                    l1_coef=self.l1_coef,
                    l2_coef=self.l2_coef,
                    weights=self.weights
                ) 
            self.weights = self.weights - self.learning_rate * grad.to_numpy()
            if verbose and self.metric and not i % verbose:
                self.score = getattr(self, "_" + self.metric)(y, y_hat)
                _logger.info(f"{i if i else 'start'} | loss: {mse} | {self.metric}: {self.score}")
            else:
                _logger.info(f"{i if i else 'start'} | loss: {mse}")
                
        y_pred = self.predict(X)
        if self.metric:
            self.score = getattr(self, "_" + self.metric)(y, y_pred)
                
    def get_coef(self):
        return np.sum(self.weights[1:])
    
    def predict(self, X: pd.DataFrame = None) -> float:
        features = self._get_bias(X)
        y_pred = features @ self.weights

        return y_pred
    
    def get_best_score(self) -> float:
        return self.score


if __name__ == "__main__":
    lin_reg = MyLineReg(n_iter=10, learning_rate=0.5)
    print(lin_reg)
