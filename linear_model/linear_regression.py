import random
import logging
import numpy as np
import pandas as pd
from typing import Union, Callable


_logger = logging.getLogger(__name__)


class MyLineReg:
    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: Union[float, Callable] = 0.1,
        metric: str = None,
        reg: str = None,
        l1_coef: float = 0.0,
        l2_coef: float = 0.0,
        sgd_sample: Union[int, float] = None,
        random_state: int = 42
) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        
    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__} class: " + ", ".join(params)
    
    def _get_bias(self, X: pd.DataFrame = None) -> pd.DataFrame:
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
        random.seed(self.random_state)
        features = self._get_bias(X)
        self.weights = np.ones(features.shape[1])

        for i in range(self.n_iter):
            y_hat = features @ self.weights
            loss = (y_hat - y)
            mse = ((loss.to_numpy() ** 2).sum() / loss.size)
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float):
                    self.sgd_sample = round(features.shape[0] * self.sgd_sample)
                sample_rows_idx = random.sample(range(features.shape[0]), self.sgd_sample)
                grad = loss.iloc[sample_rows_idx] @ features.iloc[sample_rows_idx, :] * 2 / self.sgd_sample
            else:
                grad = loss @ features * 2 / features.shape[0]
                
            if self.reg is not None:
                grad += self._regularization(
                    reg=self.reg,
                    l1_coef=self.l1_coef,
                    l2_coef=self.l2_coef,
                    weights=self.weights
                )

            if isinstance(self.learning_rate, Callable):
                self.weights -= self.learning_rate(i+1) * grad.to_numpy()
            else: 
                self.weights -= self.learning_rate * grad.to_numpy()

            if verbose and i % verbose == 0:
                if self.metric:
                    self.score = getattr(self, "_" + self.metric)(y, y_hat)
                    _logger.info(f"{i if i else 'start'} | loss: {mse} | {self.metric}: {self.score}")
                else:
                    _logger.info(f"{i if i else 'start'} | loss: {mse}")
                
        y_pred = self.predict(X)
        if self.metric:
            self.score = getattr(self, "_" + self.metric)(y, y_pred)
                
    def get_coef(self):
        return np.mean(self.weights[1:])
    
    def predict(self, X: pd.DataFrame = None) -> float:
        features = self._get_bias(X)
        y_pred = features @ self.weights

        return y_pred
    
    def get_best_score(self) -> float:
        return self.score
