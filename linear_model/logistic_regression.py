import random
import logging
import numpy as np
import pandas as pd
from typing import Union, Callable


_logger = logging.getLogger(__name__)


def _binary_clf_curve(y_true: pd.Series = None, y_pred: np.array = None):
    desc_score_indices = np.argsort(y_pred)[::-1]
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_indices = np.where(np.diff(y_pred))[0]
    end = np.array([y_true.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    thresholds = y_pred[threshold_indices]
    tps = np.cumsum(y_true)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    return tps, fps, thresholds


class MyLogReg:
    EPS = 1e-15
    def __init__(
        self,
        n_iter: int = 10,
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
            y_hat = 1 / (1 + np.exp(-features @ self.weights))
            loss = (y_hat - y)
            log_loss = -1 * (y * np.log(y_hat + self.EPS) + (1 - y) * np.log(1 - y_hat + self.EPS)).sum() / features.shape[0]
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float):
                    self.sgd_sample = round(features.shape[0] * self.sgd_sample)
                sample_rows_idx = random.sample(range(features.shape[0]), self.sgd_sample)
                grad = loss.iloc[sample_rows_idx] @ features.iloc[sample_rows_idx, :] / self.sgd_sample
            else:
                grad = loss @ features / features.shape[0]

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
                    if self.metric == "roc_auc":
                        y_pred = self.predict_proba(X)
                    else:
                        y_pred = self.predict(X)
                    self.score = getattr(self, "_" + self.metric)(y, y_hat)
                    _logger.info(f"{i if i else 'start'} | loss: {log_loss} | {self.metric}: {self.score}")
                else:
                    _logger.info(f"{i if i else 'start'} | loss: {log_loss}")

        if self.metric:
            if self.metric == "roc_auc":
                y_pred = self.predict_proba(X)
            else:
                y_pred = self.predict(X)
            self.score = getattr(self, "_" + self.metric)(y, y_pred)

    def get_coef(self):
        return np.mean(self.weights[1:])
    
    def predict_proba(self, X: pd.DataFrame = None) -> np.array:
        features = self._get_bias(X)
        y_pred = 1 / (1 + np.exp(-features @ self.weights))

        return y_pred
    
    def predict(self, X: pd.DataFrame = None) -> np.array:
        y_pred = (self.predict_proba(X) > 0.5).astype("int")

        return y_pred
    
    def get_best_score(self) -> float:
        return self.score
    
    @staticmethod
    def _accuracy(y_true: pd.Series = None, y_pred: np.array = None) -> float:
        score = y_true == y_pred

        return np.average(score)
    
    @staticmethod
    def _precision(y_true: pd.Series = None, y_pred: np.array = None) -> float:
        tp_sum = np.sum(np.logical_and(y_pred == 1, y_true == 1))
        fp_sum = np.sum(np.logical_and(y_pred == 1, y_true == 0))
        score = tp_sum / (tp_sum + fp_sum)

        return score

    @staticmethod
    def _recall(y_true: pd.Series = None, y_pred: np.array = None) -> float:
        tp_sum = np.sum(np.logical_and(y_pred == 1, y_true == 1))
        fn_sum = np.sum(np.logical_and(y_pred == 0, y_true == 1))
        score = tp_sum / (tp_sum + fn_sum)

        return score

    @staticmethod
    def _f1(y_true: pd.Series = None, y_pred: np.array = None) -> float:
        tp_sum = np.sum(np.logical_and(y_pred == 1, y_true == 1))
        fp_sum = np.sum(np.logical_and(y_pred == 1, y_true == 0))
        fn_sum = np.sum(np.logical_and(y_pred == 0, y_true == 1))
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)

        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def _roc_auc(y_true: pd.Series = None, y_pred: np.array = None) -> float:
        if np.unique(y_true).size != 2:
            raise ValueError(
                "Only two class should be present in y_true. ROC AUC score "
                "is not defined in that case."
            )
        
        y_true = y_true.to_numpy()
        tps, fps, _ = _binary_clf_curve(y_true, y_pred)

        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        zero = np.array([0])
        tpr_diff = np.hstack((np.diff(tpr), zero))
        fpr_diff = np.hstack((np.diff(fpr), zero))
        auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2

        return auc