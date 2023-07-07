import math
import numpy as np
import pandas as pd
from typing import Union


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = "euclidean") -> None:
        self.k = k
        self.metric = metric
        self.train_size = None
        self.X_train = None
        self.y_train = None
        
    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        
        return f"{self.__class__.__name__} class: " + ", ".join(params)
    
    def _transform_data(self, data: Union[pd.DataFrame, pd.Series] = None) -> np.array:
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.to_numpy()
        
        raise TypeError(f"Data must be type pandas dataframe or pandas series, not {type(data)}!")
    
    def fit(self, X: pd.DataFrame = None, y: pd.Series = None) -> None:
        self.X_train = self._transform_data(X)
        self.y_train = self._transform_data(y)
        self.train_size = X.shape

    def _vote(self, neighbor_labels: np.array = None) -> int:
        if np.mean(neighbor_labels) == 0.5:
            return 1
        
        vals, counts = np.unique(neighbor_labels, return_counts=True)
        mode_value = np.argwhere(counts == np.max(counts))

        return vals[mode_value].flatten()[0]

    def predict(self, X: pd.DataFrame = None) -> np.array:
        X = self._transform_data(X)
        y_pred = np.empty(X.shape[0], dtype="int8")

        for i, sample in enumerate(X):
            idx = np.argsort([getattr(self, "_" + self.metric)(sample, x) for x in self.X_train])[:self.k]
            knn = np.array([self.y_train[i] for i in idx])
            y_pred[i] = self._vote(knn)

        return y_pred
    
    def predict_proba(self, X: pd.DataFrame = None) -> np.array:
        X = self._transform_data(X)
        y_pred = np.empty(X.shape[0])

        for i, sample in enumerate(X):
            idx = np.argsort([getattr(self, "_" + self.metric)(sample, x) for x in self.X_train])[:self.k]
            knn = np.array([self.y_train[i] for i in idx])
            y_pred[i] = np.mean(knn)

        return y_pred
    
    @staticmethod
    def _euclidean(x1: np.array = None, x2: np.array = None) -> float:
        distance = 0
        for i in range(len(x1)):
            distance += math.pow(x1[i] - x2[i], 2)
        
        return math.sqrt(distance)

    @staticmethod
    def _manhattan(x1: np.array = None, x2: np.array = None) -> float:
        distance = 0
        for i in range(len(x1)):
            distance += np.abs(x1[i] - x2[i])

        return distance

    @staticmethod
    def _chebyshev(x1: np.array = None, x2: np.array = None) -> float:
        distance = np.empty(x1.shape[0])
        for i in range(len(x1)):
            distance[i] = np.abs(x1[i] - x2[i])

        return np.max(distance)

    @staticmethod
    def _cosine(x1: np.array = None, x2: np.array = None) -> float:
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))