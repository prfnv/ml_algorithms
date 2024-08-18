import numpy as np
import pandas as pd
from typing import Union


class KNeighborsRegressor:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform") -> None:
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = None
        self.X_train = None
        self.y_train = None

    def __repr__(self) -> str:
        params = ", ".join([f"{key}={value}" for key, value in self.__dict__.items()])
        
        return f"{self.__class__.__name__} class: {params}"
    
    def _transform_data(self, data: Union[pd.DataFrame, pd.Series]) -> np.array:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy()
        
        raise TypeError(f"Data must be of type pandas DataFrame or Series, not {type(data)}!")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X_train = self._transform_data(X)
        self.y_train = self._transform_data(y)
        self.train_size = X.shape

    def _calc_weights(self, distances: np.array, idx: np.array) -> np.array:
        if self.weight == "rank":
            ranks = np.arange(1, len(idx) + 1)
            inv_ranks = 1 / ranks
            weights = inv_ranks / inv_ranks.sum()
        elif self.weight == "distance":
            inv_distances = 1 / distances[idx]
            weights = inv_distances / inv_distances.sum()
        else:
            weights = np.ones_like(idx) / len(idx)

        return weights

    def _calculate_distance(self, sample: np.array) -> np.array:
        return np.array([getattr(self, f"_{self.metric}")(sample, x_train) for x_train in self.X_train])

    def predict(self, X: pd.DataFrame) -> np.array:
        X = self._transform_data(X)
        y_pred = np.empty(X.shape[0])

        for i, sample in enumerate(X):
            distance = self._calculate_distance(sample)
            idx = np.argsort(distance)[:self.k]
            knn = self.y_train[idx]
            weights = self._calc_weights(distances=distance, idx=idx)
            y_pred[i] = np.dot(weights, knn)

        return y_pred
    
    @staticmethod
    def _euclidean(x1: np.array, x2: np.array) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    @staticmethod
    def _manhattan(x1: np.array, x2: np.array) -> float:
        return np.sum(np.abs(x1 - x2))

    @staticmethod
    def _chebyshev(x1: np.array, x2: np.array) -> float:
        return np.max(np.abs(x1 - x2))

    @staticmethod
    def _cosine(x1: np.array, x2: np.array) -> float:
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
# Failed test #2 of 3. You answer was: 453.5610673122. Correct answer was: 375.6835507016