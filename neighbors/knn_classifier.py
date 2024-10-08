import numpy as np
import pandas as pd
from typing import Union


class KNeighborsClassifier:
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

    def _vote(self, neighbor_labels: np.array) -> int:
        vals, counts = np.unique(neighbor_labels, return_counts=True)
        return vals[np.argmax(counts)]
    
    def _calc_weights(self, neighbor_labels: np.array, idx: np.array, distance: np.array) -> tuple:
        if self.weight == "rank":
            ranks = np.arange(1, len(idx) + 1)
            weights = 1 / ranks
        elif self.weight == "distance":
            distances = distance[idx]
            weights = 1 / (distances + 1e-10)
        else:
            raise ValueError(f"Unknown weight type: {self.weight}")

        weights /= np.sum(weights)

        unique_labels = np.unique(neighbor_labels)
        aggregated_weights = np.array([np.sum(weights[neighbor_labels == label]) for label in unique_labels])

        return unique_labels, aggregated_weights

    def _calculate_distance(self, sample: np.array) -> np.array:
        return np.array([getattr(self, f"_{self.metric}")(sample, x_train) for x_train in self.X_train])

    def predict(self, X: pd.DataFrame) -> np.array:
        X = self._transform_data(X)
        y_pred = np.empty(X.shape[0], dtype="int8")

        for i, sample in enumerate(X):
            distance = self._calculate_distance(sample)
            idx = np.argsort(distance)[:self.k]
            knn = self.y_train[idx]

            if self.weight == "uniform":
                y_pred[i] = self._vote(knn)
            else:
                unique_labels, weights = self._calc_weights(neighbor_labels=knn, idx=idx, distance=distance)
                y_pred[i] = unique_labels[np.argmax(weights)]

        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        X = self._transform_data(X)
        y_pred_proba = np.empty(X.shape[0])

        for i, sample in enumerate(X):
            distance = self._calculate_distance(sample)
            idx = np.argsort(distance)[:self.k]
            knn = self.y_train[idx]

            if self.weight == "uniform":
                y_pred_proba[i] = np.mean(knn)
            else:
                unique_labels, weights = self._calc_weights(neighbor_labels=knn, idx=idx, distance=distance)
                y_pred_proba[i] = weights[unique_labels == 1][0] if 1 in unique_labels else 0

        return y_pred_proba
    
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