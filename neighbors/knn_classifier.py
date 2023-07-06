import math
import numpy as np
import pandas as pd
from typing import Union


def euclidean_distance(x1: np.array = None, x2: np.array = None) -> float:
    distance = 0
    for i in range(len(x1)):
        distance += math.pow(x1[i] - x2[i], 2)
    
    return math.sqrt(distance)


class MyKNNClf:
    def __init__(self, k: int = 3) -> None:
        self.k = k
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
            idx = np.argsort([euclidean_distance(sample, x) for x in self.X_train])[:self.k]
            knn = np.array([self.y_train[i] for i in idx])
            y_pred[i] = self._vote(knn)

        return y_pred
    
    def predict_proba(self, X: pd.DataFrame = None) -> np.array:
        X = self._transform_data(X)
        y_pred = np.empty(X.shape[0])

        for i, sample in enumerate(X):
            idx = np.argsort([euclidean_distance(sample, x) for x in self.X_train])[:self.k]
            knn = np.array([self.y_train[i] for i in idx])
            y_pred[i] = np.mean(knn)

        return y_pred
    

if __name__ == "__main__":
    data = pd.read_csv('/Users/paparfen/vscode/ml_algorithms/data/data_banknote_authentication.txt', sep=",", header=None)
    data.columns = ["variance", "skewness", "kurtosis", "entropy", 'target']
    # data = pd.read_csv('/Users/paparfen/vscode/ml_algorithms/data/iris.csv')
    # data['variety'] = data['variety'].map({'Setosa': 0, 'Virginica': 1})
    X = data.drop('target', axis=1)
    y = data['target']

    model = MyKNNClf()
    model.fit(X, y)
    print(model.predict(X.head()))
