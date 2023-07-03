import pandas as pd
import numpy as np
from typing import Tuple


def _information_gain(y: pd.Series = None) -> float:
    if y.nunique() != 1:
        first_class = ((y.value_counts()[0] / y.shape[0]) * np.log2(y.value_counts()[0] / y.shape[0]))
        second_class = ((y.value_counts()[1] / y.shape[0]) * np.log2(y.value_counts()[1] / y.shape[0]))

        return -(first_class + second_class)
        
    return 0


def get_best_split(
    X: pd.DataFrame = None,
    y: pd.Series = None
) -> Tuple[str, float, float]:
    best_thresholds = {}
    for col in X.columns:
        ig_start = _information_gain(y=y)
        unique_values = sorted(X[col].unique())
        thresholds = [np.mean([unique_values[i], unique_values[i+1]]) for i in range(len(unique_values)-1)]
        for threshold in thresholds:
            left = X[X[col] <= threshold][[col]].index.values
            right = X[X[col] > threshold][[col]].index.values
            
            ig_left = _information_gain(y=y.iloc[left])
            ig_right = _information_gain(y=y.iloc[right])
            ig = ig_start - y.iloc[left].shape[0] / y.shape[0] * ig_left - y.iloc[right].shape[0] / y.shape[0] * ig_right

            if col not in best_thresholds:
                best_thresholds[col] = {
                    'split_value': threshold,
                    'ig': ig
                }
            elif ig > best_thresholds[col]['ig']:
                best_thresholds[col]['split_value'] = threshold
                best_thresholds[col]['ig'] = ig
            else:
                continue

    best = sorted(best_thresholds.items(), key=lambda x: x[1]['ig'], reverse=True)[0]

    return best[0], best[1]['split_value'], best[1]['ig']


class MyTreeClf:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.__dict__.items()]

        return f"{self.__class__.__name__} class: " + ", ".join(params)


if __name__ == "__main__":
    data = pd.read_csv('/Users/paparfen/vscode/ml_algorithms/data/data_banknote_authentication.txt', sep=",", header=None)
    data.columns = ["col_1", "col_2", "col_3", "col_4", 'col_5']
    X = data.drop('col_5', axis=1)
    y = data['col_5']

    print(get_best_split(X=X, y=y))