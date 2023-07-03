import pandas as pd
import numpy as np
from typing import Tuple


def calculate_entropy(y: pd.Series = None) -> float:
    _, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


def calculate_overall_metric(left: pd.Series = None, right: pd.Series = None) -> float:
    n = len(left) + len(right)
    left_shape = len(left) / n
    right_shape = len(right) / n

    overall_metric = (
        left_shape * calculate_entropy(left) + right_shape * calculate_entropy(right)
    )
    
    return overall_metric


def get_best_split(
    X: pd.DataFrame = None,
    y: pd.Series = None
) -> Tuple[str, float, float]:
    best_thresholds = {}
    for col in X.columns:
        initial_entropy = calculate_entropy(y=y)
        unique_values = sorted(X[col].unique())
        thresholds = [np.mean([unique_values[i], unique_values[i+1]]) for i in range(len(unique_values)-1)]
        for threshold in thresholds:
            left = X[X[col] <= threshold][[col]].index.values
            right = X[X[col] > threshold][[col]].index.values

            left_series = y.iloc[left]
            right_series = y.iloc[right]
            
            current_overall_metric = initial_entropy - calculate_overall_metric(left=left_series, right=right_series)

            if col not in best_thresholds:
                best_thresholds[col] = {
                    "split_value": threshold,
                    "metric": current_overall_metric
                }
            elif current_overall_metric > best_thresholds[col]["metric"]:
                best_thresholds[col]["split_value"] = threshold
                best_thresholds[col]["metric"] = current_overall_metric
            else:
                continue

    best = sorted(best_thresholds.items(), key=lambda x: x[1]["metric"], reverse=True)[0]

    return best[0], best[1]["split_value"], best[1]["metric"]


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
        self.leafs_cnt = None

    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.__dict__.items()]

        return f"{self.__class__.__name__} class: " + ", ".join(params)
    
    def fit(self, X: pd.DataFrame = None, y: pd.Series = None, verbose: int = 0) -> None:
        pass