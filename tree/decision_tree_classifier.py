import pandas as pd
import numpy as np
from typing import Tuple


def calculate_entropy(y: np.array = None) -> float:
    _, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def divide_on_feature(
    X: np.ndarray = None,
    feature_i: int = None,
    threshold: float = None
) -> Tuple[np.array, np.array]:
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] <= threshold
    else:
        split_func = lambda sample: sample[feature_i] > threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return X_1, X_2


class DecisionNode():
    def __init__(
        self,
        feature_i: int = None,
        threshold: float = None,
        true_branch = None,
        false_branch = None,
        value: float = None,
    ) -> None:
        self.feature_i = feature_i
        self.threshold = threshold
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.value = value


class DecisionTree(object):
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
    ) -> None:
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.min_impurity: float = 1e-7
        self._impurity_calculation = None
        self._leaf_value_calculation = None
        self.leafs_cnt = 0

    def fit(self, X: pd.DataFrame = None, y: pd.Series = None) -> None:
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            X = X.to_numpy()
            y = y.to_numpy().reshape(-1, 1)
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X: np.ndarray = None, y: np.ndarray = None, current_depth: int = 0) -> DecisionNode:
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                thresholds = [np.mean([unique_values[i], unique_values[i+1]]) for i in range(len(unique_values)-1)]

                for threshold in thresholds:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                
                    if len(Xy1) > 0 or len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self._impurity_calculation(y, y1, y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": Xy1[:, n_features:],
                                "rightX": Xy2[:, :n_features],
                                "righty": Xy2[:, n_features:]
                            }

        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)

            return DecisionNode(
                feature_i=best_criteria["feature_i"],
                threshold=best_criteria["threshold"],
                true_branch=true_branch,
                false_branch=false_branch
            )

        leaf_value = self._leaf_value_calculation(y)
        
        return DecisionNode(value=leaf_value)
    
    def print_tree(self, tree=None, indent: str = " "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class MyTreeClf(DecisionTree): 
    def _calculate_information_gain(self, y: np.ndarray = None, y1: np.array = None, y2: np.array = None) -> float:
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

        return info_gain
      
    def _majority_vote(self, y: np.ndarray = None) -> int:
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count

        return most_common

    def fit(self, X: pd.DataFrame = None, y: pd.Series = None) -> None:
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(MyTreeClf, self).fit(X, y)