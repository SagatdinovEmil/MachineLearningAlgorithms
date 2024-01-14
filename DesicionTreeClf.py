import numpy as np
import pandas as pd
from numpy.linalg import norm
import random
from typing import Union, NoReturn, Dict, Callable

class MyTreeClf:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __str__(self) -> str:
        return (f"MyTreeClf class: max_depth={self.max_depth}, "
                f"min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}")

def calc_entropy(y: pd.Series) -> float:
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calc_ig(y: pd.Series, y_left: pd.Series, y_right: pd.Series) -> float:
    parent_entropy = calc_entropy(y)
    n = len(y)
    weighted_entropy = (len(y_left) / n * calc_entropy(y_left)) + (len(y_right) / n * calc_entropy(y_right))
    ig = parent_entropy - weighted_entropy
    return ig

def get_best_split(X: pd.DataFrame, y: pd.Series) -> Union[str, float, float]:
    best_ig = 0
    col_name = ""
    split_value = 0

    for column in X.columns:
        X_col = X[column]
        values = np.sort(X_col.unique())
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

        for threshold in thresholds:
            y_left = y[X_col <= threshold]
            y_right = y[X_col > threshold]
            ig = calc_ig(y, y_left, y_right)

            if ig > best_ig:
                best_ig = ig
                col_name = column
                split_value = threshold

    return col_name, split_value, best_ig
