import numpy as np
import pandas as pd
from numpy.linalg import norm
import random
from typing import Union, NoReturn, Dict, Callable

class MyKNNReg:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k: int = k
        self.train_size: tuple
        self.X: pd.DataFrame
        self.y: pd.Series
        self.metric: str = metric
        self.weight: str = weight

    def __str__(self):
        return f"MyKNNReg class: k={self.k}"

    def __call__(self):
        return f"MyKNNReg class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def predict(self, X: pd.DataFrame) -> np.array:
        result = np.zeros(len(X))
        for i in range(len(X)):
            distances = np.zeros(self.train_size[0])
            for j in range(len(self.X)):
                distances[j] = self.calc_metric(X.iloc[i], self.X.iloc[j])
            indices_of_nearest = np.argpartition(distances, self.k)[:self.k]
            indices_of_nearest = indices_of_nearest[np.argsort(distances[indices_of_nearest])]
            distances = distances[indices_of_nearest]
            nearest_y = self.y[indices_of_nearest]
            if self.weight == "uniform":
                result[i] = np.mean(nearest_y)
            else:
                weights = np.zeros(self.k)
                if self.weight == "rank":
                    rank_weights = 1 / np.arange(1, self.k + 1)
                    weights = rank_weights / np.sum(rank_weights)
                elif self.weight == "distance":
                    inv_distances = 1 / distances
                    sum_inv_distances = np.sum(inv_distances)
                    weights = inv_distances / sum_inv_distances
                result[i] = np.dot(weights, nearest_y)
        return result


    def calc_metric(self, x1: pd.Series, x2: pd.Series) -> float:
        def euclidean(x1: pd.Series, x2: pd.Series):
            result = 0
            for i in range(len(x1)):
                result += (x1.iloc[i] - x2.iloc[i]) ** 2
            return result ** 0.5

        def manhattan(x1: pd.Series, x2: pd.Series):
            result = 0
            for i in range(len(x1)):
                result += abs(x1.iloc[i] - x2.iloc[i])
            return result

        def chebyshev(x1: pd.Series, x2: pd.Series):
            result = float("-inf")
            for i in range(len(x1)):
                result = max(result, abs(x1.iloc[i] - x2.iloc[i]))
            return result

        def cosine(x1: pd.Series, x2: pd.Series):
            dot = x1.dot(x2)
            norm1 = norm(x1)
            norm2 = norm(x2)
            similarity = dot / (norm1 * norm2)
            return 1 - similarity
        if self.metric == "euclidean":
            return euclidean(x1, x2)
        if self.metric == "manhattan":
            return manhattan(x1, x2)
        if self.metric == "chebyshev":
            return chebyshev(x1, x2)
        if self.metric == "cosine":
            return cosine(x1, x2)
