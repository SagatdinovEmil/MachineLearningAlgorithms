import numpy as np
import pandas as pd
from numpy.linalg import norm
import random
from typing import Union, NoReturn, Dict, Callable

class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = "euclidean"):
        self.k = k
        self.train_size: tuple
        self.X: pd.DataFrame
        self.y: pd.Series
        self.metric = metric

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"

    def __call__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def predict(self, X: pd.DataFrame) -> np.array:
        result = np.zeros(len(X), dtype=int)
        for j in range(len(X)):
            distances = np.zeros(self.train_size[0])
            for i in range(len(self.X)):
                distances[i] = self.calc_metric(X.iloc[j], self.X.iloc[i])
            indices_of_nearest = np.argpartition(distances, self.k)[:self.k]
            fashion = 1 if np.mean(self.y[indices_of_nearest]) >= 0.5 else 0
            result[j] = fashion
        return result


    def predict_proba(self, X: pd.DataFrame) -> np.array:
        result = np.zeros(len(X))
        for j in range(len(X)):
            distances = np.zeros(self.train_size[0])
            for i in range(len(self.X)):
                distances[i] = self.calc_metric(X.iloc[j], self.X.iloc[i])
            indices_of_nearest = np.argpartition(distances, self.k)[:self.k]
            result[j] = np.mean(self.y[indices_of_nearest])
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
