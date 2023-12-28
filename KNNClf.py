import numpy as np
import pandas as pd
import random
from typing import Union, NoReturn, Dict, Callable

class MyKNNClf:
    def __init__(self, k: int = 3):
        self.k = k
        self.train_size: tuple
        self.X: pd.DataFrame
        self.y: pd.Series

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"

    def __call__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def predict(self, X):
        result = np.zeros(len(X))
        for j in range(len(X)):
            distances = np.zeros(self.train_size)
            for i in range(len(self.X)):
                distances[i] = self.calc_metric(X[j], self.X[i])
            indices_of_nearest = np.argpartition(distances, self.k)[:self.k]
            fashion = 1 if np.mean(self.y[indices_of_nearest]) >= 0.5 else 0
            result[j] = fashion
        return result


    def predict_proba(self, X):
        pass

    def calc_metric(self, x1, x2):
        sum = 0
        for i in range(len(x1)):
            sum += (x1 - x2) ** 2
        return sum ** 0.5
