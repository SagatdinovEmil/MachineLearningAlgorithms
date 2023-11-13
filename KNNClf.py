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
