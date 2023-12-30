import numpy as np
import pandas as pd
from numpy.linalg import norm
import random
from typing import Union, NoReturn, Dict, Callable

class MyKNNReg:
    def __init__(self, k: int = 3):
        self.k: int = k
        self.train_size: tuple
        self.X: pd.DataFrame
        self.y: pd.Series

    def __str__(self):
        return f"MyKNNReg class: k={self.k}"

    def __call__(self):
        return f"MyKNNReg class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape