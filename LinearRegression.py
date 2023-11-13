import numpy as np
import pandas as pd
import random
from typing import Union, NoReturn, Dict, Callable


def add_fictive(X: np.array) -> np.array:
    return np.insert(X, 0, np.ones(len(X)), axis=1)


class MyLineReg:
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1, metric: Union[None, str] = None,
                 reg: str = None, l1_coef: float = 0, l2_coef: float = 0, sgd_sample: Union[int, float, None] = None,
                 random_state: int = 42) -> NoReturn:
        self.n_iter: int = n_iter
        self.learning_rate: Callable
        if isinstance(learning_rate, float):
            self.learning_rate = lambda iter: learning_rate
        else:
            self.learning_rate = learning_rate
        self.weights: np.array = []
        self.metric: Union[None, str] = metric
        self.best_score: float = 0
        self.reg: str = reg
        self.l1_coef: float = l1_coef
        self.l2_coef: float = l2_coef
        self.sgd_sample: int = sgd_sample
        self.random_state: int = random_state

    def __str__(self) -> str:
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def __call__(self) -> str:
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False) -> NoReturn:
        random.seed(self.random_state)
        X = add_fictive(X.values)
        y = y.values
        self.weights = np.ones(len(X[0]))
        for i in range(1, self.n_iter+1):
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
            elif isinstance(self.sgd_sample, float):
                quantity = round(X.shape[0]*self.sgd_sample)
                sample_rows_idx = random.sample(range(X.shape[0]), quantity)
            elif isinstance(self.sgd_sample, type(None)):
                sample_rows_idx = list(range(X.shape[0]))
            X_iter = X[sample_rows_idx]
            y_iter = y[sample_rows_idx]
            y_pred_iter = np.dot(X_iter, self.weights)
            loss = np.mean((y_pred_iter - y_iter) ** 2) + self.calc_reg()
            grads = (2 / len(X_iter)) * np.dot(np.transpose(X_iter), np.dot(X_iter, self.weights) - y_iter) \
                + self.calc_reg_gradient()
            self.weights -= self.learning_rate(i) * grads
            if verbose:
                if i % verbose == 0:
                    y_pred = np.dot(X, self.weights)
                    metric = self.calc_metric(y, y_pred)
                    self.log(i, loss, metric)
        y_pred = np.dot(X, self.weights)
        self.best_score = self.calc_metric(y, y_pred)

    def predict(self, X: pd.DataFrame) -> np.array:
        X = add_fictive(X.values)
        y: np.array = np.dot(X, self.weights)
        return y

    def log(self, i: int, loss: float, metric: float = 0) -> NoReturn:
        if self.metric is None:
            print(f"{i} | loss: {loss}")
        else:
            print(f"{i} | loss: {loss} | {self.metric}: {metric}")

    def get_coef(self) -> np.array:
        return self.weights[1:]

    def calc_metric(self, y: np.array, y_pred: np.array) -> float:
        def r2(y_: np.array, y_pred_: np.array):
            ss_total = np.sum((y_ - y_pred_) ** 2)
            ss_residual = np.sum((y_ - np.mean(y_)) ** 2)
            r2 = 1 - (ss_total / ss_residual)
            return r2

        metrics: Dict[Union[str, None]: Callable] = {"mae": lambda y_, y_pred_: np.mean(np.abs(y_ - y_pred_)),
                                                     "mse": lambda y_, y_pred_: np.mean((y_ - y_pred_) ** 2),
                                                     "rmse": lambda y_, y_pred_: np.mean((y_ - y_pred_) ** 2) ** 0.5,
                                                     "mape": lambda y_, y_pred_: np.mean(
                                                         np.abs((y_ - y_pred_) / y_)) * 100,
                                                     "r2": r2,
                                                     None: lambda y_, y_pred_: 0}
        return metrics[self.metric](y, y_pred)

    def calc_reg(self):
        regs: Dict[Union[str, None]: Callable] = {"l1": lambda w: self.l1_coef * np.mean(abs(w)),
                                                  "l2": lambda w: self.l2_coef * np.mean(w ** 2),
                                                  "elasticnet": lambda w: self.l1_coef * np.mean(abs(w)) +
                                                                          self.l1_coef * np.mean(abs(w)),
                                                  None: lambda x: 0}
        return regs[self.reg](self.weights)

    def calc_reg_gradient(self):
        regs: Dict[Union[str, None]: Callable] = {"l1": lambda w: self.l1_coef * np.sign(w),
                                                  "l2": lambda w: self.l2_coef * 2 * w,
                                                  "elasticnet": lambda w: self.l1_coef * np.sign(w) +
                                                                          self.l2_coef * 2 * w,
                                                  None: lambda x: 0}
        return regs[self.reg](self.weights)

    def get_best_score(self):
        return self.best_score
