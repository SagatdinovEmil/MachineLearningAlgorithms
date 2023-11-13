import numpy as np
import pandas as pd
from typing import NoReturn, Union, Callable, Dict


def add_fictive(X: np.array) -> np.array:
    return np.insert(X, 0, np.ones(len(X)), axis=1)

class MyLogReg:
    def __init__(self, n_iter: int = 10, learning_rate: Union[float, Callable] = 0.1,  metric: Union[None, str] = None,
                reg: str = None, l1_coef: float = 0, l2_coef: float = 0, sgd_sample: Union[int, float, None] = None,
                 random_state: int = 42) -> NoReturn:
        self.n_iter: int = n_iter
        if isinstance(learning_rate, float):
            self.learning_rate = lambda iter: learning_rate
        else:
            self.learning_rate = learning_rate
        self.weights: np.array = np.array([])
        self.metric: Union[None, str] = metric
        self.best_score: float = 0
        self.reg: str = reg
        self.l1_coef: float = l1_coef
        self.l2_coef: float = l2_coef
        self.sgd_sample: Union[int, float, None] = sgd_sample
        self.random_state: int = random_state

    def __str__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def __call__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def __sigmoid(self, z: Union[np.array, float]) -> Union[np.array, float]:
        return 1 / (1 + np.exp(z))

    def __log_loss(self, y_pred: np.array, y: np.array):
        return -np.mean(y*np.log(y_pred + 1e-15) + (1-y)*np.log(1-y_pred + 1e-15)) + self.calc_reg()

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

    def get_coef(self) -> np.array:
        return self.weights[1:]

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[bool, int]) -> NoReturn :
        X = add_fictive(X.values)
        y = y.values
        self.weights = np.ones(len(X[0]))
        for i in range(1, self.n_iter+1):
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
            elif isinstance(self.sgd_sample, float):
                quantity = int(round(X.shape[0]*self.sgd_sample, 0))
                sample_rows_idx = random.sample(range(X.shape[0]), quantity)
            elif isinstance(self.sgd_sample, type(None)):
                sample_rows_idx = list(range(X.shape[0]))
            X_iter = X[sample_rows_idx]
            y_iter = y[sample_rows_idx]
            z = np.dot(self.weights, X_iter.T)
            y_pred = self.__sigmoid(-z)
            loss = self.__log_loss(y_pred, y_iter)
            grads = (y_pred-y_iter) @ X_iter / len(X_iter) + self.calc_reg_gradient()
            self.weights -= self.learning_rate(i) * grads
            if verbose:
                if i % verbose == 0:
                    z = np.dot(self.weights, X.T)
                    y_pred = self.__sigmoid(-z)
                    metric = self.calc_metric(X, y)
                    loss = np.mean(y*np.log(y_pred + 1e-15) + (1-y)*np.log(1-y_pred + 1e-15))
                    print(f"{i} | {loss} | {self.metric}: {metric}")
        self.best_score = self.calc_metric(X, y)

    def predict_proba(self, X: np.array) -> np.array:
        z = np.dot(self.weights, X.T)
        y_pred = self.__sigmoid(-z)
        return y_pred

    def predict(self, X: pd.DataFrame) -> np.array:
        return (self.predict_proba(X) > 0.5).astype(int)

    def calc_metric(self, X: pd.DataFrame, y_true: np.array) -> float:
        def _calc_TP(y: np.array, y_pred: np.array) -> float:
            return ((y_pred == 1) & (y == 1)).sum()

        def _calc_FP(y: np.array, y_pred: np.array) -> float:
            return ((y_pred == 1) & (y == 0)).sum()

        def _calc_TN(y: np.array, y_pred: np.array) -> float:
            return ((y_pred == 0) & (y == 0)).sum()

        def _calc_FN(y: np.array, y_pred: np.array) -> float:
            return ((y_pred == 0) & (y == 1)).sum()

        def _calc_accuracy(y: np.array, y_pred: np.array) -> float:
            return (y == y_pred).sum() / len(y)

        def _calc_precision(y: np.array, y_pred: np.array) -> float:
            TP = _calc_TP(y, y_pred)
            FP = _calc_FP(y, y_pred)
            return TP / (TP+FP)

        def _calc_recall(y: np.array, y_pred: np.array) -> float:
            TP = _calc_TP(y, y_pred)
            FN = _calc_FN(y, y_pred)
            return TP / (TP+FN)

        def _calc_F1(y: np.array, y_pred: np.array) -> float:
            precision = _calc_precision(y, y_pred)
            recall = _calc_recall(y, y_pred)
            return 2 * (precision * recall) / (precision + recall)

        def _calc_roc_auc(y: np.array, y_pred_proba: np.array) -> float:
            positive_samples: int = (y == 1).sum()
            negative_samples: int = (y == 0).sum()
            asc_score_indices = np.argsort(y_pred_proba)[::-1]
            y = y[asc_score_indices]
            scores = y_pred_proba[asc_score_indices]
            scores = np.round(scores, decimals=10)
            roc_auc_sum: float = 0
            for i in range(len(y)):
                if y[i] == 0:
                    for j in range(i):
                        if y[j] == 1:
                            if scores[j] > scores[i]:
                                # print(f"{scores[j]}>{scores[i]}")
                                roc_auc_sum += 1
                            elif scores[j] == scores[i]:
                                # print(f"{scores[j]}=={scores[i]}")
                                roc_auc_sum += 0.5
            return (1 / (positive_samples * negative_samples)) * roc_auc_sum

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        metrics: Dict[Union[str, None], Callable] = {
            "accuracy": lambda y_true, y_pred: _calc_accuracy(y_true, y_pred),
            "precision": lambda y_true, y_pred: _calc_precision(y_true, y_pred),
            "recall": lambda y_true, y_pred: _calc_recall(y_true, y_pred),
            "f1": lambda y_true, y_pred: _calc_F1(y_true, y_pred),
            "roc_auc": lambda y_true, y_pred_proba: _calc_roc_auc(y_true, y_pred_proba),
            None: lambda _, __: 0
        }
        
        if self.metric=="roc_auc":
            return metrics[self.metric](y_true, y_pred_proba)
        return metrics[self.metric](y_true, y_pred)

    def get_best_score(self):
        return self.best_score
