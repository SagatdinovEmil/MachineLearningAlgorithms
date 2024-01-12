import numpy as np
import pandas as pd
from numpy.linalg import norm
import random
from typing import Union, NoReturn, Dict, Callable

class MyTreeClf:
    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __str__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def __call__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"
