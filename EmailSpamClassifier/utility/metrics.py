import numpy as np
from typing import *
import math
import numpy as np


def gini(left_split: List[np.array], right_split: List[np.array]) -> float:
    all_size = left_split[1].size + right_split[1].size
    left_probs = (1-np.sum((np.bincount(left_split[1])/left_split[1].size)**2))*(left_split[1].size/all_size)
    right_probs = (1-np.sum((np.bincount(right_split[1])/right_split[1].size)**2))*(right_split[1].size/all_size)
    return left_probs+right_probs


def entropy(left_split: List[np.array], right_split: List[np.array]) -> float:
    yall = np.hstack([left_split[1], right_split[1]])
    probs = np.bincount(yall)/yall.size
    return (probs * np.log(1/probs)).sum()


def gaussian_dist(x: np.array, mu: np.array, std: np.array) -> np.array:
    return (1 / (np.sqrt(2 * math.pi) * std)) * np.exp(-((x - mu) ** 2 / (2 * (std ** 2))))
