import numpy as np


class Classifier:
    def __init__(self):
        raise NotImplementedError

    def train(self, x: np.array, y: np.array) -> None:
        raise NotImplementedError

    def predict(self, x: np.array) -> np.array:
        raise NotImplementedError

    @staticmethod
    def evaluate(pred: np.array, y: np.array) -> float:
        """
        print evaluation result
        :param pred: predicted labels
        :param y: true labels
        """
        return (pred == y).sum()/pred.size
