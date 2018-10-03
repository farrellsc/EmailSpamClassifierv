from EmailSpamClassifier.classifier.classifier import Classifier
import numpy as np


class NearestNeighbors(Classifier):
    def __init__(self, k, norm):
        super(NearestNeighbors).__init__()
        self.coords: np.array = None
        self.labels: np.array = None
        self.params = {
            "k": k,
            "norm": norm
        }

    def train(self, x: np.array, y: np.array) -> None:
        """
        fill data structure with training data x and label y
        :param x: already existing node coords
        :param y: labels for x
        :return:
        """
        self.coords = x
        self.labels = y

    def predict(self, x: np.array) -> np.array:
        """
        :param x: test data
        :return: predicted labels
        """
        pred = np.empty([x.shape[0], 1])
        for i in range(x.shape[0]):
            curCoord = x[i, :].flatten()
            tmpx = np.copy(self.coords)
            dist = np.abs(np.sum((tmpx - curCoord)**self.params["norm"], axis=1)**(1/self.params["norm"]))
            kNeighbors = np.argsort(dist)[:self.params['k']]
            # voting
            pred[i] = np.bincount(self.labels[kNeighbors]).argmax()
        return pred
