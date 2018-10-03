from EmailSpamClassifier.classifier.classifier import Classifier
from EmailSpamClassifier.utility.metrics import gaussian_dist
import numpy as np
import math


class NaiveBayes(Classifier):
    def __init__(self):
        super(NaiveBayes).__init__()
        self.bayesianTable = None
        self.clsTable = None
        self.cls_num = -1

    def train(self, x: np.array, y: np.array) -> None:
        self.cls_num = np.bincount(y).size
        self.bayesianTable = np.zeros([self.cls_num, x.shape[1], 2])
        for i in range(self.cls_num):
            for j in range(x.shape[1]):
                curClsFeatures = x[y==i, j]
                self.bayesianTable[i, j] = curClsFeatures.mean(), curClsFeatures.std()

    def predict(self, x: np.array) -> np.array:
        pred = np.empty([x.shape[0], 1])
        for i in range(x.shape[0]):
            probs = np.zeros([self.cls_num])
            for j in range(self.cls_num):
                table = self.bayesianTable[j, :, :].T
                sample, mu, std = x[i, :], table[0, :], table[1, :]
                curValue = gaussian_dist(sample, mu, std)
                curValue = curValue[std != 0]
                probs[j] += curValue.mean()
                print(curValue)
                print("mean:", curValue.mean(), "size:", curValue.size, "prod: ", probs[j])
            pred[i] = probs.argmax()
        print(pred.T)
        return pred
