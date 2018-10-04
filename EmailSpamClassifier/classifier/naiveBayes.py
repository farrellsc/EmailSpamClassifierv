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
        self.bayesianTable = np.zeros([self.cls_num, x.shape[1], x.max()+1])
        for i in range(self.cls_num):
            for j in range(x.shape[1]):
                curClsFeatures = x[y==i, j]
                self.bayesianTable[i, j, curClsFeatures] += 1

    def predict(self, x: np.array) -> np.array:
        pred = np.empty([x.shape[0], 1])
        for i in range(x.shape[0]):
            probs = np.zeros([self.cls_num])
            for j in range(self.cls_num):
                curValues = self.bayesianTable[j, :, :] + 1     # feature * maxValue
                curValues = np.apply_along_axis(lambda x: x/x.sum(), 1, curValues)
                # print(curValues.shape, curValues.max(), curValues.min(), curValues[0,:].sum())
                curProbs = curValues[np.arange(curValues.shape[0]), x[i, :]]
                curProbs = curProbs-curProbs.mean()+1
                probs[j] = np.prod(curProbs)
            pred[i] = probs.argmax()
        print(pred.T)
        return pred
