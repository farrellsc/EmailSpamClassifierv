from EmailSpamClassifier.dataLoader.dataLoader import DataLoader
from EmailSpamClassifier.classifier.classifier import Classifier
from EmailSpamClassifier.classifier.naiveBayes import NaiveBayes
from EmailSpamClassifier.classifier.nearestNeighbors import NearestNeighbors
from EmailSpamClassifier.classifier.decisionTree import DecisionTree
from unittest import TestCase
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


class TestClassifier(TestCase):
    def setUp(self):
        self.database = "/media/zzhuang/00091EA2000FB1D0/CU/SEM1/MachineLearning4771/HW/HW1/hw1data/"
        self.testbase = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/EmailSpamClassifier/scripts/test/data/"
        self.dataLoader = pickle.load(
            open("/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/EmailSpamClassifier/data/sampleDataLoader", 'rb'))
        self.data, self.labels = self.dataLoader.get_data()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=0.2)

    def test_knn(self):
        print("testing k nearest neighbor classifier")
        knn = NearestNeighbors(5000, 1)
        knn.train(self.x_train, self.y_train)
        pred = knn.predict(self.x_train)
        accu = Classifier.evaluate(pred.T, self.y_train)
        print("knn result: %f" % accu)
