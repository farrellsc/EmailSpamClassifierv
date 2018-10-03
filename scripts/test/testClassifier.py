from EmailSpamClassifier.dataLoader.dataLoader import DataLoader
from EmailSpamClassifier.classifier.classifier import Classifier
from EmailSpamClassifier.classifier.naiveBayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB
from EmailSpamClassifier.classifier.nearestNeighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from EmailSpamClassifier.classifier.decisionTree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from unittest import TestCase
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
# import logging

# logger = logging.getLogger()


class TestClassifier(TestCase):
    def setUp(self):
        self.database = "/media/zzhuang/00091EA2000FB1D0/CU/SEM1/MachineLearning4771/HW/HW1/hw1data/"
        self.testbase = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/EmailSpamClassifier/scripts/test/data/"
        self.dataLoader = pickle.load(
            open("/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/EmailSpamClassifier/data/allDataLoader", 'rb'))
        [self.x_train, self.y_train], [self.x_test, self.y_test] = self.dataLoader.get_data()
        print("x_train:", self.x_train.shape)
        print("y_train:", self.y_train.shape)
        print("x_test:", self.x_test.shape)
        print("y_test:", self.y_test.shape)
        # logger.setLevel(logging.DEBUG)
        # logger.info("testing classifier")

    # def test_naiveBayes(self):
    #     print("testing naive bayes classifier")
    #     nb = NaiveBayes()
    #     nb.train(self.x_train, self.y_train)
    #     pred = nb.predict(self.x_train)
    #     accu = Classifier.evaluate(pred.T, self.y_train)
    #     print("naive bayes result: %f" % accu)
    #
    #     sknb = GaussianNB()
    #     sknb.fit(self.x_train, self.y_train)
    #     skpred = sknb.predict(self.x_train)
    #     skaccu = Classifier.evaluate(skpred, self.y_train)
    #     print("sklearn bayesian module result: %f" % skaccu)

    # def test_decisionTree(self):
    #     print("testing decision tree classifier")
    #     dt = DecisionTree(10, 5)
    #     dt.train(self.x_train, self.y_train)
    #     pred = dt.predict(self.x_test)
    #     accu = Classifier.evaluate(pred.T, self.y_test)
    #     print("decision tree result: %f" % accu)
    #
    #     skdt = DecisionTreeClassifier()
    #     skdt.fit(self.x_train, self.y_train)
    #     skpred = skdt.predict(self.x_test)
    #     skaccu = Classifier.evaluate(skpred.T, self.y_test)
    #     print("sklearn decision tree result: %f" % skaccu)
    #
    # def test_knn(self):
    #     print("testing k nearest neighbor classifier")
    #     k = 10
    #     knn = NearestNeighbors(k, 2)
    #     knn.train(self.x_train, self.y_train)
    #     pred = knn.predict(self.x_train)
    #     accu = Classifier.evaluate(pred.T, self.y_train)
    #     print("knn result: %f" % accu)
    #
    #     skknn = KNeighborsClassifier(n_neighbors=k)
    #     skknn.fit(self.x_train, self.y_train)
    #     skpred = skknn.predict(self.x_train)
    #     skaccu = Classifier.evaluate(skpred, self.y_train)
    #     print("sklearn knn module result: %f" % skaccu)
