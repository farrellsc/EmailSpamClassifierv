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
        self.database = "//MachineLearning4771/HW/HW1/hw1data/"
        self.testbase = "/root/EmailSpamClassifierv/scripts/test/data/"
        self.dataLoader = pickle.load(
            open("/root/EmailSpamClassifierv/data/DataLoader2000", 'rb'))
        [self.x_train, self.y_train], [self.x_test, self.y_test] = self.dataLoader.get_data()
        print("x_train:", self.x_train.shape)
        print("y_train:", self.y_train.shape)
        print("x_test:", self.x_test.shape)
        print("y_test:", self.y_test.shape)
        # logger.setLevel(logging.DEBUG)
        # logger.info("testing classifier")
       
    def test_decisionTree(self):
        print("testing decision tree classifier")
        dt = DecisionTree(15, 20)
        dt.train(self.x_train, self.y_train)
        pred = dt.predict(self.x_test)
        accu = Classifier.evaluate(pred.T, self.y_test)
        print("data size: %d, decision tree result: %f" % (self.y_train.size + self.y_test.size, accu))
    
        skdt = DecisionTreeClassifier()
        skdt.fit(self.x_train, self.y_train)
        skpred = skdt.predict(self.x_test)
        skaccu = Classifier.evaluate(skpred.T, self.y_test)
        print("sklearn decision tree result: %f" % skaccu)
    
