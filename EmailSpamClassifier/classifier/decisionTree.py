from EmailSpamClassifier.classifier.classifier import Classifier
from EmailSpamClassifier.utility.DTNode import DTNode
from EmailSpamClassifier.utility.metrics import entropy, gini
from copy import deepcopy
import numpy as np
from typing import *


class DecisionTree(Classifier):
    def __init__(self, max_tree_depth, min_sample_split):
        super(DecisionTree).__init__()
        self.dataset: np.array = None
        self.labels: np.array = None
        self.root = None
        self.params = {
            "max_tree_depth": max_tree_depth,
            "min_sample_split": min_sample_split
        }

    def train(self, x: np.array, y: np.array):
        self.dataset = x
        self.labels = y
        feature_index, feature_value, groups = self.split_node(self.dataset, self.labels)
        # print("first split: left: %d, right: %d" % (groups[0][1].size, groups[1][1].size))
        self.root = DTNode(feature_index, feature_value, groups, None, 1, groups[0][1].size + groups[1][1].size)
        self.build_tree(self.root)

    def build_tree(self, root: DTNode) -> None:
        """
        recursively build decision tree, with tree root as self.root
        :param root: current node
        """
        print("\n\nnew node!!! depth: %d, current size: %d" % (root.depth, root.leftGroup[1].size + root.rightGroup[1].size))
        if root.depth >= self.params["max_tree_depth"] or \
                root.leftGroup[1].size + root.rightGroup[1].size <= self.params['min_sample_split'] or \
                root.leftGroup[1].size == 0 or root.rightGroup[1].size == 0:
            self.set_leaf(root)
            return
        
        # left
        if root.leftGroup[0].size != 0 and root.leftGroup[1].size != 0:
            left_feature_index, left_feature_value, left_groups = self.split_node(root.leftGroup[0], root.leftGroup[1])
            root.leftChild = DTNode(left_feature_index, left_feature_value, deepcopy(left_groups), root, root.depth+1,
                                    left_groups[0][1].size + left_groups[1][1].size)
            self.build_tree(root.leftChild)

        # right
        if root.rightGroup[0].size != 0 and root.rightGroup[1].size != 0:
            right_feature_index, right_feature_value, right_groups = self.split_node(root.rightGroup[0], root.rightGroup[1])
            root.rightChild = DTNode(right_feature_index, right_feature_value, deepcopy(right_groups), root, root.depth+1,
                                     right_groups[0][1].size + right_groups[1][1].size)
            self.build_tree(root.rightChild)

        del root.leftGroup, root.rightGroup

    def split_node(self, x: np.array, y: np.array):
        """
        for current dataset and labels, find the best splitting point (feature index and sample value)
        :param x: dataset
        :param y: labels
        :return: [feature index, feature value, [[leftx, lefty], [rightx, righty]]]
        """
        # print('feature num: %d, row num: %d' % (x.shape[1], x.shape[0]))
        selected = {
            "metric": y.size,
            "feature_index": -1,
            "feature_value": y.size,
            "left_split": None,
            "right_split": None
        }

        for feature_index in range(x.shape[1]):
            unique_values = np.unique(x[:, feature_index])
            for feature_value in unique_values:
                left_indices = [ii for ii in range(x.shape[0]) if x[ii, feature_index] <= feature_value]
                right_indices = [ii for ii in range(x.shape[0]) if x[ii, feature_index] > feature_value]
                left_split = [x[left_indices, :], y[left_indices]]
                right_split = [x[right_indices, :], y[right_indices]]
                metric = gini(left_split, right_split)
                if metric < selected['metric']:
                    selected['metric'] = metric
                    selected['feature_index'] = feature_index
                    selected['feature_value'] = feature_value
                    selected['left_split'] = left_split
                    selected['right_split'] = right_split

                # if feature_index % 100 == 0:
                #     yall = np.hstack([left_split[1], right_split[1]])
                #     probs = np.bincount(yall)
                #     print(feature_index, feature_value, metric, left_split[1].size, right_split[1].size, probs)
            if feature_index % 100 == 0:
                print("choices: %d, index: %d, feature: %f, metric: %f, left size: %d, right size: %d"
                      % (unique_values.size, feature_index, selected['feature_value'], selected['metric'], selected['left_split'][1].shape[0], selected['right_split'][1].shape[0]))
            if feature_index == 500:
                break

        res = [selected['feature_index'], selected['feature_value'], [selected['left_split'], selected['right_split']]]
        return res

    def set_leaf(self, root: DTNode) -> None:
        """
        use root.leftGroup: [left x, left y], root.rightGroup: [right x, right y] to set root.label
        """
        left = np.bincount(root.leftGroup[1])
        right = np.bincount(root.rightGroup[1])
        tmp = np.zeros([max(left.size, right.size)])
        tmp[: left.size] += left
        tmp[: right.size] += right
        root.label = tmp.argmax()
        # print("--------------")
        # print(root.label)
        # print("--------------")
        del root.leftGroup, root.rightGroup

    def predict(self, x: np.array) -> np.array:
        pred = np.empty([x.shape[0], 1])
        for i in range(x.shape[0]):
            sample = x[i, :]
            iter = self.root
            while iter.label == -1:
                if sample[iter.featureIndex] <= iter.featureValue:
                    iter = iter.leftChild
                else:
                    iter = iter.rightChild
            pred[i] = iter.label
        return pred
