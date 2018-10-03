from typing import *
import numpy as np


class DTNode:
    def __init__(self, feature_index: int, feature_value: float, groups: List[List[np.array]],
                 parent, depth: int, size: int):
        """

        :param featureIndex: current splitting feature index
        :param featureValue: current splitting feature value
        :param groups: [[left x, left y], [right x, right y]]
        :param parent: parent DTNode (shouldn't contain `groups`)
        :param depth: current depth
        """
        self.featureIndex = feature_index
        self.featureValue = feature_value
        # you need to delete left/rightGroup after you have linked DTNode children to current node
        self.leftGroup: List[np.array] = groups[0]
        self.rightGroup: List[np.array] = groups[1]
        self.leftChild = None
        self.rightChild = None
        self.parent = parent
        self.depth = depth
        self.label = -1
        self.size = size