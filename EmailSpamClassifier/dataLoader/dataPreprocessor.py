from typing import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


class DataPreprocessor:
    """
    input all data and labels, split them into train/validation/test datasets(lists)
    """
    def __init__(self, data: List[str], labels: List[int]) -> None:
        self.data = data
        self.labels = labels

    def split_datasets(self, ratio: List[int], seed: int) -> List[List[str]]:
        """
        return train, test datasets according to ratio
        :return:
        """
        if ratio[0] + ratio[1] != 1:
            raise BaseException("ratio should sum up to 1")
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels,
                                                            test_size=ratio[1], random_state=seed)
        return [x_train, y_train, x_test, y_test]
