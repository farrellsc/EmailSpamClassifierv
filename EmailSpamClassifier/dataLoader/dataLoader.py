from sklearn.model_selection import train_test_split
from stemming.porter2 import stem
from typing import *
import pickle
import re
import numpy as np


class DataLoader:
    def __init__(self, text: List[str], labels: List[int]) -> None:
        """
        add text to dataloader, then construct iterator
        all samples in dataloader share a same label
        :param text: raw string
        """
        # raw_text is a string, containing all contents in an email
        # raw_text: "Subject: re : meter 984132 for 1 / 16 / 99 10 , 000 should be allocated to adonis # 51862 on the
        #            5 th only . nothing should be allocated or confirmed for adonis on the 16 th . additionally , it
        #            looks like 5000 was confirmed as a receipt from mitchell ( track id 3167 ) on the 16 th . this
        #            should be - 0 - also . by taking the conf / allocation for adonis"
        self.raw_text: List[str] = text

        self.labels = np.array(labels)

        # stripped_n_stemmed is a string, it's the content of the email after all symbols have been stripped from
        # raw_text, and the tailing forms have been removed for all words. All letters should be in lower case.
        # stripped_n_stemmed[0]: "Subject re meter 984132 for 1 16 99 10 000 should be allocate to adonis 51862 on
        #                 the 5 th only nothing should be allocate or confirm for adonis on the 16 th additional it
        #                 look like 5000 was confirm as a receipt from mitchell track id 3167 on the 16 th this should
        #                 be 0 also by taking the conf allocation for adonis"
        self.stripped_n_stemmed: List[List] = []

        # word_dict is the dictionary encoding of words, with keys as word string, and values as a unique word id.
        # word_dict: {
        #               "subject": "0",
        #               "as": "1",
        #               "content": "2",
        #               "allocation": "3",
        #               ...
        #            }
        self.word_dict = {}
        self.reverse_word_dict = {}

        # x_train, x_test are word dict encoding for emails
        # stripped_n_stemmed: ["only nothing should be allocate", ...]
        # x_train/test: [[0,1,1,1,0,0,1,...], ...]
        self.x_train = []
        self.x_test = []
        self.y_train: np.array = None
        self.y_test: np.array = None

    def split_train_test_n_bagging(self, test_ratio):
        x_train, x_test, self.y_train, self.y_test = train_test_split(self.stripped_n_stemmed, self.labels,
                                                                      test_size=test_ratio)
        self.build_word_dict(x_train)
        self.bagging(x_train, self.x_train)
        self.bagging(x_test, self.x_test)

    def save_process(self, output_path) -> None:
        pickle.dump(open(output_path, 'w'), {
            "stripped_n_stemmed": self.stripped_n_stemmed,
            "word_dict": self.word_dict,
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test
        })

    def strip_stem(self) -> None:
        """
        strip all symbols, perform word stemming and bag words with unique indices
        stripped & stemmed words are stored in self.strpped_n_stemmed
        meanwhile construct a word:id encoding, store the map in self.word_dict
        encode text content into word id for each input string. store the result in self.bagged_text
        :return:
        """
        for raw_email in self.raw_text:
            stripped_n_stemmed_email = []
            for word in raw_email.split():
                word = word.strip().lower()
                word = self.strip_symbol(word)
                if word == "":
                    continue
                word = self.stem_word(word)
                stripped_n_stemmed_email.append(word)
            self.stripped_n_stemmed.append(stripped_n_stemmed_email)

    def build_word_dict(self, stripped_n_stemmed_data):
        dict_size = 0
        for email in stripped_n_stemmed_data:
            for word in email:
                if self.word_dict.get(word, None) is None:
                    self.word_dict[word] = dict_size
                    self.reverse_word_dict[dict_size] = word
                    dict_size += 1

    def bagging(self, input, res) -> None:
        for line in input:
            curBag = [0 for _ in range(len(self.word_dict))]
            for word in line:
                if self.word_dict.get(word, None) is not None:
                    curBag[self.word_dict[word]] += 1
            res.append(curBag)

    def strip_symbol(self, word: str) -> str:
        word = re.sub(r'[^\w]', "", word)
        return word.strip()

    def stem_word(self, word: str) -> str:
        return stem(word)

    def get_data(self) -> List[List[np.array]]:
        """
        return x and y in np array formation
        """
        return [[np.array(self.x_train), np.array(self.y_train)], [np.array(self.x_test), np.array(self.y_test)]]
