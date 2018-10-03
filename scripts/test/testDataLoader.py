from EmailSpamClassifier.dataLoader.dataLoader import DataLoader
from unittest import TestCase
import os
import codecs
import pickle


class TestDataProcessor(TestCase):
    def setUp(self):
        self.database = "/media/zzhuang/00091EA2000FB1D0/CU/SEM1/MachineLearning4771/HW/HW1/hw1data/"
        self.testbase = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/EmailSpamClassifier/scripts/test/data/"
        ham_sample_name = "2143.2000-09-05.farmer.ham.txt"
        ham_sample = "".join(
            codecs.open(self.database + "ham/" + ham_sample_name, encoding='utf8', errors='ignore').readlines())
        spam_sample_name = "0006.2003-12-18.GP.spam.txt"
        spam_sample = "".join(
            codecs.open(self.database + "spam/" + spam_sample_name, encoding='utf8', errors='ignore').readlines())
        self.oneHamLoader = DataLoader([ham_sample], [0])
        self.oneSpamLoader = DataLoader([spam_sample], [1])

    # def test_strip_stem_bag(self):
    #     self.oneHamLoader.strip_stem()
    #     with open(self.testbase + "test_strip_stem.txt", 'w') as out:
    #         nnn = " ".join(self.oneHamLoader.stripped_n_stemmed[0])
    #         out.write(nnn)
    #     with open(self.testbase + "test_bag_dict.txt", "w") as out:
    #         for key in self.oneHamLoader.word_dict.keys():
    #             out.write(key + " " + str(self.oneHamLoader.word_dict[key]) + "\n")
    #     self.oneHamLoader.bagging()
    #     with open(self.testbase + "test_bag.txt", 'w') as out:
    #         nnn = " ".join([str(one) for one in self.oneHamLoader.bagged_text[0]])
    #         out.write(nnn)
    #         out.write("\n" + str(sum(self.oneHamLoader.bagged_text[0])))

    def test_bulk_load(self):
        filenames = os.listdir(self.database + "all/")
        labels = [0 if filename.endswith("ham.txt") else 1 for filename in filenames ]
        self.dataLoader = DataLoader(
            ["".join(codecs.open(self.database + "all/" + name, encoding='utf8', errors='ignore').readlines())
             for name in filenames], labels)
        self.dataLoader.strip_stem()
        self.dataLoader.split_train_test_n_bagging(0.2)
        pickle.dump(self.dataLoader,
                    open("/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/EmailSpamClassifier/data/allDataLoader", 'wb'))
