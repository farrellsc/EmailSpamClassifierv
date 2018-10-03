## Solution for COMS 4771 HW1 Problem 6

#### Problem: Email spam classification case study
This datafile contains email data of around 5,000 emails divided in two folders ‘ham’ and ‘spam’ (there are about 3,500 emails in the ‘ham’ folder, and 1,500 emails in the ‘spam’ folder). Each email is a separate text file in these folders. These emails have been slightly preprocessed to remove meta-data information.

(i). (Embedding text data in Euclidean space) The first challenge you face is how to systematically embed text data in a Euclidean space. It turns out that one successful way of transforming text data into vectors is via “Bag-of-words” model. Basically, given a dictionary of all possible words in some order, each text document can be represented as a word count vector of how often each word from the dictionary occurs in that document.

Example: suppose our dictionary D with vocabulary size 10 (|D| = 10). The words
(ordered in say alphabetical order) are:
```
1: also
2: football
3: games
4: john
5: likes
6: Mary
7: movies
8: to
9: too
10: watch
```
Then any text document created using this vocabulary can be embedded in R | D| by counting how often each word appears in the text document.

Say, an example text document t is:
`John likes to watch football. Mary likes movies.`
Then the corresponding word count vector in |D| = 10 dimensions is:
`[ 0 1 0 1 2 1 1 1 0 1]`
(because the word “also” occurs 0 times, ”football” occurs 1 time, etc. in the document.)
While such an embedding is extremely useful, a severe drawback of such an embedding is that it treats similar meaning words (e.g. watch, watches, watched, watching, etc.) independently as separate coordinates. To overcome this issue one should preprocess the entire corpus to remove the common trailing forms (such as “ing”, “ed”, “es”, etc.) and get only the root word. This is called word-stemming. Your first task is to embed the given email data in a Euclidean space by: first performing word stemming, and then applying the bag-of-words model.

(ii). Once you have a nice Euclidean representation of the email data. Your next task is to develop a spam classifier to classify new emails as spam or not-spam. You should compare performance of naive-bayes, nearest neighbor (with L 1 , L 2 and L ∞ metric) and decision tree classifiers.
(you may use builtin functions for performaing basic linear algebra and probability calculations but you should write the classifiers from scratch.)
You must submit your code to Courseworks to receive full credit.

(iii). Which classifier (discussed in part (ii)) is better for the email spam classification dataset? You must justify your answer with appropriate performance graphs demonstrating the superiority of one classifier over the other. Example things to consider: you should evaluate how the classifier behaves on a holdout ‘test’ sample for various splits of the data; how does the training sample size affects the classification performance.