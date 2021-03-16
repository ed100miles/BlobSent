import nltk
import random
import re
from statistics import mode
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import pickle

## Class to average classifier confidence
class ClassificationClass(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        classifications = []
        for classifier in self._classifiers:
            classification = classifier.classify(features)
            classifications.append(classification)
        return mode(classifications)

    def confidence(self, features):
        classifications = []
        for classifier in self._classifiers:
            classification = classifier.classify(features)
            classifications.append(classification)    
        return  classifications.count(mode(classifications)) / len(classifications)

positive_revs = open('/home/ed/Documents/gitCode/NLP/Data/positive.txt', 'r').read()
negative_revs = open('/home/ed/Documents/gitCode/NLP/Data/negative.txt', 'r').read()

documents = []

for review in positive_revs.split('\n'):
    documents.append((review, 'pos'))

for review in negative_revs.split('\n'):
    documents.append((review, 'neg'))

pickle_documents = open('documents.pickle', 'wb')
pickle.dump(documents, pickle_documents)
pickle_documents.close()

# Collect all words in list, sort by freq', add top 3k to word_features list
all_words = []

pos_rev_words = word_tokenize(positive_revs)
neg_rev_words = word_tokenize(negative_revs)

for w in pos_rev_words:
    all_words.append(w.lower())

for w in neg_rev_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

pickle_all_words = open('all_words.pickle', 'wb')
pickle.dump(all_words, pickle_all_words)
pickle_all_words.close()

word_features = list(all_words.keys())[:5000]

pickle_word_features = open('word_features.pickle', 'wb')
pickle.dump(word_features, pickle_word_features)
pickle_word_features.close()

# print(all_words.most_common(50))
# print(all_words['ostentatious'])

### TODO: Think about getting rid of punctuation and stop words later?? 

## Following marks value of dict above to True/False depending on if it 's in the review

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) # to return bool 

    return features

featuresets = [(find_features(review), category) for (review, category) in documents]

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print('NB Acc:', (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(30)

## Classifiers: 

MNB_Classifier = SklearnClassifier(MultinomialNB())
BernoulliNB = SklearnClassifier(BernoulliNB())
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter=300))
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SVC_classifier = SklearnClassifier(SVC())
LinearSVC_classifier = SklearnClassifier(LinearSVC())
NuSVC_classifier = SklearnClassifier(NuSVC())

classifiers = [MNB_Classifier,
                BernoulliNB,
                LogisticRegression_classifier,
                SGDClassifier_classifier,
                SVC_classifier,
                LinearSVC_classifier,
                NuSVC_classifier]

## Train classifiers:

for classifier in classifiers:
    # regex to get shortened, string not instance classifier name. 
    classifier_name = re.search('\((.*)\(', str(classifier)).group()
    classifier_name = classifier_name.replace("(", "")
    classifier.train(training_set)
    print(f'{classifier_name} OOSacc:',(nltk.classify.accuracy(classifier, testing_set))*100)

    pickle_classifier = open(f'{classifier_name}.pickle', 'wb')
    pickle.dump(classifier, pickle_classifier)
    pickle_classifier.close()

classification_class = ClassificationClass(MNB_Classifier,
                                            BernoulliNB,
                                            LogisticRegression_classifier,
                                            SGDClassifier_classifier,
                                            SVC_classifier,
                                            LinearSVC_classifier,
                                            NuSVC_classifier)

# Get ClassificationClass Acc, + test .classify + .confidence attributes. 

print('ClassificationClass OOS_acc: ', (nltk.classify.accuracy(classification_class, testing_set)))
print('Classification:', classification_class.classify(testing_set[0][0]))
print('Classification confidence:', classification_class.confidence(testing_set[0][0]))

# test_case = testing_set[0][0]
# test_case_label = testing_set[0][1]

# test_case_str = ''
# for x in test_case: 
#     test_case_str = test_case_str + x + ' '

# print(test_case_str)
# print(test_case_label)

