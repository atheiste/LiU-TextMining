# -*- coding: utf-8 -*-
"""
Created on Fri May  3 02:29:28 2013

@author: Seby
"""
from __future__ import division
import random,nltk
from nltk.corpus import movie_reviews

# documents = [(words, "pos|neg"), (words, "pos|neg"), ...]
documents = [(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()[:4]
            for fileid in movie_reviews.fileids(category)[:4]]

# Use only binary features [has(’word’)] based on the 1000 most 
# frequent words in the corpus.
random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:1000]

def document_features(document):
    document_word = set([w.lower() for w in document])
    features = {}
    for word in word_features:
        features["has('{0}')".format(word)] = (word.lower() in document_word)
    return features

# Construct a naive Bayes classifier from a training sample consisting of 80%
# of the data
featuresets = [(document_features(d),c) for (d,c) in documents]
threshold = int(len(featuresets)*0.8)
train_set, test_set = [featuresets[:threshold],featuresets[threshold:]]
classifier = nltk.NaiveBayesClassifier.train(train_set)

#==============================================================================
#  Evaluate a classifier in terms of Accuracy, Precision, Recall, F-Measure
#==============================================================================

def evaluate(classifier, test_set):

    def f_measure(precision,recall,alpha):
        try:
            return 1/(alpha*(1/precision)+(1-alpha)*1/recall)
        except ZeroDivisionError:
            return 0

    test = classifier.batch_classify([fs for (fs,l) in test_set])
    gold = [l for (fs,l) in test_set]
    matrix = nltk.ConfusionMatrix(gold, test)
    tp = (matrix['pos','pos'])
    fn = (matrix['pos','neg'])
    fp = (matrix['neg','pos'])

    accuracy = nltk.classify.accuracy(classifier, test_set)

    precision = tp/(tp+fp or 1)
    recall = tp/(tp+fn or 1)
    f = f_measure(precision,recall,float(raw_input("F-Measure Alpha: ")))

    return (accuracy,precision,recall,f)

#==============================================================================
#  Show evaluation Results
#==============================================================================

print ("Accuracy: {:.2f} - Precision: {:.2f} - Recall: {:.2f} - F-measure: {:.2f}".format(*evaluate(classifier, test_set)))


