# coding: utf-8
"""
The application was optimized to run using PyPy 2.0 so we strongly recommend
to run on PyPy. The time difference is between 10 - 12 times (that counts).

@author: Seby, AtheIste
"""
from __future__ import division, print_function

import nltk
import random
import re

from itertools import dropwhile
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from collections import Counter, defaultdict


#==============================================================================
#  Helper functions
#==============================================================================

def build_inverted_index(documents, keywords):
    '''
    Build inverted index for each keyword in a form
    {'keyword': {<document index>: TF, ..., 'df': DF, 'idf': IDF}, ...}
    where TF is a frequency of keyword in a document, DF is number
    of documents which contains the keyword and IDF is a measure of
    informativeness of the keyword
    '''
    index = defaultdict(dict)
    counters = [Counter(d) for d,c in documents]
    N = len(documents)
    for keyword in keywords:
        df = 0
        for i, counter in enumerate(counters):
            if counter[keyword]:
                index[keyword][i] = counter[keyword]
                df += 1
        index[keyword]['df'] = df
        index[keyword]['idf'] = math.log(N / df, 10)
    return index

def get_dfs(documents, keywords):
    '''
    Compute IDFS for every keyword
    :rvalue: list( tuple(keyword, idf), tuple(keyword, idf) )
    '''
    idfs = []
    doc_sets = [set(d) for d,c in documents]
    N = len(documents)
    for keyword in keywords:
        df = 0
        for d in range(N):
            df += int(keyword in doc_sets[d])
        idfs.append( (keyword, df) )
    return idfs


#==============================================================================
#  Feature extractors
#  Document Features improved
#==============================================================================

def has_features(document, features_words):
    '''Produces binary features "has('word'): <true|false>" '''
    features = {}
    for word in features_words:
        features['has({})'.format(word)] = (word in document)
    return features

def count_features(document, features_words):
    '''Produces count features "count('word'): <number>" '''
    features = {}
    counter = Counter(document)
    for word in features_words:
        features['count({})'.format(word)] = counter[word]
    return features

## Function which builds up training/testing set
def create_sets(documents, features_words):
    featuresets = []
    print( "length of features ", len(features_words) )
    print( *random.sample(features_words, 15) )
    for document in documents:
        features = {}
        #~ features.update(count_features(document[0], features_words))
        features.update(has_features(document[0], features_words))

        featuresets.append((features, document[1]))

    threshold = int(len(documents)*0.8)
    return [featuresets[:threshold], featuresets[threshold:]]

#==============================================================================
#  Evaluate a classifier in terms of Accuracy, Precision, Recall, F-Measure
#==============================================================================

def evaluate(classifier, test_set):

    def f_measure(precision,recall,alpha):
        try:
            return 1/(alpha*(1/precision)+(1-alpha)*1/recall)
        except ZeroDivisionError:
            return 1

    test = classifier.batch_classify([fs for (fs,l) in test_set])
    gold = [l for (fs,l) in  test_set]
    matrix = nltk.ConfusionMatrix(gold, test)
    tp = (matrix['pos','pos'])
    fn = (matrix['pos','neg'])
    fp = (matrix['neg','pos'])

    accuracy = nltk.classify.accuracy(classifier, test_set)

    precision = tp/(tp+fp or 1)
    recall = tp/(tp+fn or 1)
    f = f_measure(precision, recall, 0.5)

    return [accuracy,precision,recall,f]


#==============================================================================
#  Analysis function
#==============================================================================
def analysis(documents, document_preprocess, features, features_preprocess):
    for f in document_preprocess:
        for i in range(len(documents)):
            documents[i][0] = map(f, documents[i][0])
        features = map(f, features)

    for feat_func in features_preprocess:
        features = feat_func(documents, features)
    features = set(features)

    train_set, test_set = create_sets(documents, features)
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    x = evaluate(classifier, test_set)
    classifier.show_most_informative_features(n=20)
    return x

#==============================================================================
#  Features generators
#==============================================================================

def freq_features(documents, features):
    return nltk.FreqDist(features).keys()[:1000]


def df_features(documents, features):
    features = set(features)
    N = len(documents) // 2
    dfs = get_dfs(documents, features)
    # sort according to idf value
    dfs.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in dropwhile(lambda x: x[1] > int(N * 1), dfs)][:1000]

#==============================================================================
#  Helper functions for features cleaning
#==============================================================================

def punctuation_remove(documents, features):
    processed = []
    for word in features:
        processed.append(
        ''.join([c for c in word.lower() if re.match("[a-z\-\' \n\t]", c)]))
    return processed

STOP_WORDS = stopwords.words('english')
def stopwords_remove(documents, features):
    processed = []
    for word in features:
        if word.lower() in STOP_WORDS:
            continue
        processed.append(word)
    return processed

#==============================================================================
#  Program main block
#==============================================================================

documents = []
for category in movie_reviews.categories():
    file_ids = movie_reviews.fileids(category)
    for fileid in file_ids:
        documents.append( [movie_reviews.words(fileid), category] )

# set of all words across all documents
feature_candidates = movie_reviews.words()
print("Unique words: ", len(feature_candidates))

wnl = nltk.WordNetLemmatizer()
p_stemmer = nltk.PorterStemmer()
l_stemmer = nltk.LancasterStemmer()

results = []
# (document/features preprocessing functions, features cleaning functions)
analysis_functions = [
    ((str.lower, ),      (df_features, )),
    ((),                 (df_features, )),
    ((wnl.lemmatize, ),  (df_features, )),
    ((p_stemmer.stem, ), (df_features, )),
    ((),                 (punctuation_remove, df_features)),
    ((),                 (punctuation_remove, df_features)),
    ((wnl.lemmatize, ),  (stopwords_remove,   df_features)),
    ((p_stemmer.stem, ), (stopwords_remove,   df_features)),
    ((),                 (stopwords_remove,   df_features)),
]

SAMPLES = 3
for i in range(SAMPLES):
    random.shuffle(documents)
    results.append([])
    for doc_clean, feat_clean in analysis_functions:
        result = analysis(documents, doc_clean, feature_candidates, feat_clean)
        print ("Accuracy: {:.2f} - Precision: {:.2f} - Recall: {:.2f} - "
                "F-measure: {:.2f}".format(*result))
        results[i].append(result)
    print("")
print("")

for col in range(len(analysis_functions)):
    sums = [0, 0, 0, 0]
    for row in range(SAMPLES):
        for i in range(len(sums)):
            sums[i] += results[row][col][i]

    for i in range(len(sums)):
        sums[i] /= SAMPLES

    print ("Accuracy: {:.2f} - Precision: {:.2f} - Recall: {:.2f} - "
           "F-measure: {:.2f}".format(*sums))



