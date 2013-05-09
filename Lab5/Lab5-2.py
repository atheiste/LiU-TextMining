# coding: utf-8
"""
The application was optimized to run using PyPy 2.0 so we strongly recommend
to run on PyPy. The time difference is between 10 - 12 times (that counts).

@author: Seby, AtheIste
"""
from __future__ import division, print_function
import random
import nltk
import re
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from collections import Counter
from nltk import bigrams
from nltk import trigrams


#==============================================================================
# bigram_measures = nltk.collocations.BigramAssocMeasures()
# trigram_measures = nltk.collocations.TrigramAssocMeasures()
# 
# bifinder = BigramCollocationFinder.from_words(movie_reviews.words())
# trifinder = TrigramCollocationFinder.from_words(movie_reviews.words())
# 
# bifinder.nbest(bigram_measures.pmi, 10)
# trifinder.nbest(bigram_measures.pmi, 10)
#==============================================================================



documents = []
for category in movie_reviews.categories():
    file_ids = movie_reviews.fileids(category)
    for fileid in file_ids:
        documents.append( [movie_reviews.words(fileid), category] )

all_words = nltk.FreqDist(w for w in movie_reviews.words())


wnl = nltk.WordNetLemmatizer()
p_stemmer = nltk.PorterStemmer()
l_stemmer = nltk.LancasterStemmer()


#==============================================================================
#  Create bigrams - trigrams - collocations
#==============================================================================


    
def create_collocations(features_words):
    collocations=[]
    return collocations   


#==============================================================================
#  Create Sets Functions
#  Document Features improved
#==============================================================================

def count_collates_features(document, coll):
    features = {}
    return features

def has_collates_features(document, coll):
    features = {}
    return features

def count_trigrams_features(document, trigr):
    features = {}
    for t in trigr:
        summ = 0
        for i in range(2,len(document)):
            if((document[i-2],document[i-1],document[i]) == t):
                summ+=1
        features['count(%s,%s,%s)' % (document[i-2],document[i-1],document[i])] = summ
    return features

def count_bigrams_features(document, bigr):
    features = {}
    for b in bigr:
        summ = 0
        for i in range(1,len(document)):
            if((document[i-1],document[i]) == b):
                summ+=1
        features['count(%s,%s)' % (document[i-1],document[i])] = summ
    return features

def has_trigrams_features(document, trigr):
    features = {}
    for t in trigr:
        found = False
        for i in range(1,len(document)):
            if ((document[i-2],document[i-1],document[i]) == t):
                found = True
                break
        features['has(%s)' % str(t)] = found      
    return features
    
def has_bigrams_features(document, bigr):
    features = {}
    for b in bigr:
        found = False
        for i in range(1,len(document)):
            if ((document[i-1],document[i]) == b):
                found = True
                break
        features['has(%s)' % str(b)] = found      
    return features


def has_features(document, features_words):
    features = {}
    for word in document:
        features['has(%s)' % word] = (word in features_words)
    return features

def count_features(document, features_words):
    features = {}
    counter = Counter(document)
    for word in features_words:
        features['count(%s)' % word] = counter[word]
    return features


def create_sets(documents, features_words):
    featuresets = []
    bigr = bigrams(features_words)
    trigr = trigrams(features_words)
    coll = create_collocations(features_words)    
    
    l = len(documents)
    for i in range(l):
        print("Computing the feature set for document {0} of {1}".format(i,l))
        features = {}
        #features.update(has_features(document[0], features_words))
        #features.update(has_bigrams_features(document[0], bigr))
        features.update(has_trigrams_features(documents[i][0], trigr))
        #features.update(has_collates_features(document[0], collates))
        #features.update(count_features(document[0], features_words))
        #features.update(count_bigrams_features(document[0], bigr))
        #features.update(count_trigrams_features(document[0], trigr))
        #features.update(count_collates_features(document[0], collates)
        
        featuresets.append((features, documents[i][1]))
        
    threshold = int(len(documents)*0.8)
    return [featuresets[:threshold],featuresets[threshold:]]

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
    #f = f_measure(precision,recall,float(raw_input("F-Measure Alpha: ")))
    f = f_measure(precision, recall, 0.5)

    return [accuracy,precision,recall,f]


#==============================================================================
#  Analysis definition
#==============================================================================
def analysis(documents, document_preprocess, features, features_preprocess):
    for f in document_preprocess:
        for i, document in enumerate(documents):
            documents[i][0] = map(f, document[0])
        features = map(f, features)

    for feat_func in features_preprocess:
        features = feat_func(features)
    features = set(features[:1000])
#    features = features[:1000]

    train_set, test_set = create_sets(documents, features)
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    x = evaluate(classifier, test_set)
    classifier.show_most_informative_features(n=20)
    return x

def punctuation_remove(features):
    processed = []
    for word in features:
        processed.append(
        ''.join([c for c in word.lower() if re.match("[a-z\-\' \n\t]", c)]))
    return processed

STOP_WORDS = stopwords.words('english')
def stopwords_remove(features):
    processed = []
    for word in features:
        if word.lower() in STOP_WORDS:
            continue
        processed.append(word)
    return processed

results = []
# (document/features preprocessing functions, features cleaning functions)
analysis_functions = [
    ((str.lower, ),      (),),
    ((wnl.lemmatize, ),  (),),
    ((p_stemmer.stem, ), (),),
    ((),                 (punctuation_remove, ),),
    ((),                 (punctuation_remove, stopwords_remove),),
    ((wnl.lemmatize, ),  (stopwords_remove, ),),
    ((p_stemmer.stem, ), (stopwords_remove, ),),
    ((),                 (stopwords_remove, ),),
]

SAMPLES = 1
for i in range(SAMPLES):
     random.shuffle(documents)
     results.append([])
     for doc_fs, feat_fs in analysis_functions:
         results[i].append(
             analysis(documents, doc_fs, all_words.keys()[:1000], feat_fs))

for col in range(len(analysis_functions)):
    sums = [0, 0, 0, 0]
    for row in range(SAMPLES):
        for i in range(len(sums)):
            sums[i] += results[row][col][i]

    for i in range(len(sums)):
        sums[i] /= SAMPLES

    print ("Accuracy: {:.2f} - Precision: {:.2f} - Recall: {:.2f} - "
           "F-measure: {:.2f}".format(*sums))



