# coding: utf-8
"""
The application was optimized to run using PyPy 2.0 so we strongly recommend
to run on PyPy. The time difference is between 10 - 12 times (that counts).

@author: Seby, AtheIste
"""
from __future__ import division, print_function

import math
import nltk
import re
import random

from collections import Counter

from nltk import bigrams
from nltk import trigrams
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

wnl = nltk.WordNetLemmatizer()
p_stem = nltk.PorterStemmer()
l_stem = nltk.LancasterStemmer()
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
STOP_WORDS = set(stopwords.words('english'))


#==============================================================================
#                             Features Function 
#           (document , words|bigrams|trigrams|collocations|idfs)
#                           returning a dictionary
#==============================================================================

#==============================================================================
# def df_features(documents, features):
#     '''Removes unusable features based on DF value'''
#     features_set = set(features)
#     N = len(documents) // 2
#     X = 600
#     dfs = get_dfs(documents, features_set)
#     # sort according to idf value
#     dfs.sort(key=lambda x: x[1], reverse=True)
#     # take only X words which has df < N
#     return [x[0] for x in dropwhile(lambda x: x[1] > int(N * 1), dfs)][:X]
#==============================================================================

def tf_idf_features(document, idfs):
    '''Produces value features "tf-idf('word1'): <value>" '''
    features = {}
    counter = Counter(document)
    for i in idfs:
        tf = counter[i[0]]
        if tf > 0:
            tf = (1+math.log(tf,10))
        features['tf-idf(%s)' % (i[0])] = tf*i[1]
    return features


def count_trigrams_features(document, trigr):
    '''Produces count features "count('word1','word2','word3'): <number>" '''
    features = {}
    for t in trigr:
        summ = 0
        for i in range(2,len(document)):
            if((document[i-2],document[i-1],document[i]) == t):
                summ+=1
        features['count(%s,%s,%s)' % (document[i-2],document[i-1],document[i])] = summ
    return features

def count_bigrams_features(document, bigr):
    '''Produces count features "count('word1','word2'): <number>" '''
    features = {}
    for b in bigr:
        summ = 0
        for i in range(1,len(document)):
            if((document[i-1],document[i]) == b):
                summ+=1
        features['count(%s,%s)' % (document[i-1],document[i])] = summ
    return features

def has_trigrams_features(document, trigr):
    '''Produces binary features "has('word1','word2','word3'): <true|false>" '''
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
    '''Produces binary features "has('word1','word2'): <true|false>" '''
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
    '''Produces binary features "has('word'): <true|false>" '''
    features = {}
    document = set(document)
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

#==============================================================================
#                            End of Features Functions
#==============================================================================

#==============================================================================
#  Create idfs for a set of keywords
#==============================================================================

def get_idfs(documents, keywords):
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
        if df > 0:    
            idfs.append( (keyword, math.log(N/df,10) ))
    return idfs

#==============================================================================
#  Features thresholding
#==============================================================================

def freq_filter(features):
    '''Keeps only 1000 the most frequent features'''
    return nltk.FreqDist(features).keys()[:1000]
   

#==============================================================================
#  Collocations Creators 
#==============================================================================

def create_bi_collocations(features_words,document_preprocess):
    finder = BigramCollocationFinder.from_words(movie_reviews.words())
    finder.apply_freq_filter(3)
    bicoll = finder.nbest(bigram_measures.pmi,1000)
    for f in document_preprocess:
        bicoll = [(f(a),f(b)) for (a,b) in bicoll if (f(a) and f(b))]
    return bicoll

def create_tri_collocations(features_words,document_preprocess):
    finder = TrigramCollocationFinder.from_words(movie_reviews.words())
    finder.apply_freq_filter(3)
    tricoll = finder.nbest(trigram_measures.pmi,1000)
    for f in document_preprocess:
        tricoll = [(f(a),f(b)) for (a,b) in tricoll if (f(a) and f(b))]
    return tricoll

#==============================================================================
#  Analysis function
#==============================================================================
def analysis(documents, document_preprocess, features_words, 
             features_preprocess, features_func):
    '''
    Entry point to analysis, creates feature sets, trains a classificator and
    calls evaluation
    '''
    
# Preprocessing    
    for f in document_preprocess:
        for i in range(len(documents)):
            documents[i][0] = filter(None,map(f, documents[i][0]))  
        features_words = filter(None,map(f,features_words))

    for feat_func in features_preprocess:
        features_words = feat_func(features_words)
    features_words = set(features_words)


    featuresets = []
    print( list(features_words)[:10] )
    
# Features creation
    
    if (("has_bigram" or "count_bigram") in features_func):
        bigr = bigrams(features_words)
    if (("has_trigram" or "count_trigram") in features_func):    
        trigr = trigrams(features_words)
    if (("has_bcoll" or "count_bcoll") in features_func):    
        bcoll = create_bi_collocations(features_words,document_preprocess)    
    if (("has_tcoll" or "count_tcoll") in features_func):    
        tcoll = create_tri_collocations(features_words,document_preprocess)
    if (("tf-idef") in features_func):    
        idf = get_idfs(documents, features_words)    
    
    l = len(documents)
    for i in range(l):
        print(".",end="")
        features = {}
        
        for f in features_func:
            if f == "has_feature":
                features.update(has_features(documents[i][0], features_words))
            elif f == "has_bigram":
                features.update(has_bigrams_features(documents[i][0], bigr))
            elif f == "has_trigram":
                features.update(has_trigrams_features(documents[i][0], trigr))

            elif f == "has_bcoll":
                features.update(has_bigrams_features(documents[i][0], bcoll))
            elif f == "has_tcoll":
                features.update(has_trigrams_features(documents[i][0], tcoll))
            
            elif f == "count_feature":
                features.update(count_features(documents[i][0], features_words))
            elif f == "count_bigram":
                features.update(count_bigrams_features(documents[i][0], bigr))
            elif f == "count_trigram":
                features.update(count_trigrams_features(documents[i][0], trigr))
            
            elif f == "count_bcoll":
                features.update(count_bigrams_features(documents[i][0], bcoll))
            elif f == "count_tcoll":
                features.update(count_trigrams_features(documents[i][0], tcoll))
                
            elif f == "tf-idf":
                features.update(tf_idf_features(documents[i][0], idf))

                
        featuresets.append((features, documents[i][1]))
        
    threshold = int(len(documents)*0.8)
    
    train_set = featuresets[:threshold]
    test_set = featuresets[threshold:]
    print("")

# Training    

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    x = evaluate(classifier, test_set)
    classifier.show_most_informative_features(n=20)
    return x

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
#  Helper functions for feature cleaning
#==============================================================================

def rm_punct(feature):      
    return ''.join([c for c in feature.lower() if re.match("[a-z\-\' \n\t]", c)]) 


def rm_stops(feature):
    if feature.lower() in STOP_WORDS:
        return None   
    else:
        return feature

#==============================================================================
#  Program main block
#==============================================================================

# Load documents in format [ (list(words ...), label), (list(words ...), label) ]
documents = []
for category in movie_reviews.categories():
    file_ids = movie_reviews.fileids(category)
    for fileid in file_ids:
        documents.append( [movie_reviews.words(fileid), category] )

# All words across all documents
feature_candidates = movie_reviews.words()[:]




# Define analysis mix
analysis_functions = [
  # in the first tuple we have the preprocessing function applied on 
  # both documents and features
  #
  # Possible Values (1 or more):
  #   - str.Lower 
  #   - p_stem.stem 
  #   - l_stem.stem
  #   - wnl.lemmatize
  #   - rm_punct
  #   - rm_stops
  #
  # the second tuple contains the functions applied only on features
  #
  # Possible Values 
  #   - freq_filter
  #  
  # the third tuble contains the kind of features to compute
  #
  # Possible Values (1 or more):
  #   - "has_feature"
  #   - "count_feature"
  #   - "has_bigram"
  #   - "count_bigram"
  #   - "has_trigram"
  #   - "count_trigram"
  #   - "count_bcoll"
  #   - "count_tcoll"
  #   - "tf-idf"
 
    ((str.lower, ),     (freq_filter,),     ("has_feature","count_feature",)),
  #  ((p_stem.stem, ),   (freq_filter,),     ("has_feature",)),
  #  ((l_stem.stem, ),   (freq_filter,),     ("has_feature",)),
  #  ((wnl.lemmatize, ), (freq_filter,),     ("has_feature",)),        
  #  ((rm_punct,),       (freq_filter,),     ("has_feature",)),
    ((rm_stops,),       (freq_filter,),     ("has_feature",)),
    
]


#==============================================================================
#  Run `SAMPLES` times every analyse mix from `analysis_functions` and save
#  the result (tuple of accuracy, precision, recall, f-measure) to `results`
#==============================================================================
SAMPLES = 1

results = []
for i in range(SAMPLES):
    random.shuffle(documents)
    results.append([])
    for doc_clean, feat_clean, features_func in analysis_functions:
        #try:
        result = analysis(documents, doc_clean, feature_candidates, feat_clean, features_func)
        #except Exception as e:
         #   print(e)
         #   result = (0,0,0,0)
        print ("Accuracy: {:.2f} - Precision: {:.2f} - Recall: {:.2f} - "
                "F-measure: {:.2f}".format(*result))
        results[i].append(result)
    print("")
print("")



print ("-"*106)
print ("| {:<10}\t| {:<10}\t| {:<10}\t| {:<10}\t| {:<55}|".format("Acc.","Prec.","Rec.","F-Meas.","Setup"))
print ("-"*106)
    
   

for col in range(len(analysis_functions)):
    sums = [0, 0, 0, 0]
    for row in range(SAMPLES):
        for i in range(len(sums)):
            sums[i] += results[row][col][i]

    for i in range(len(sums)):
        sums[i] /= SAMPLES
    descr = " ".join(elem.__name__ for elem in analysis_functions[col][0]) + " " + " ".join(elem.__name__ for elem in analysis_functions[col][1]) + " " + " ".join(elem for elem in analysis_functions[col][2])    
    print ("| {0:1.2f}\t| {1:1.2f}\t| {2:1.2f}\t| {3:1.2f}\t| {4:<55}|".format(
        sums[0],sums[1],sums[2],sums[3],descr[:50])
    )  
         
print ("-"*106)
    


