# coding: utf-8
"""
The application was optimized to run using PyPy 2.0 so we strongly recommend
to run on PyPy. The time difference is between 10 - 12 times (that counts).

Laboratory 5 
@author: seby912 & tompe625
"""
from __future__ import division, print_function

import math
import nltk
import re
import random
import sys

from collections import Counter
from itertools import dropwhile
from nltk import bigrams
from nltk import trigrams
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from numpy import array

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

def df_feats(documents, features):
     '''Removes unusable features based on DF value'''
     features_set = set(features)
     N = len(documents) // 2
     X = 600
     dfs = get_idfs(documents, features_set)
     # sort according to idf value
     dfs.sort(key=lambda x: x[1], reverse=True)
     # take only X words which has df < N
     return [x[0] for x in dropwhile(lambda x: x[1] > int(N * 1), dfs)][:X]

def avg_w_l_feats(document):
    '''Produces value features "avg<>n: <true|false>" '''
    features = {}
    avg = sum([len(w) for w in document])/len(document)
    features['avg<2'] = (avg<2)  
    features['2<=avg<=4'] = (avg>=2 and avg<=4)
    features['avg>4'] = (avg>4)  
    return features

def lex_d_feats(document):
    '''Produces value features "lex<>n: <true|false>" '''
    features = {}
    lex = len(set(document))/len(document)
    features['lex<2'] = (lex<2)  
    features['2<=lex<=4'] = (lex>=2 and lex<=4)
    features['lex>4'] = (lex>4)  
    features = {}
    return features

def tf_idf_feats(document, idfs):
    '''Produces value features "tf-idf('word1')<>n: <true|false>" '''
    features = {}
    counter = Counter(document)
    for i in idfs:
        tf = counter[i[0]]
        if tf > 0:
            tf = (1+math.log(tf,10))
        summ = tf*i[1]
        features['tf-idf(%s)<0.5' % (i[0])] = (summ<0.5)
        features['0.5<=tf-idf(%s)<=1.5' % (i[0])] = (summ>=0.5 and summ<=1.5)  
        features['tf-idf(%s)>1.5' % (i[0])] = (summ>1.5)
    return features

def count_trigrams_feats(document, trigr):
    '''Produces count features "count('word1','word2','word3')<>n: <true|false>" '''
    features = {}
    for t in trigr:
        summ = 0
        for i in range(2,len(document)):
            if((document[i-2],document[i-1],document[i]) == t):
                summ+=1
        features['count(%s,%s,%s)<2' % (document[i-2],document[i-1],document[i])] = (summ<2)
        features['2<=count(%s,%s,%s)<=4' % (document[i-2],document[i-1],document[i])] = (summ>=2 and summ<=4)
        features['count(%s,%s,%s)>4' % (document[i-2],document[i-1],document[i])] = (summ>4)    
        
    return features

def count_bigrams_feats(document, bigr):
    '''Produces count features "count('word1','word2')<>n: <true|false>" '''
    features = {}
    for b in bigr:
        summ = 0
        for i in range(1,len(document)):
            if((document[i-1],document[i]) == b):
                summ+=1        
        features['count(%s,%s)<2' % (document[i-1],document[i])] = (summ<2)
        features['2<=count(%s,%s)<=4' % (document[i-1],document[i])] = (summ>=2 and summ<=4)
        features['count(%s,%s)>4' % (document[i-1],document[i])] = (summ>4)    
    return features

def has_trigrams_feats(document, trigr):
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
    
def has_bigrams_feats(document, bigr):
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

def has_feats(document, features_words):
    '''Produces binary features "has('word'): <true|false>" '''
    features = {}
    document = set(document)
    for word in features_words:
        features['has({})'.format(word)] = (word in document)
    return features

def count_feats(document, features_words):
    '''Produces count features "count('word')<>n: <true|false>" '''
    features = {}
    counter = Counter(document)
    for word in features_words:
        summ = counter[word]
        features['count(%s)<2' % (word)] = (summ<2)  
        features['2<=count(%s)<=4' % (word)] = (summ>=2 and summ<=4)
        features['count(%s)>4' % (word)] = (summ>4)        
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
        tricoll = [(f(a),f(b),f(c)) for (a,b,c) in tricoll if (f(a) and f(b) and f(c))]
    return tricoll

    
#==============================================================================
#  Preprocessing  
#==============================================================================
    
def pre_process(documents, document_preprocess, features_words, 
                features_preprocess, light):
    for f in document_preprocess:
        for i in range(len(documents)):
            documents[i][0] = [f(w) for w in documents[i][0] if f(w)]
            
        features_words = [f(w) for w in features_words if f(w)]

    for feat_func in features_preprocess:
        features_words = feat_func(features_words)
    
    bigr = []    
    trigr = []
    bcoll = []    
    tcoll = []
    idf = []
    
    if not light:
        bigr = bigrams(features_words)
        trigr = trigrams(features_words)
        bcoll = create_bi_collocations(features_words,document_preprocess)    
        tcoll = create_tri_collocations(features_words,document_preprocess)
        idf = get_idfs(documents, features_words)  
    return [documents,features_words,bigr,trigr,bcoll,tcoll,idf]

#==============================================================================
#  Feature Set
#==============================================================================
    
def get_results(preprocess_results,features_func):
    documents,features_words,bigr,trigr,bcoll,tcoll,idf = preprocess_results    
    featuresets = []    
    l = len(documents)
    for i in range(l):
        print(".",end="")
        features = {}
        
        for f in features_func:
            if f == "has_feature":
                features.update(has_feats(documents[i][0], features_words))
            elif f == "has_bigram":
                features.update(has_bigrams_feats(documents[i][0], bigr))
            elif f == "has_trigram":
                features.update(has_trigrams_feats(documents[i][0], trigr))

            elif f == "has_bcoll":
                features.update(has_bigrams_feats(documents[i][0], bcoll))
            elif f == "has_tcoll":
                features.update(has_trigrams_feats(documents[i][0], tcoll))
            
            elif f == "count_feature":
                features.update(count_feats(documents[i][0], features_words))
            elif f == "count_bigram":
                features.update(count_bigrams_feats(documents[i][0], bigr))
            elif f == "count_trigram":
                features.update(count_trigrams_feats(documents[i][0], trigr))
            
            elif f == "count_bcoll":
                features.update(count_bigrams_feats(documents[i][0], bcoll))
            elif f == "count_tcoll":
                features.update(count_trigrams_feats(documents[i][0], tcoll))
                
            elif f == "tf-idf":
                features.update(tf_idf_feats(documents[i][0], idf))
                
            elif f == "lex_d":
                features.update(lex_d_feats(documents[i][0]))
                
            elif f == "avg_w_l":
                features.update(avg_w_l_feats(documents[i][0]))

                
        featuresets.append((features, documents[i][1]))
        
    threshold = int(len(documents)*0.8)
    

    # Training    

    classifier = nltk.NaiveBayesClassifier.train(featuresets[:threshold])
    x = evaluate(classifier, featuresets[threshold:])
    classifier.show_most_informative_features(n=20)
    return array(x)

#==============================================================================
#  Evaluate a classifier in terms of Accuracy, Precision, Recall, F-Measure
#==============================================================================

def evaluate(classifier, test_set):

    def f_measure(precision,recall,alpha):
        try:
            return 1/(alpha*(1/precision)+(1-alpha)*1/recall)
        except ZeroDivisionError:
            print("Division by 0")
            return 1

    test = classifier.batch_classify([fs for (fs,l) in test_set])
    gold = [l for (fs,l) in  test_set]
    matrix = nltk.ConfusionMatrix(gold, test)
    print("")
    
    print(matrix)
    tp = (matrix['pos','pos'])
    fn = (matrix['pos','neg'])
    fp = (matrix['neg','pos'])

    accuracy = nltk.classify.accuracy(classifier, test_set)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f = f_measure(precision, recall, 0.5)
    print ("F-measure: {:.2f} - Accuracy: {:.2f} - Precision: {:.2f} - "
    "Recall: {:.2f}".format(f,accuracy,precision,recall))
    return [f,accuracy,precision,recall]


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
#  Show results in a table
#==============================================================================

def show_results(results,descr):
    table = []
    print ("-"*106)
    print ("| {:}| {:}| {:}| {:}| {:<67}|".format("F-Meas.","Acc.   ","Prec.  ","Rec.   ","Setup  "))
    print ("-"*106)
        
    for i in range(len(results)):
        sums = sum(results[i])/len(results[i])
        table.append("|  {0:1.3f} |  {1:1.3f} |  {2:1.3f} |  {3:1.3f} | {4:<67}|".format(
            sums[0],sums[1],sums[2],sums[3],descr[i])
        )
        
    for i in sorted(table,reverse=True):
        print(i)
    print ("-"*106)


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

stepA = [(str.lower, ),     
         (freq_filter,),     
         ("has_feature",)]

stepA_d = ["Lower + Freq. Filter + Has(feat)"]

stepB = [(rm_stops,p_stem.stem, ),
         (rm_stops,rm_punct,p_stem.stem, ),
         (str.lower, ),     
         (p_stem.stem, ),   
         #(l_stem.stem, ),   
         (wnl.lemmatize, ),         
         (rm_punct,),       
         (rm_stops,),      
         (rm_stops,wnl.lemmatize, ),  
         (rm_punct,p_stem.stem, ),    
         (rm_punct,wnl.lemmatize, ),  
        ]

stepB_d = ["Rm Stop words + Port Stemmer + Freq. Filter + Has(feat)",
           "Rm Stop words + Rm Puncts + Port Stemmer + Freq. Filter + Has(feat)",
           "Lower + Freq. Filter + Has(feat)",
           "Port Stemmer + Freq. Filter + Has(feat)",
           #"Lancaster Stemmer + Freq. Filter + Has(feat)",
           "Lemmatizer + Freq. Filter + Has(feat)",
           "Rm Puncts + Freq. Filter + Has(feat)",
           "Rm Stops + Freq. Filter + Has(feat)",
           "Rm Stops + Lemmatizer + Freq. Filter + Has(feat)",
           "Rm Puncts + Port Stemmer + Freq. Filter + Has(feat)",           
           "Rm Puncts + Lemmatizer + Freq. Filter + Has(feat)",                     
           ]

stepC = [ ("has_feature",),
          ("has_feature","has_bigram",),
          ("has_feature","has_trigram",),
          ("has_feature","has_bcoll",),
          ("has_feature","has_tcoll",),
          ("has_feature","count_feature",),
          ("has_feature","count_bigram",),
          ("has_feature","count_trigram",),
          ("has_feature","count_bcoll",),
          ("has_feature","count_tcoll",),
          ("has_feature","avg_l_w",),
          ("has_feature","lex_d",),
        ]
 
stepC_d = ["Rm Stop w. + P. Stem. + Has(feat)",
           "Rm Stop w. + P. Stem. + Has(feat) + Has(bigr)",
           "Rm Stop w. + P. Stem. + Has(feat) + Has(trigr)",
           "Rm Stop w. + P. Stem. + Has(feat) + Has(b-coll)",
           "Rm Stop w. + P. Stem. + Has(feat) + Has(t-coll)",
           "Rm Stop w. + P. Stem. + Has(feat) + Count(feat)",
           "Rm Stop w. + P. Stem. + Has(feat) + Count(bigr)",
           "Rm Stop w. + P. Stem. + Has(feat) + Count(trigr)",
           "Rm Stop w. + P. Stem. + Has(feat) + Count(b-coll)",         
           "Rm Stop w. + P. Stem. + Has(feat) + Count(t-coll)",
           "Rm Stop w. + P. Stem. + Has(feat) + Avarage W. Lenght",
           "Rm Stop w. + P. Stem. + Has(feat) + Lexical Diversity",                    
           ]    

stepC1_d = ["Rm Stop w. + Rm Puncts + P. Stem. + Has(feat)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Has(bigr)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Has(trigr)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Has(b-coll)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Has(t-coll)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Count(feat)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Count(bigr)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Count(trigr)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Count(b-coll)",         
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Count(t-coll)",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Avarage W. Lenght",
           "Rm Stop w. + Rm Puncts  + P. Stem. + Has(feat) + Lexical Diversity",                    
           ]       

stepD = [ ("has_feature","tf-idf",),
          ("has_feature",),
          ("has_feature","has_bcoll",),
          ("has_feature","has_tcoll",),
        ]
        
stepD_d = ["Rm Stop w. + P. Stem. + Has(feat) + Tf-Idf(feat)",
           "Rm Stop w. + P. Stem. + Has(feat)",
           "Rm Stop w. + P. Stem. + Has(feat) + Has(b-coll)",
           "Rm Stop w. + P. Stem. + Has(feat) + Has(t-coll)",
          ]            

#==============================================================================
#  Run `SAMPLES` times every analyse mix from `analysis_functions` and save
#  the result (tuple of accuracy, precision, recall, f-measure) to `results`
#==============================================================================
SAMPLES = int(sys.argv[1])

for arg in sys.argv[2:]:
    
    if arg == "A":
        results = []
        for i in range(SAMPLES):
            random.shuffle(documents)
            pp = pre_process(documents, stepA[0], feature_candidates, stepA[1],True)
            results.append(get_results(pp, stepA[2]))
        show_results([results],stepA_d)

    
    if arg == "B":
        results = []
        for i in range(SAMPLES):
            random.shuffle(documents)
            for l in range(len(stepB)):
                if l == len(results): 
                    results.append([])
                pp = pre_process(documents, stepB[l], feature_candidates, stepA[1],True)
                results[l].append(get_results(pp, stepA[2]))
        show_results(results,stepB_d)

    
    if arg == "C":
        results = []
        results1 = []
                    
        for i in range(SAMPLES):
            random.shuffle(documents)
            pp = pre_process(documents, stepB[0], feature_candidates, stepA[1],False)
            for l in range(len(stepC)):
                if l == len(results): 
                    results.append([])
                results[l].append(get_results(pp, stepC[l])) 
                
            pp = pre_process(documents, stepB[1], feature_candidates, stepA[1],False)
            for l in range(len(stepC)):
                if l == len(results1): 
                    results1.append([])
                results1[l].append(get_results(pp, stepC[l])) 

        show_results(results,stepC_d)             
        show_results(results1,stepC1_d)        
    
    
    if arg == "D":
        results = []
        for i in range(SAMPLES):
            random.shuffle(documents)
            pp = pre_process(documents, stepB[0],feature_candidates, stepA[1],False)
            for l in range(len(stepD)):
                if l == len(results): 
                    results.append([])
                results[l].append(get_results(pp, stepD[l]))
        show_results(results,stepD_d)
     
        

    


