# -*- coding: utf-8 -*-
"""
Created on Fri May  3 02:29:28 2013

@author: Seby
"""
from __future__ import division, print_function
import random,nltk,re
from nltk.corpus import movie_reviews
from numpy import array
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.collocations import *
from itertools import chain

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()


finder = BigramCollocationFinder.from_words(movie_reviews.words())
finder2 = TrigramCollocationFinder.from_words(movie_reviews.words())

print(sorted(finder.nbest(bigram_measures.raw_freq, 10)))
print(sorted(finder.nbest(trigram_measures.raw_freq, 10)))

documents = [(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

all_words = nltk.FreqDist(w for w in movie_reviews.words())


wnl = nltk.WordNetLemmatizer()
p_stemmer = nltk.PorterStemmer()  
l_stemmer = nltk.LancasterStemmer()


print(".")    

#==============================================================================
#  :-)
#==============================================================================

def ret(string):
    return string

#==============================================================================
#  Create Sets Functions
#==============================================================================

def create_sets2(documents,function,words,tresh):
    #word_features = 
    
    def document_features(document):        
        document_word = set(map(function,document))
        features = {}
        for word in words:
            features['count(%s)' % word] = document_word.count(word)            
            
        return features
        
    featuresets = [(document_features(d),c) for (d,c) in documents]
    return [featuresets[:tresh],featuresets[tresh:]]

#==============================================================================
#  Document Features improoved
#==============================================================================

def create_sets(documents,function,words):
    #word_features = 
    
    def document_features(document):        
        document_word = set(map(function,document))
        features = {}
        for word in words:
            features['contains(%s)' % word] = (function(word) in document_word)
        return features
        
    featuresets = [(document_features(d),c) for (d,c) in documents]
    return [featuresets[:tresh],featuresets[tresh:]]
    
#==============================================================================
#  Evaluate a classifier in terms of Accuracy, Precision, Recall, F-Measure
#==============================================================================

def evaluate(classifier, test_set):

    def f_measure(precision,recall,alpha):  
        return 1/(alpha*(1/precision)+(1-alpha)*1/recall)
    
    test = classifier.batch_classify([fs for (fs,l) in test_set])  
    gold = [l for (fs,l) in  test_set]  
    matrix = nltk.ConfusionMatrix(gold, test)
    tp = (matrix['pos','pos'])
    fn = (matrix['pos','neg'])
    fp = (matrix['neg','pos'])
    
    accuracy = nltk.classify.accuracy(classifier, test_set)
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    #f = f_measure(precision,recall,float(raw_input("F-Measure Alpha: ")))    
    f = f_measure(precision,recall,0.5)    
    
    return array([accuracy,precision,recall,f])


#==============================================================================
#  Analysis definition
#==============================================================================
def analysis(documents,create_sets,preprocess_function,words):
    train_set, test_set = create_sets(documents,preprocess_function,words)
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(classifier.show_most_informative_features(n=20))
    x = evaluate(classifier, test_set)
    print ("Accuracy: {:.2f} - Precision: {:.2f} - Recall: {:.2f} - F-measure: {:.2f}".format(*x)) 
    return x

r1=[]
r2=[]
r3=[]
r4=[]
r5=[]
r6=[]
r7=[]
r8=[]
r9=[]
r10=[]

for i in range(5):
     random.shuffle(documents)  
     r1.append(analysis(documents,str.lower,all_words.keys()[:1000],len(documents)*0.8))
     r2.append(analysis(documents,wnl.lemmatize,all_words.keys()[:1000],len(documents)*0.8))
     r3.append(analysis(documents,p_stemmer.stem,all_words.keys()[:1000],len(documents)*0.8))
     r4.append(analysis(documents,ret,([w for w in all_words.keys() if not w in [",",".","?","!",":",";","-","'","\""]])[:1000],len(documents)*0.8))
     r5.append(analysis(documents,ret,([w for w in all_words.keys() if not w in [",",".","?","!",":",";","-","'","\""]+stopwords.words('english')])[:1000],len(documents)*0.8))
     r6.append(analysis(documents,wnl.lemmatize,([w for w in all_words.keys() if not w in stopwords.words('english')])[:1000],len(documents)*0.8))
     r7.append(analysis(documents,p_stemmer.stem,([w for w in all_words.keys() if not w in stopwords.words('english')])[:1000],len(documents)*0.8))
     r8.append(analysis(documents,ret,([w for w in all_words.keys() if not w in stopwords.words('english')])[:1000],len(documents)*0.8))
    
      
print(sum(r1)/len(r1))  
print(sum(r2)/len(r2))  
print(sum(r3)/len(r3))  
print(sum(r4)/len(r4))  
print(sum(r5)/len(r5))  
print(sum(r6)/len(r6))  
print(sum(r7)/len(r7))  
print(sum(r8)/len(r8))  


  
