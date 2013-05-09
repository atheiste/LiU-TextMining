# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:48:17 2013

@author: Seby
"""

from __future__ import division, print_function
import nltk, re, pprint
from urllib import urlopen
from itertools import chain

#==============================================================================
# Importing a web page from the Internet
#==============================================================================
#url = "http://www.guardian.co.uk/politics/2013/apr/08/iron-lady-margaret-thatcher"
#html = urlopen(url).read()

#==============================================================================
# Importing an html file stored in the hard disk
#==============================================================================
f = open('Margaret.html')
html = f.read()
f.close()

raw = nltk.clean_html(html)
text = (raw[7364:22643])


#==============================================================================
# 1.Extract some 3-500 words of main text content using NLTK tools. It is
# likely that you need to filter out some tabs and newlines. For this purpose
# the regular expression module (re) is useful. If you fail you may clean your
# text in an editor. (NLTK p. 80-86)
#==============================================================================
subText = text[0:1691]
print("Extracted words::\n", subText)

#==============================================================================
# 2.Normalize...
#==============================================================================

text = re.sub('[^(\x20-\x7F)]*','',subText)

#==============================================================================
# segment, tokenize...
#==============================================================================

sents = nltk.sent_tokenize(text)
tokens = map(nltk.word_tokenize, sents)


print ("\n\n-------------------------------\n\n")

print ("Tokenized sentences::\n", tokens)

print ("\n\n-------------------------------\n\n")


porter = nltk.PorterStemmer()
stemmedWords = map(porter.stem, chain(*tokens))
print ("PortStemmed words::\n", stemmedWords)
# chemistri ?!?

wnl = nltk.WordNetLemmatizer()
lemmatizedLSTT = map(wnl.lemmatize, chain(*tokens))
print ("WordNetLematized words::\n", lemmatizedLSTT)
# wa ?!?!??

#==============================================================================
# 3. Use one of the NLTK taggers to tag the text for parts-of-speech. Inspect
# the results and estimate its accuracy. (NLTK, p. 179-210)
#==============================================================================

taggedLLSTT = map(nltk.pos_tag, tokens)
print ("Tagged sentences::\n", taggedLLSTT)

#==============================================================================
# 4.Apply the NLTK named-entity recognizer to the text. Evaluate its
# performance using precision and recall. This means you have to identify
# names in the text  and in the system output yourself. (NLTK, 239; 281-284);
#==============================================================================

for s in taggedLLSTT:
    print (nltk.ne_chunk(s, binary=False))
