# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:48:17 2013

It should work. Some parts of code need to be rewritten in a more pythonic way.
It takes up to 8 minutes to download 1997 descriptions. It takes 2 more minutes
to build the Inverted Index and 3 more minutes to build the Doc Weight Matrix.

I'm using the most popular app in the US to avoid having descriptions with
oriental characters. Let me know what do you think. If there is too much
"black magic" just ask me any question.

C ya :-)

@author: Seby
"""

from __future__ import division, print_function
import sys
import os
import nltk, re, pprint, math, operator, time, threading, urllib2
from os.path import dirname, join
from itertools import chain

def map_async(function, args, threads=10):
	'''
	Simulates built-in `map` function. Each call is asynchronous
	using thread pool with variable number of thread
	'''
	def wrap():
		while(1):
			try:
				lock.acquire()
				param = args_copy.pop()
			except:
				return
			finally:
				lock.release()

			try:
				result = function(param)
			except Exception as e:
				print(e)
				return

			try:
				lock.acquire()
				param = res.append(result)
			except:
				return
			finally:
				lock.release()

	lock = threading.Lock()
	args_copy = list(args)
	threads = [threading.Thread(target=wrap) for i in range(threads)]
	res = []

	for thread in threads: thread.start()
	for thread in threads: thread.join()

	return res


#==============================================================================
# Step1: App crawling
# You will access the app websites such as Google Play (https://play.google.com
# /store) and AppBrain (http://www.appbrain.com/) to obtain the desired
# description texts for the apps. Thereis no constraint on which kind of
# apps you choose. The number of apps should not be less than
# 1000. Store the description text in the files.
#==============================================================================

categories = [
"business",
"comics",
"communication",
"education",
"entertainment",
"finance",
"health-and-fitness",
"medical",
"sports"
]

ROOT = join(dirname(__file__), "apps-descs")

#==============================================================================
# Open a web page
#==============================================================================

def open_web_page(url):
	opener = urllib2.build_opener()
	# without the following line i dont get any response
	# with, i get some HTTP 500
	opener.addheaders = [('User-agent', 'Mozilla/19.0')]
	response = opener.open(url)
	page = response.read()
	opener.close()
	return page

#==============================================================================
# Searching all the URLs from a generic pages leading to an App
#==============================================================================

def get_app_urls(url):
	urls=[]
	print("Parsing {0}".format(url))

	for i in range(30): #black magic:
		page = open_web_page(url)
		m = re.findall('<a href=\"\/app\/(.*?)\" class',page)
		if len(m) > 0:
			break
		time.sleep(5)

	for t in m:
		urls.append("http://www.appbrain.com/app/"+t)
		print(".",end="")
	return urls

#==============================================================================
# Searching the description of the App
#==============================================================================
patTitle = re.compile('h1 itemprop=\"name\".*?>(.*?)<',
				 re.U|re.L|re.M|re.I|re.S)
patDescr = re.compile('itemprop=\"description\".*?>(.*?)</div',
				 re.U|re.L|re.M|re.I|re.S)
patClean = re.compile('<a.*?/a>|&[\w]+;|\\[tn]|<[^<]+?>',
				 re.U|re.L|re.M|re.I|re.S)
patAdult = re.compile('violates the Android market guidelines',
				 re.U|re.L|re.M|re.I|re.S)

# This will return the title and the description of the app
def get_app_descs(args):
	i, url = args

	for j in range(30): # more black magic
		page = open_web_page(url)
		m = "".join(patDescr.findall(page))

		if len(m) > 0:
			break
		if j > 10:
			w = "".join(patAdult.findall(page))
			if w > 0:
				break
		time.sleep(5)

	m = patClean.sub(" ", m)
	descr = "".join(patTitle.findall(page)) + "\n\n" + m

	filename = "App{0!s}.txt".format(i)
	f = open(join(ROOT, filename), "w")
	f.write(descr)
	f.close()

	return descr


#==============================================================================
# Loading Data
#==============================================================================
descriptions = {}

if not os.path.exists(ROOT) :

	print("Creating folder")
	os.mkdir(ROOT)

	print("Downloading descriptions")
	print("--- Getting URLs")

	app_urls = []
	thrds = []

	category_urls = []

	for s in categories:
		for i in range(0,200,10):
			category_urls.append("http://www.appbrain.com/apps/country-united-states/{0}/?o={1!s}".format(s, i))

	app_urls = map_async(get_app_urls, category_urls)  # extract app urls from page at category_urls
	app_urls = list(chain(*app_urls))  # flatten 2D list into 1D list

	print("\n\n{0!s} URLs collected\n--- Getting descriptions".format(len(app_urls)))

	app_descs = map_async(get_app_descs, enumerate(app_urls))
	print("Finished collecting application descriptions {}/{}".format(len(app_descs), len(app_urls)))

	descriptions = dict(enumerate(app_descs))  # I know ...

else:
	print("The descriptions have already been downloaded")
	print("--- Loading files")

	for filename in os.listdir(ROOT):
		if filename.startswith("App"):
			f = open(join(ROOT, filename), "r")
			i = re.match("App(\d+).txt", filename)
			descriptions[int(i.group(1))] = f.read()
			f.close()
			print(".", end="")

#==============================================================================
# Step2: Index construction
# Build inverted index on the texts. You will use NLTK to preprocess the text,
# such as tokenizing, normalizing, etc. Compute and store tf, df in the
# inverted index.
#==============================================================================


#==============================================================================
# tokenizing & normalizing (lower case + lemmatizing)
#==============================================================================

documents={}
wnl = nltk.WordNetLemmatizer()
for key in descriptions:
	documents[key]= map(
	wnl.lemmatize,(nltk.word_tokenize(re.sub("\W"," ",descriptions[key].lower()))))

#==============================================================================
# tf = number of times that t occurs in d
#==============================================================================
# df = number of documents in the collection that the term occurs in
#==============================================================================
# inverted index = For each term t, we store a list of all documents that
# contains t.
#==============================================================================


#==============================================================================
# creating a list containing the unique words in the documents
#==============================================================================

keywords = list(set(chain(*documents.values())))

#==============================================================================
# Creating the Inverted Index:
#
# An element of invInd is: 'term':[df,{IndexDoc1:tf,IndexDoc2:tf,...}]
# Ex. 'student':[3,{10:1, 43:2, 1345:1}]'
#==============================================================================

def buildInvIndex(keywords, documents):
	print("\n--- Building Inverted Index")
	invInd = {}
	for k in keywords:
		d = {}
		for key in documents.iterkeys():
			if k in documents[key]:
				d[key]=documents[key].count(k)
		invInd[k]=[len(d.keys()),d]
	return invInd

invInd = buildInvIndex(keywords,documents)

#==============================================================================
# Step3: Query processing
# Vector space model. The input parameter is the set of keywords and integer k,
# for top-k query answering. The query processor should compute the
# vector space similarities of the query to the documents. Top-k documents
# are returned according to the ranked similarity values.
#==============================================================================


#==============================================================================
# Function that creates the document weight matrix
#
# w = tf*idf = (1+log(tf))*(log(N/df))
#==============================================================================

def buildDocWeightMatrix(keywords,documents,invInd):
	print("\n--- Building Doc Weight Matrix\n")
	N = len(documents)
	wMatrix = []
	s = sorted(documents.keys())
	for key in s:
		wMatrixRow = []
		for k in keywords:
			try:
				tf = invInd[k][1][key]
			except KeyError:
				tf = 0
			if tf > 0:
				tf = (1+math.log(tf,10))
			idf = math.log(N/invInd[k][0],10)
			wMatrixRow.append(tf*idf)
		wMatrix.append(wMatrixRow)

# vector normalization
	l = len(wMatrix)
	for i in range(l):
		somma = sum(x**2 for x in wMatrix[i])
		radice = math.sqrt(somma)
		if radice > 0:
			(wMatrix[i]) = [x/radice for x in wMatrix[i]]
	return wMatrix

#==============================================================================
# Function that creates the query vector representation
#
# w = tf*idf
#
# tf = 1 if the user keyword is in our keywords 0 otherwise
#==============================================================================

def buildQueryVector(query,keywords,documents,invInd):
	userKeywords = map(
	wnl.lemmatize,(nltk.word_tokenize(re.sub("\W"," ",query.lower()))))

	N = len(documents)
	vector = []
	for k in keywords:
		idf = 0;
		tf = 0;
		if k in userKeywords:
			tf = 1
			df = invInd[k][0]
			if df > 0:
				idf = math.log(N/df,10)
		w = tf*idf
		vector.append(w)
	return vector

#==============================================================================
#  Distance
#==============================================================================

def computeDistance(v1,v2):
	return sum([a*b for a,b in zip(v1,v2)])

#==============================================================================
#  Execute Query
#==============================================================================

def queryExec(queryVector,wMatrix,k):
	results = {}
	l = len(wMatrix)
	for i in range(l):
		results["App "+str(sorted(documents.keys())[i])] = computeDistance(
		queryVector,wMatrix[i])

	sorted_results = sorted(results.iteritems(), key=operator.itemgetter(1),
							reverse=True)
	pprint.pprint(sorted_results[:k])

#==============================================================================
#  Reading Input
#==============================================================================
wMatrix = buildDocWeightMatrix(keywords,documents,invInd)

while 1:
	query = raw_input("Insert a set of keywords (empty for exit): ")
	if not query:
		print("Quitting")
		sys.exit(0)
	try:
		k=input('Max n. of results:')
		queryVector = buildQueryVector(query,keywords,documents,invInd)
		print("")
		queryExec(queryVector, wMatrix, k)
		print("")
	except ValueError:
		print ("Not a number")
		print("")
