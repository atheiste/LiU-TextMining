# -*- coding: utf-8 -*-
"""
Laboratory 4 - Data models and Information Retrieval for Textual Data
@author: seby912 & tompe625
"""

from __future__ import division, print_function
import nltk, re, math, operator, time, threading, urllib2,os,sys
from os.path import dirname, join
from itertools import chain


#==============================================================================
# Step1: App crawling
# You will access the app websites such as Google Play (https://play.google.com
# /store) and AppBrain (http://www.appbrain.com/) to obtain the desired
# description texts for the apps. Thereis no constraint on which kind of
# apps you choose. The number of apps should not be less than
# 1000. Store the description text in the files.
#==============================================================================

categories = [
 "BOOKS_AND_REFERENCE",
 "COMMUNICATION",
 "FINANCE",
 "PHOTOGRAPHY",
 "ENTERTAINMENT",
 "EDUCATION",
 "BUSINESS",
 "MEDICAL",
 "WEATHER",
 "MUSIC_AND_AUDIO",
 "NEWS_AND_MAGAZINES",
 "PRODUCTIVITY",
 "HEALTH_AND_FITNESS",
 "SHOPPING",
 "SOCIAL"
]

ROOT = join(dirname(__file__), "apps-descs")

#==============================================================================
# Simulates built-in `map` function. Each call is asynchronous
# using thread pool with variable number of thread	
#==============================================================================

def map_async(function, args, threads=10):
    
    def wrap():
        while(1):
            try:
                lock.acquire()
                param = args_copy.pop()
            except:
                return
            finally:
                lock.release()
                
            result = function(param)
            
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

    return list(reversed(res))

#==============================================================================
# Open a web page
#==============================================================================

def open_web_page(url):
    page =""
    opener = urllib2.build_opener()
    opener.addheaders = [('Host', ''),('Accept-Language','en-gb,en;q=0.5'), 
    ('User-agent', 'Mozilla/5.0(Windows; U; Windows NT 5.1; en-GB; rv:1.9.0.1) Gecko/2008070208Firefox/3.0.1')]
    response = opener.open(url)
    page = response.read()
    opener.close()  
    
    return page
    
#==============================================================================
# Searching all the URLs from a generic pages leading to an App
#==============================================================================

def get_app_urls(url):
    urls=[]
    
    for i in range(15): 
        page = open_web_page(url)
        m = re.findall('<a href=\"\/store\/apps\/details(.*?)\" class',page)
        if len(m) > 0:
            break
        time.sleep(5)

    for t in m:
        urls.append("https://play.google.com/store/apps/details"+t+"&hl=en")
        print(".",end="")
    return urls

#==============================================================================
# Searching the description of the App
#==============================================================================
patTitle = re.compile('<h1.*?>(.*?)<\/h1>',
                 re.U|re.L|re.M|re.I|re.S)
patDescr = re.compile('itemprop=\"description\".*?>(.*?)</div',
                 re.U|re.L|re.M|re.I|re.S)
patDescrT = re.compile('<div id=\"doc-translated-text\".*?>(.*?)</div',
                 re.U|re.L|re.M|re.I|re.S)                   
patClean = re.compile('<a.*?/a>|&.*?;|\\[tn]|<[^<]+?>',
                 re.U|re.L|re.M|re.I|re.S)

# This will return the title and the description of the app
def get_app_descs(args):
    i, url = args
    descs = ""
    m = ""
    for j in range(30): 
        page = open_web_page(url)
        
        # Check if the description is in english or not
        # and search for a translated text
        if page.find("<div class=\"translate-label goog-inline-block\">")<0:
            m = "".join(patDescr.findall(page))
        else:
            m = "".join(patDescrT.findall(page))
        if len(m) > 0:
            break
        time.sleep(5)
            
    if len(m)>0:
        t = "".join(patTitle.findall(page))
        m = patClean.sub(" ", m)
        t = patClean.sub(" ", t)
        descs = ( t + "\n\n" + m)
            
    print(".",end="")
    return descs

#==============================================================================
# Saving Data
#==============================================================================

def save_app_descs(desc):
    l = len(desc)
    for i in range(l):
        filename = "App{0!s}.txt".format(i)
        f = open(join(ROOT, filename), "w")
        f.write(desc[i])
        f.close()

#==============================================================================
# Loading Data
#==============================================================================
allDesc = {}
      
app_descs=[]

if not os.path.exists(ROOT) :

    print("Creating folder")
    os.mkdir(ROOT)

if len(os.listdir(ROOT))==0:
    
      print("Downloading descriptions")
      print("--- Getting URLs")

      app_urls = []
      thrds = []

      category_urls = []
      for s in categories:
         for i in range(0,73,24):
              category_urls.append("https://play.google.com/store/apps/category/{0}/collection/topselling_free?start={1!s}&num=24&hl=en".format(s, i))

      app_urls = map_async(get_app_urls, category_urls)  # extract app urls from page at category_urls
      app_urls = list(chain(*app_urls))  # flatten 2D list into 1D list
 
      print("\n\n{0!s} URLs collected\n\n--- Getting descriptions".format(len(app_urls)))

      app_descs = map_async(get_app_descs, enumerate(app_urls))
      l = len(app_descs)
      print("\n\nFinished collecting application descriptions {}/{}".format(l, len(app_urls)))
      save_app_descs(app_descs)
      allDesc = dict(enumerate(app_descs))

    
else:
    print("The descriptions have already been downloaded\n--- Loading files")
    
    for filename in os.listdir(ROOT):
        if filename.startswith("App"):
            f = open(join(ROOT, filename), "r")
            i = re.match("App(\d+).txt", filename)
            allDesc[int(i.group(1))] = f.read()
            f.close()            
            print(".", end="")
    print("")
                 
#==============================================================================
# Step2: Index construction
# Build inverted index on the texts. You will use NLTK to preprocess the text,
# such as tokenizing, normalizing, etc. Compute and store tf, df in the
# inverted index.
#==============================================================================



#==============================================================================
# tokenizing & normalizing (lower case + lemmatizing)
#==============================================================================
print("\n--- Tokenizing & Normalizing")

documents={}
wnl = nltk.WordNetLemmatizer()
for key in allDesc.iterkeys():
    documents[key]= map(
    wnl.lemmatize,(nltk.word_tokenize(re.sub("\W"," ",allDesc[key].lower()))))

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
print("\n--- Creating a list of keywords")
keywords = list(set(chain(*documents.values())))

#==============================================================================
# Creating the Inverted Index:
#
# An element of invInd is: 'term':[df,{IndexDoc1:tf,IndexDoc2:tf,...}]
# Ex. 'student':[3,{10:1, 43:2, 1345:1}]'
#==============================================================================

def build_inv_index(keywords, documents):
    print("\n--- Building Inverted Index")
    invInd = {}
    for k in keywords:
        d = {}
        for key in documents.iterkeys():
            if k in documents[key]:
                d[key]=documents[key].count(k)
        invInd[k]=[len(d.keys()),d]  
    return invInd        

invInd = build_inv_index(keywords,documents)

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

def build_doc_weight_matrix(keywords,documents,invInd):
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

def build_query_vector(query,keywords,documents,invInd):
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

def compute_distance(v1,v2):
    return sum([a*b for a,b in zip(v1,v2)])
    
#==============================================================================
#  Execute Query
#==============================================================================

def query_exec(queryVector,wMatrix,k):
    results = {}
    l = len(wMatrix)
    for i in range(l):
        results[sorted(documents.keys())[i]] = compute_distance(
        queryVector,wMatrix[i]) 

    sorted_results = sorted(results.iteritems(), key=operator.itemgetter(1),
                            reverse=True)
    r = (sorted_results[:k])
    print ("-"*74)
    print ("| App{:<5}| {:<40}\t| {:<6}\t|".format("File","App Name","Score"))
    print ("-"*74)
    
    for k in r:
        print("| App{:<5}| {:<40}\t| {:.4f}\t|".format(str(k[0]),allDesc[k[0]].split('\n', 1)[0],k[1]))
    print ("-"*74)
#==============================================================================
#  Reading Input
#==============================================================================
        
wMatrix = build_doc_weight_matrix(keywords,documents,invInd)

while 1:
	query = raw_input("Insert a set of keywords (empty for exit): ")
	if not query:
		print("Quitting")
		sys.exit(0)
	try:
		k=input('Max n. of results: ')
		queryVector = build_query_vector(query,keywords,documents,invInd)
		print("")
		query_exec(queryVector, wMatrix, k)
		print("")
	except Exception,e: 
            print (str(e))
            print ("Not a number")
            print("")