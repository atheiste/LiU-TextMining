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
import nltk, re, pprint, os, math, operator, time, threading, urllib2
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

def openWebPage(url):
    opener = urllib2.build_opener()
    # without the following line i dont get any response
    # with, i get some HTTP 500 
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    response = opener.open(url)
    page = response.read()
    opener.close()  
    return page
    
#==============================================================================
# Searching all the URLs from a generic pages leading to an App
#==============================================================================
    
def getAppUrls(url):        
    urls=[]
    for i in range(30): #black magic:
        page = openWebPage(url)
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
def getAppDescr(urls,i):
    descrs = []
    for url in urls:
        for j in range(30): #more black magic
            page = openWebPage(url)
            m = "".join(patDescr.findall(page))
                    
            print("[{}-{}] ".format(j,i+len(descrs)+1)+url)        
            if len(m) > 0:
                break
            if j > 10:
                w = "".join(patAdult.findall(page))
                if w > 0:
                    break
            time.sleep(5)
            
        
        
        m = patClean.sub(" ", m)
        descrs.append(("".join(patTitle.findall(page)) + "\n\n" + m))   
    
    l = len(descrs)        
    for k in range(l):
        filename = "App{0}.txt".format(i+k)
        f = open(join(ROOT, filename), "w")
        f.write(descrs[k])
        f.close()
    return descrs    
    

#==============================================================================
# Loading Data
#==============================================================================

class appName3d (threading.Thread):
    urls = []
    def __init__(self, url):
        threading.Thread.__init__(self)
        self.url = url
    def run(self):
        self.urls = getAppUrls(self.url)

class appDescr3d (threading.Thread):
    descr = ""
    def __init__(self, urls, j):
        threading.Thread.__init__(self)
        self.urls = urls
        self.j = j
    def run(self):
        self.descrs = getAppDescr(self.urls,self.j)        

if not os.path.exists(ROOT) :
    
    print("Creating folder")
    os.mkdir(ROOT)

allDesc = {}

if len(os.listdir(ROOT))==0:
    
    print("Downloading descriptions\n--- Getting URLs\n") 
    
    appUrls = []
    thrds = []

    for s in categories:
        for i in range(0,200,10):
            url = "http://www.appbrain.com/apps/country-united-states/"+s+"/?o="+str(i);
            thread = appName3d(url)
            thrds.append(thread)
            
            
    for t in thrds:
        t.start()
        
    for t in thrds:
        t.join()             
            
    for t in thrds:
        appUrls.extend(t.urls)
    print("\n\n{} URLs collected\n--- Getting descriptions".format(len(appUrls)))                    
            
    thrds = []
    
    l = len(appUrls);
    for i in range(0,l,10):
        thread = appDescr3d(appUrls[i:i+10],i)
        thrds.append(thread)
    
    for t in thrds:
        t.start()
        
    for t in thrds:
        t.join()            
    
    l1 = len(thrds)
    for i in range(l):
        l2 = len(thrds[i].descrs)
        for j in range(l2):
            allDesc[i*10+j]=thrds[i].descrs[j]
        
    print("Finished collecting Descrs...{}/{}".format(len(allDesc.keys()),len(appUrls)))
                
        
    print("")
    print("Descriptions loaded")

else:
    print("The descriptions have already been downloaded\n--- Loading files")
    
    for filename in os.listdir(ROOT):
        if filename.startswith("App"):
            f = open(join(ROOT, filename), "r")
            i = re.match("App(\d+).txt", filename)
            allDesc[int(i.group(1))] = f.read()
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
    query = raw_input("Insert a set of keywords: ")
    try:
        k=int(raw_input('Max n. of results:'))
        queryVector = buildQueryVector(query,keywords,documents,invInd)
        print("")        
        queryExec(queryVector, wMatrix, k) 
        print("")
    except ValueError:
        print ("Not a number")
        print("")    