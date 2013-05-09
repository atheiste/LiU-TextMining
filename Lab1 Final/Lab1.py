# coding: utf-8
"""
Laboratory 1 - Python
@author: seby912 & tompe625
"""

"""--------------- 1 --------------------"""

#==============================================================================
# (a) Define the variable parrot containing the sentence "It is dead, that is what is wrong with it".
#==============================================================================
parrot = "it is dead, that is what is wrong with it"

#==============================================================================
# (b) Count the number of characters (letters, blank space, commas, periods etc) in the sentence.
#==============================================================================
count = len(parrot) 
print(count)

#==============================================================================
# (c) Write code that counts the number of letters in the sentence.
#==============================================================================
count = len(filter(lambda c: c.isalpha(), parrot)) 
print(count)

#==============================================================================
# (d) Separate the sentence into a list of words. Call the list ParrotWords
#==============================================================================
ParrotWords = parrot.split()
print(ParrotWords) 

#==============================================================================
# (e) Merge (concatenate) ParrotWords into a sentence again.
#==============================================================================
print("".join(ParrotWords))

"""---------------- 2 -------------------"""

#==============================================================================
# (a) Write a for loop that produces the following output on the screen:
# The next number in the loop is 5
# The next number in the loop is 6
# ...
# The next number in the loop is 10
# [Hint: the range() function has more than one argument].
#==============================================================================
for a in range(5,11):
    print("The next number in the loop is %d" % (a))

#==============================================================================
# (b) Write a while-loop that repeatedly generates a random number from a uniform 
# distribution over the interval [0;1], and prints the sentence 'The random number 
# is smaller than 0.9' on the screen until the generated random number is smaller than
# 0.9. [Hint: Python has a random module with basic random number generators].   
#==============================================================================
import random
while random.random()<0.9:
    print ("The random number is smaller than 0.9")

#==============================================================================
# (c) Write a for-loop that iterates over the list
# names = ['Ludwig','Rosa','Mona','Amadeus']
# and writes the following to the screen:
# The name Ludwig is nice
# The name Rosa is nice
# ...
# The name Amadeus is nice
# Use Python's string formatting capabilities (the %s stuff ...) to solve the problem.
#==============================================================================
names = ['Ludwig','Rosa','Mona','Amadeus']
for name in names:
        print("The name %s is nice" % (name))

#==============================================================================
# (d) Write a for-loop that iterates over the list names = ['Ludwig','Rosa','Mona','Amadeus']
# and produces the list nLetters = [6,4,4,7] that counts the letters in each name.
# [Hint: the pretty version uses the enumerate() function]
#==============================================================================
nLetters=["","","",""]
for ind, itm in enumerate(names):
    nLetters[ind] = (len(itm))        
print(nLetters)

#==============================================================================
#  other possible solutions
#==============================================================================
nLetters=[]
for itm in names:
    nLetters.append(len(itm))
print(nLetters)

nLetters=[]
for i in range(0,len(names)):
    nLetters.append(len(names[i]))        
print(nLetters)

#==============================================================================
# (e) Solve the previous question using a list comprehension.
#==============================================================================
nList = [len(name) for name in names]    
print(nList) 

#==============================================================================
# (f) Use a list comprehension to produce a list that indicates if the name has 
# more than four letters. The answer should be shortLong = ['long','short','short','long'].
#==============================================================================
def lunghezza(name):
    if len(name)>4: 
        name = "long" 
    else: 
        name = "short"
    return name

shortLong = [lunghezza(name) for name in names]    
print(shortLong)

#==============================================================================
# (g) Write a loop that simultaneously loops over the lists names and
# shortLong to write the
# following to the screen
# The name Ludwig is a long name
# The name Rosa is a short name
# ...
# The next Amadeus is a long name
# [Hint: use the zip() function and Python's string formatting.]       
#==============================================================================
for x in zip(names,shortLong):
    print ("The name %s is a %s name" % (x[0],x[1]))

"""---------------- 3 ----------------------"""

#==============================================================================
# (a) Make a dictionary named Amadeus containing the information that the student Amadeus
# is a male (M), scored 8 on the Algebra exam and 13 on the History exam.
#==============================================================================
Amadeus = {'Sex':'M','Algebra':8,'History':13}

#==============================================================================
# (b) Make three more dictionaries, one for each of the students: Rosa, Mona and Ludwig,
# from the information in the following table:
#         Sex     Algebra     History
# Rosa    F           19      22
# Mona    F           6       27
# Ludwig  M           9       5
#==============================================================================
Rosa = {'Sex':'F','Algebra':19,'History':22}
Mona = {'Sex':'F','Algebra':6,'History':27}
Ludwig = {'Sex':'M','Algebra':9,'History':5}

#==============================================================================
# (c) Combine the four students in a dictionary named students such that a user of your
# dictionary can type students['Amadeus']['History'] to retrive Amadeus score on the
# history test. [HINT: The values in a dictionary can be dictionaries]
#==============================================================================
students = {'Amadeus':Amadeus,'Rosa':Rosa,'Mona':Mona,'Ludwig':Ludwig}
print(students['Amadeus']['History'])

#==============================================================================
# (d) Add the new student Karl to the dictionary students. Karl scored 14 on the Algebra
# exam and 10 on the History exam.
#==============================================================================
students['Karl']={'Sex':'M','Algebra':14,'History':10}
print(students['Karl']['History'])

#==============================================================================
# (e) Use for-loop to print out the names and scores of all students on the screen. The output
# should look like something this (the order of the students doesn't matter):
# Student Amadeus scored 8 on the Algebra exam and 13 on the History exam
# Student Rosa scored 19 on the Algebra exam and 22 on the History exam
# ...
# [Hints: Dictionaries are iterables. A really pretty solution involves the .items()
# method of a dictionary]
#==============================================================================
for student in students.items():
    print("Student %s scored %d on the Algebra exam and %d on the History exam" % (student[0],student[1].get('Algebra'),student[1].get('History')))
    
"""----------------- 4 ---------------------"""

#==============================================================================
# (a) Define two lists: list1 = [1,3,4] and list2 = [5,6,9]. 
# Try list1*list2. Does it work?
#==============================================================================
list1 = [1,3,4]
list2 = [5,6,9]
#list1*list2 is not working because Python can't multiply sequence by non-int of type 'list'

#==============================================================================
# (b) Import everything from scipy (from scipy import *). Convert list1 and
# list2 into arrays (name them array1 and array2). Now try array1*array2.
#==============================================================================
from scipy import *
array1 = array(list1)
array2 = array(list2)
print(array1 * array2)

#==============================================================================
# (c) Let matrix1 be a 2-by-3 array with array1 and array2 as its two rows. Let
# matrix2 be a diagonal matrix with elements 1, 2 and 3. Try matrix1*matrix2. 
# Why doesn't this work?
#==============================================================================
matrix1 = array([list1,list2])
print (matrix1)
print (type(matrix1))

matrix2 = array([[1,0,0],[0,2,0],[0,0,3]])
print (type(matrix2))
print (matrix2)
#matrix1*matrix2 "ValueError: operands could not be broadcast together with shapes (2,3) (3,3)"

#==============================================================================
# (d) Compute the usual matrix product of matrix1 and matrix2.
#==============================================================================
print dot(matrix1,matrix2)

"""----------------- 5 ---------------------"""

#==============================================================================
# (a) Write a function CircleArea(radius) that computes the area of a circle with radius
# radius. Call the function to show that it works. [Hint: the number pi needs to be loaded
# from the math module]
#==============================================================================
from math import *

def CircleArea(radius):
        return pi*radius*radius    
print(CircleArea(3))    


#==============================================================================
# (b) Modify the CircleArea function so that it checks it the radius is positive and prints
# 'The radius must be positive' to the screen if it is not. Also, if the radius is not positive the
# function should return None.
#==============================================================================
def CircleArea(radius):
    if radius > 0:
        return pi*radius*radius
    else:
        print("The radius must be positive")
        return None    
        
print(CircleArea(-3))    

#==============================================================================
# (c) Now write another function RectangleArea(base,height) that computes the area of
# a rectangle. 
#==============================================================================
def RectangleArea(base,height):
        return base*height

#==============================================================================
# Put both functions in a text file named Geometry.py. Close the Python
# interpreter (or all of Spyder, if you prefer). Start the interpreter and load the two area
# functions from the module.
#==============================================================================




    


