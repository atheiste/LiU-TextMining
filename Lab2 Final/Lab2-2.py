# coding: utf-8
"""
Laboratory 2 - Basic Statistics
@author: seby912 & tompe625
"""

"""--------------- 2 --------------------"""

from matplotlib import pyplot
from scipy.stats import expon, norm
from sklearn import svm

#==============================================================================
# 1.Generate two samples X1 and X2, each of size n=100, and the values in these 
# samples should be Exponential(scale=5)
#==============================================================================
X1 = expon.rvs(scale=5.0, size=100)
X2 = expon.rvs(scale=5.0, size=100)

#==============================================================================
# 4.Pack X1 and X2 into tuple X with function zip()
#==============================================================================
X = zip(X1, X2)

#==============================================================================
# 2.For each X1(i) and X2(i) Generate Y(i) value according to the following formula
#==============================================================================
fromIntToStr = lambda x: "Blue" if x == 0 else "Orange"

Y1 = map(lambda (x, y): 0 if x*y<=30+norm.rvs(0, scale=10) else 1, X)
Y = map(fromIntToStr,  Y1)

#==============================================================================
# 3. Plot the data in the coordinates X1 and X2 and specify color for the points as Y (
# use scatter()) Are the data well separated?
#==============================================================================
pyplot.subplot(1, 3, 1)
pyplot.title("Original Data") 
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.scatter(X1, X2, color=Y)

#==============================================================================
# Fit the following SVM models to the data (X,Y):
# a. Kernel=Linear
#==============================================================================
linclf = svm.LinearSVC()
linclf.fit(X, Y1)

#==============================================================================
# b. Kernel=rbf, gamma=0.7
#==============================================================================
rbfclf = svm.SVC(kernel='rbf', gamma=0.7)
rbfclf.fit(X, Y1)

#==============================================================================
# Use the fitted models to predict the Y values for all X values.
#==============================================================================
colorsLinear = map(fromIntToStr, linclf.predict(X))
colorsRBF = map(fromIntToStr, rbfclf.predict(X))

#==============================================================================
# Make the same kind of plots as in step 3 but use the fitted values to show the 
# color of the points. Conclusions?
#==============================================================================
pyplot.subplot(1, 3, 2)
pyplot.title("Linear")
pyplot.xlabel("X1") 
pyplot.scatter(X1, X2, color=colorsLinear)

pyplot.subplot(1, 3, 3)
pyplot.title("RBF")
pyplot.xlabel("X1") 
pyplot.scatter(X1, X2, color=colorsRBF)
pyplot.suptitle('Basic Statistics')

pyplot.show()
