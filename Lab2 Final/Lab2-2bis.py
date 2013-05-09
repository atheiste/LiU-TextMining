# coding: utf-8
"""
Laboratory 2 - Basic Statistics
@author: seby912 & tompe625
"""

"""--------------- 2.1 --------------------"""

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
Y = map(lambda (x, y): 'Blue' if x*y<=30+norm.rvs(0, scale=10) else 'Orange', X)


#==============================================================================
# 3. Plot the data in the coordinates X1 and X2 and specify color for the points as Y (
# use scatter()) Are the data well separated?
#==============================================================================
pyplot.subplot(1, 3, 1)
pyplot.scatter(X1, X2, color=Y)

#==============================================================================
# Fit the following SVM models to the data (X,Y):
# a. Kernel=Linear
#==============================================================================
linclf = svm.LinearSVC()
linclf.fit(X, Y)

#==============================================================================
# b. Kernel=rbf, gamma=0.7
#==============================================================================
rbfclf = svm.SVC(kernel='rbf', gamma=0.7)
rbfclf.fit(X, Y)

#==============================================================================
# Use the fitted models to predict the Y values for all X values.
#==============================================================================
colorsLinear = linclf.predict(X)
colorsRBF = rbfclf.predict(X)

#==============================================================================
# Make the same kind of plots as in step 3 but use the fitted values to show the 
# color of the points. Conclusions?
#==============================================================================
pyplot.subplot(1, 3, 2)
pyplot.scatter(X1, X2, color=colorsLinear)
pyplot.subplot(1, 3, 3)
pyplot.scatter(X1, X2, color=colorsRBF)

pyplot.show()