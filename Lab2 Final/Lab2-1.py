# coding: utf-8
"""
Laboratory 2 - Basic Statistics - 1
@author: seby912 & tompe625
"""

from __future__ import division, print_function
from scipy.stats import poisson, ttest_1samp
from matplotlib import pyplot
from collections import Counter
import operator

#==============================================================================
# 1.Use module scipy.stats to generate a sample X with 50 observations that come 
# from Poisson distribution having mean value 4.
#==============================================================================
dim = 50
X=poisson.rvs(4, size=dim)

#==============================================================================
# 2.Write the code that computes an approximation to the mean value and the 
# variance by using the sample:
#==============================================================================
mean = reduce(operator.add, X) / dim
variance = reduce(lambda x, y: x+y*y, X) / dim - mean*mean

print("Variance: {0}".format(variance)) 
print("Mean: {0}".format(mean))


#==============================================================================
# 3.Check the theoretical values for the mean and the variance of this distribution 
# How close/far are those values to the ones you have computed in step 2? 
# Conclusions?
#==============================================================================

# The theoretical value for the mean and the variance should be equal to 4,
# because mean and variance are equal in the Poisson distribution.
# In this example we got 5.56 for the variance and 4.0 for the mean

#==============================================================================
# 4.Use Counter module in the collections to find out the unique values present in
# X and their frequencies
#==============================================================================
Xcount = Counter(X)
print(Xcount)

#==============================================================================
# 5.Create a barplot showing frequencies for all observed values of X. Does this 
# look like a Poisson distribution? (use bar())
#==============================================================================

# The produced plots are equal (in shape of distribution)
pyplot.subplot(1, 2, 1)
pyplot.title("Generated IIDs")
pyplot.bar(Xcount.keys(), Xcount.values())  # simulating histogram

#==============================================================================
# 6.For each Z from 0 to 13, compute a probability mass function from the Poisson 
# distribution with mean 4 and present the result as a barplot (use poisson.pmf()
# and bar()). Compare this plot with the previous one and make conclusions.
#==============================================================================
Z = range(14)
pmf = poisson.pmf(Z, 4)  # probability mass func(x, mu): exp(-mu) * mu**k / k!
pyplot.subplot(1, 2, 2) 
pyplot.title("Computed using probability mass function")
pyplot.bar(Z, pmf)

#==============================================================================
# 7.Use function mstats.ttest_1samp and the sample you have generated in step 1
# to test whether
#==============================================================================

#a. Mean value is 2.0
print(ttest_1samp(X, 2.0))

#b. Mean value is 3.7
print(ttest_1samp(X, 3.7))

#c. Mean value is 4.3
print(ttest_1samp(X, 4.3))

pyplot.show()

