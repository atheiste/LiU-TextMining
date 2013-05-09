# coding: utf-8
"""
Laboratory 1 - Assignment 5c
@author: seby912 & tompe625
"""

from __future__ import division
from math import *

#==============================================================================
# Put both functions in a text file named Geometry.py. Close the Python
# interpreter (or all of Spyder, if you prefer). Start the interpreter and load the two area
# functions from the module.
#==============================================================================
def CircleArea(radius):
    if radius > 0:
        return pi*radius*radius
    else:
        print("The radius must be positive")
        return False    
        
def RectangleArea(base,height):
        return base*height

#==============================================================================
# (d) Now define another function in your Geometry module that computes the area of a triangle. 
# Try to import the new function from the module. Why does it not work?
#==============================================================================        
def TriangleArea(base,height):
        return base*height/2
        
#Python 2.7.3 (default, Aug  1 2012, 05:14:39) 
#[GCC 4.6.3] on linux2
#Type "help", "copyright", "credits" or "license" for more information.
#>>> import Geometry as gm
#>>> gm.CircleArea(2)
#12.566370614359172
#>>> gm.RectangleArea(2,4)
#8
#>>> gm.TriangleArea(2,4)
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#AttributeError: 'module' object has no attribute 'TriangleArea'

# It doesn't work because python cannot find the new function because we 
# wrote it after the import, so we need to reload the module. 


#==============================================================================
# [Hint: try import imp followed by imp.reload(Geometry)]
#==============================================================================        

#>>> import imp
#>>> imp.reload(gm)
#<module 'Geometry' from 'Geometry.py'>
#>>> gm.TriangleArea(2,4)
#4.0
