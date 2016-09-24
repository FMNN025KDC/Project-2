# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:24:45 2016
@author:  tfy12dol
"""
from  scipy import *
from  pylab import *

def getGradient(function):
    def grad(x):
       return evaluateGradient(function,x)         
    return grad
    
def evaluateGradient(function,x,epsilon = 1e-5):
    h = zeros(shape(x))
    res = zeros(shape(x))    
    for i in range(0,len(x)):
        h[i] = epsilon
        res[i] = (function(x + h) - function(x - h)) / (2 * epsilon)
        h[i] = 0.0
    return res
    
def getHessian(fgradient):
    def hess(x):
        return evaluateHessian(fgradient,x)
    return hess
    
def evaluateHessian(fgradient,x,epsilon = 1e-5):
    h = zeros(shape(x))
    res = zeros((len(x),len(x)))
    for i in range(0,len(x)):
        def fgradienti(x):
            return fgradient(x)[i]
        row = evaluateGradient(fgradienti,x)
        res[i,:] = row
    return res
    
    
    
    
    
    
    
    
#    for i in range(0,len(point)):
##                print(array(point))
#                xf = array(point)
#                xf[i] = xf[i] +epsilon
#                xb = array(point)
#                xb[i] = xb[i] -epsilon
#                res[i] = (function(xf) - function(xb)) / (2 * epsilon)