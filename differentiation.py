# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:24:45 2016
@author:  David Olsson, Kristoffer Svendsen, Claes Svensson
"""
from  scipy import *
from  pylab import *

def getGradient(function):
    """Returns the gradient of a funciton as a function object. Uses finite 
        differences approximation.
    
        Args:
            function: function object to find the gradient of.
            
        Returns:
            function object of the gradient.
    """
    def grad(x):
       return evaluateGradient(function,x)         
    return grad
    
def evaluateGradient(function,x,epsilon = 1e-5):
    """Evaluates the gradient of function in point x using finite difference 
        approximation.
        
        Args:
            function: function object to find the gradient of.
            x: nparray object contining point to evaluate in.
            epsilon: {Optional} {Default = 1e-5} Finite difference step size.
            
        Returns:
            nparray object containing the value of the gradient in point x.
    """
    h = zeros(shape(x))
    res = zeros(shape(x))    
    for i in range(0,len(x)):
        # Set the step on the correct variable.
        h[i] = epsilon
        # Approximate derivative using central difference approximation.
        res[i] = (function(x + h) - function(x - h)) / (2 * epsilon)
        # Reset step for next iteration.
        h[i] = 0.0
    return res
    
def getHessian(fgradient):
    """Returns the hessian of a funciton as a function object. Uses finite 
        differences approximation.
    
        Args:
            fgradient: function object of gradient to find the hessian of.
            
        Returns:
            function object of the hessian.
    """
    def hess(x):
        return evaluateHessian(fgradient,x)
    return hess
    
def evaluateHessian(fgradient,x,epsilon = 1e-5):
    """Evaluates the hessian of function in point x using finite difference 
        approximation.
        
        Args:
            fgradient: function object of gradient to find the hessian of.
            x: nparray object contining point to evaluate in.
            epsilon: {Optional} {Default = 1e-5} Finite difference step size.
            
        Returns:
            nparray object containing the value of the hessian in point x.
    """
    h = zeros(shape(x))
    res = zeros((len(x),len(x)))
    for i in range(0,len(x)):
        # Define new gradient function which returns only the i:th element of 
        # the gradient in a point x.
        def fgradienti(x):
            return fgradient(x)[i]
        # Evaluate new funciton object and store the result as a row in the 
        # hessian.
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