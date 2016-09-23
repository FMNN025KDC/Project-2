# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:19:40 2016
@author:  tfy12dol
"""
from  scipy import *
from  pylab import *

class Optimizer():
    def __init__(self, function, fgradient = None):
        self.function = function
        if fgradient is None:     
            self.fgradient = Optimizer.calculateDifference(function)
        else:
            self.fgradient = fgradient
            
    @classmethod
    def calculateDifference(cls,function,Hessian = False):
        if Hessian:
            function = cls.calculateDifference(function)
        
        def differentiate(x):
            epsilon = 1e-8
            if (not isinstance(x,ndarray)) and (not isinstance(x,list)): 
                point = array([x]) +0.0
                res = array([x]) +0.0
            else:
                point = array(x) +0.0
                res = array(x) +0.0
                
            if Hessian:
                epsilon = 1e-5
                res = zeros((len(point),len(point)))
            
            for i in range(0,len(point)):
#                print(array(point))
                xf = array(point)
                xf[i] = xf[i] +epsilon
                xb = array(point)
                xb[i] = xb[i] -epsilon
                res[i] = (function(xf) - function(xb)) / (2 * epsilon)

            return res
        return differentiate
        
#    @classmethod
#    def calculateHessian(cls,gfunction,epsilon = 1e-5):
#        def hess(x):
#            if (not isinstance(x,ndarray)) and (not isinstance(x,list)): 
#                point = array([x]) +0.0
#                res = array([x]) +0.0
#            else:
#                point = array(x) +0.0
#                res = array(x) +0.0
#            
#            res = array([[0.,0.],[0.,0.]])
#            
#            for i in range(0,len(point)):
##                print(array(point))
#                xf = array(point)
#                xf[i] = xf[i] +epsilon
#                xb = array(point)
#                xb[i] = xb[i] -epsilon
#                res[i] = (gfunction(xf) - gfunction(xb)) / (2 * epsilon)
#
#            return res
#        return hess
                
                