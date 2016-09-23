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
            self.fgradient = Optimizer.calculateGradient(function)
        else:
            self.fgradient = fgradient
            
    @classmethod
    def calculateGradient(cls,function,epsilon = 1e-8):
        def grad(x):
            point = array(x) +0.0
            res = array(x) +0.0
            
            for i in range(0,len(point)):
                xf = array(point)
                xf[i] = xf[i] +epsilon
                xb = array(point)
                xb[i] = xb[i] -epsilon
                res[i] = (function(xf) - function(xb)) / (2 * epsilon)

            return res
        return grad
                