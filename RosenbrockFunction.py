# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 20:50:05 2016
@author: David Olsson, Kristoffer Svendsen, Claes Svensson
"""
from  scipy import *
from  pylab import *

def Rosenbrock():
    def RBFunc(x):
        x = array(x)

        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    return RBFunc
    
def RosenbrockGradient():
    def RBGrad(x):
        x = array(x)
        
        return array([2 * (-1 + x[0] + 200 * x[0]**3 - 200 * x[0] * x[1]), 200 * (-x[0]**2 + x[1])])
    return RBGrad