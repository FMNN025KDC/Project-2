# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 20:50:05 2016
@author: David
"""
from  scipy import *
from  pylab import *

def Rosenbrock():
    def RBFunc(x):
        x = array(x)
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    return RBFunc