# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 21:03:17 2016
@author: David
"""
from  scipy import *
from  pylab import *

def NumericGrad(func,h=0.00001):
    def NumGrad(x):
        x = array(x)
        ret = zeros(shape(x))
        for i in range(0,len(x)):
            pDx = array(x)
            pDx[i] = pDx[i] + h
            nDx = array(x)
            nDx[i] = nDx[i] - h
#            print(nDx)
#            print(pDx)
#            print(i)
            ret[i] = (func(pDx) - func(nDx)) / (2 * h)
#            print(Dy)
        return ret
        
    return NumGrad
        