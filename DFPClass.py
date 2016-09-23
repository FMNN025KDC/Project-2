# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:27:08 2016

@author: PeterS
"""
from scipy import *
from pylab import *
class DFP(Optimizer):
    
    @classmethod
    def updateHess(cls,delta,gamma,hessOld):
        gamma = array(gamma)[None]
        delta = array(delta)[None]
        
        
        gamma = gamma.reshape(size(gamma),1)
        delta = delta.reshape(size(delta),1)
        
        H = hessOld + (delta*delta.T)/(delta.T*) \
        -(hessOld*gamma*gamma.T*hessOld)/(gamma.T*hessOld*gamma)
        
        return H