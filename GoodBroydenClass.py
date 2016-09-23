# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:31:24 2016

@author: PeterS
"""
from scipy import *
from pylab import *


class GoodBroyden(Optimizer):
    
    @classmethod
    def updateHess(cls,delta,gamma,hessOld):
        gamma = array(gamma)[None]
        delta = array(delta)[None]
        
        
        gamma = gamma.reshape(size(gamma),1)
        delta = delta.reshape(size(delta),1)
        
        u = array(delta - hessOld*gamma)[None]
        a = 1/(u.T*gamma)
        
        H = hessOld + a*u*u.T
        
        return H
        
        
        
        
    
    