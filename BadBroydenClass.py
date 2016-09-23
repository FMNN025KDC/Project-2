# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:13 2016

@author: PeterS
"""
from scipy import *
from pylab import *


class BadBroyden(Optimizer):
    
    @classmethod
    def updateHess(cls,delta,gamma,hessOld):
        gamma = array(gamma)[None]
        delta = array(delta)[None]
        
        
        gamma = gamma.reshape(size(gamma),1)
        delta = delta.reshape(size(delta),1)
        
        H = hessOld + ((delta - hessOld*gamma)*delta.T*hessOld) \
        /(delta.T*hessOld*gamma)
        
        
        
        return H