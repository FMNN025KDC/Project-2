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
#        print("delta",delta)
        u = array(delta - dot(hessOld,gamma))
#        print(u)
        a = 1/(dot(u.T,gamma))
#        print(a)
        H = hessOld + a*dot(u,u.T)
#        print(H)
        return H
        
        
        
        
    
    