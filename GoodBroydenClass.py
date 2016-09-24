
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:31:24 2016
@author: PeterS
"""
from scipy import *
from pylab import *


class GoodBroyden(Optimizer):
    
    @classmethod
    def updateHess(cls,delta,gam,hessOld):
#        gam = array(gam)[None]
#        delta = array(delta)[None]
        print(hessOld)
        
#        gam = gam.reshape(size(gam),1)
#        delta = delta.reshape(size(delta),1)
        
        u = delta - dot(hessOld,gam) #1x2 
        print('u',u)
        a = 1/dot(u.T,gam)    #scalar
        print('a',a)
        H = hessOld + a*outer(u,u.T)
        return H
