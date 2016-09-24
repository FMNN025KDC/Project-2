# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 11:14:08 2016
@author: poff
"""
from  scipy import *
from  pylab import *



class DFP(Optimizer):
    
    @classmethod
    def updateHess(cls,delta,gam,hessOld):
#        gam = array(gam)[None]
#        delta = array(delta)[None]
#        print(hessOld)
        
#        gam = gam.reshape(size(gam),1)
#        delta = delta.reshape(size(delta),1)
        
        u = delta - dot(hessOld,gam) #1x2 
    #    print('u',u)
        a = 1/dot(u.T,gam)    #scalar
     #   print('a',a)
        ggT=outer(gam,gam.T)
        HggT=dot(hessOld,ggT)
        gTH=dot(gam.T,hessOld)
        

        H = hessOld + outer(delta,delta.T)/(dot(delta.T,gam))-dot(HggT,hessOld)/dot(gTH,gam)
        return H
