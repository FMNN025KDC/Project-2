# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 14:33:31 2016
@author: poff
"""
from  scipy import *
from  pylab import *


class BFGS(Optimizer):
    
    @classmethod
    def updateHess(cls,delta,gam,hessOld):
        dd=outer(delta,delta.T) #matrix
        dTg=dot(delta.T,gam) #scalar
        ddT=outer(delta,delta.T)
        dgT=outer(delta,gam.T)
        gdT=outer(gam,delta.T)
        gTHg=dot(gam.T,dot(hessOld,gam))
        
        term2=dot((1+gTHg/dTg),ddT/dTg)
        term3=(dot(dgT,hessOld)+dot(hessOld,gdT))/(dTg)       
        
        H=hessOld+term2-term3
 
        return H
