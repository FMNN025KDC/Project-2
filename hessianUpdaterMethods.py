# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:14:59 2016
@author:  tfy12dol
"""
from  scipy import *
from  pylab import *

def GBmethod(H_k, delta, gamma):
    u = delta - H_k.dot(gamma)
    a = 1 / (u.dot(gamma))
    
    H_k1 = H_k + a * outer(u,u)
    return H_k1
    
def BBmethod(H_k, delta, gamma):
    u = delta - H_k.dot(gamma)   
    a = 1 / (delta.dot(H_k.dot(gamma)))
    
    H_k1 = H_k + a * outer(u,delta) * H_k
    return H_k1
    
def DFPmethod(H_k, delta, gamma):
    H_k1 = H_k + outer(delta,delta) / (delta.dot(gamma)) - \
        outer(H_k.dot(gamma),gamma.dot(H_k)) / (gamma.dot(H_k).dot(gamma))
    return H_k1
    
def BFGSmethod(H_k, delta, gamma):
    u1 = (1 + gamma.dot(H_k).dot(gamma) / (delta.dot(gamma))) * \
        outer(delta,delta) / delta.dot(gamma)
    u2 = (outer(delta,gamma).dot(H_k) + outer(H_k.dot(gamma),delta)) \
        / delta.dot(gamma)
        
    H_k1 = H_k + u1 - u2
    return H_k1