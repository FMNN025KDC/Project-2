# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:14:59 2016
@author:  David Olsson, Kristoffer Svendsen, Claes Svensson
"""
from  scipy import *
from  pylab import *

def GBmethod(H_k, delta, gamma):
    """Updates hessian using the 'Good Broyden' method.
    
        Args:
            H_k: nparray object containing the old hessian.
            delta: nparray object of delta parameter (x_k1 - x_k).
            gamma: nparray object of gamma parameter 
                (fgradient(x_k1) - fgradient(x_k)).
    """
    u = delta - H_k.dot(gamma)
    a = 1 / (u.dot(gamma))
    
    H_k1 = H_k + a * outer(u,u)
    return H_k1
    
def BBmethod(H_k, delta, gamma):
    """Updates hessian using the 'Bad Broyden' method.
    
        Args:
            H_k: nparray object containing the old hessian.
            delta: nparray object of delta parameter (x_k1 - x_k).
            gamma: nparray object of gamma parameter 
                (fgradient(x_k1) - fgradient(x_k)).
    """
    u = delta - H_k.dot(gamma)   
    a = 1 / (delta.dot(H_k.dot(gamma)))
    
    H_k1 = H_k + a * outer(u,delta) * H_k
    return H_k1
    
def DFPmethod(H_k, delta, gamma):
    """Updates hessian using the 'DFP' method.
    
        Args:
            H_k: nparray object containing the old hessian.
            delta: nparray object of delta parameter (x_k1 - x_k).
            gamma: nparray object of gamma parameter 
                (fgradient(x_k1) - fgradient(x_k)).
    """
    H_k1 = H_k + outer(delta,delta) / (delta.dot(gamma)) - \
        outer(H_k.dot(gamma),gamma.dot(H_k)) / (gamma.dot(H_k).dot(gamma))
    return H_k1
    
def BFGSmethod(H_k, delta, gamma):
    """Updates hessian using the 'BFGS' method.
    
        Args:
            H_k: nparray object containing the old hessian.
            delta: nparray object of delta parameter (x_k1 - x_k).
            gamma: nparray object of gamma parameter 
                (fgradient(x_k1) - fgradient(x_k)).
    """
    u1 = (1 + gamma.dot(H_k).dot(gamma) / (delta.dot(gamma))) * \
        outer(delta,delta) / delta.dot(gamma)
    u2 = (outer(delta,gamma).dot(H_k) + outer(H_k.dot(gamma),delta)) \
        / delta.dot(gamma)
        
    H_k1 = H_k + u1 - u2
    return H_k1