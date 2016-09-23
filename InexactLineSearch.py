# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:47:38 2016
@author:  tfy12dol
"""
from  scipy import *
from  pylab import *

class InexactLineSearch(Optimizer):
    def __init__(self, function, fgradient = None,rho = 0.1, sigma = 0.7, tau = 0.1, xi = 9):
        super().__init__(function,fgradient)
        
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.xi = xi
#        self.alpha = 1
        self.alpha_U = 1e99
        self.alpha_L = 0
        
    def getAlpha(self,x_k,s_k):
        func_alpha = self.function_alpha(x_k,s_k)
        grad_alpha = super().calculateGradient(func_alpha)
#        grad_alpha = fgradient_alpha(x_k,s_k)
#        return grad_alpha
        
        
    
    
    
    def function_alpha(self,x_k,s_k):
        def func_alpha(alpha):
            return self.function(x_k + alpha * s_k)
        return func_alpha
        
#    def fgradient_alpha(self,x_k,s_k):
#        def grad_alpha(alpha):
#            return self.fgradient(x_k + alpha * s_k)
#        return grad_alpha
        
        
    def goldsteinCondition(self,alpha_0,func_alpha,grad_alpha):
        LC = (func_alpha(alpha_0) >= func_alpha(self.alpha_L) + \
            (1 - self.rho) * (alpha_0 - self.alpha_L) * grad_alpha(self.alpha_L))
        RC = (func_alpha(alpha_0) <= func_alpha(self.alpha_L) + \
        self.rho*(alpha_0 - alpha_L))
#        LC = (self.function(alpha_0) >= self.function(self.alpha_L) \
#            + (1 - self.rho) * (alpha_0 - self.alpha_L) * self.fgradient(alpha_L))
#        RC = (self.function(alpha_0) <= self.function(alpha_L) \
#            + self.rho * (alpha_0 - alpha_L) * self.fgradient(self.alpha_L))
            
    