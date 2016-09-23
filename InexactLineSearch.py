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
#        self.alpha_U = 1e99
#        self.alpha_L = 0
        
    def getAlpha(self,x_k,s_k):
        alpha_U = 1e99
        alpha_L = 0
        alpha_0 = 1
        
        func_alpha = self.function_alpha(x_k,s_k)
        grad_alpha = super().calculateGradient(func_alpha)
#        grad_alpha = fgradient_alpha(x_k,s_k)
#        return grad_alpha
        LC,RC = self.goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha)
        
        while (not (LC and RC)):
            if (not LC):
#                print("LC")
                Dalpha_0 = (alpha_0 - alpha_L) * \
                    grad_alpha(alpha_0) / (grad_alpha(alpha_L) - grad_alpha(alpha_0))
                Dalpha_0 = max(Dalpha_0[0], self.tau * (alpha_0 - alpha_L))
                Dalpha_0 = min(Dalpha_0, self.xi * (alpha_0 - alpha_L))
                alpha_L = alpha_0
                alpha_0 = alpha_0 + Dalpha_0
            else:
#                print("RC")
                alpha_U = min(alpha_0, alpha_U)
#                print("Alpha:")
#                print(alpha_0)
                Balpha_0 = (alpha_0 - alpha_L)**2 * grad_alpha(alpha_0) / \
                    (2 * (func_alpha(alpha_L) - func_alpha(alpha_0) + (alpha_0 - alpha_L) * grad_alpha(alpha_L)))
                Balpha_0 = max(Balpha_0[0], alpha_L + self.tau * (alpha_U - alpha_L))
                Balpha_0 = min(Balpha_0, alpha_U - self.tau * (alpha_U - alpha_L))
                alpha_0 = Balpha_0
            
            LC,RC = self.goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha)
            
            
            
        return alpha_0
            
    
    
    
    def function_alpha(self,x_k,s_k):
        def func_alpha(alpha):
            return self.function(x_k + alpha * s_k)
        return func_alpha
        
#    def fgradient_alpha(self,x_k,s_k):
#        def grad_alpha(alpha):
#            return self.fgradient(x_k + alpha * s_k)
#        return grad_alpha
        
        
    def goldsteinCondition(self,alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha):
        LC = (func_alpha(alpha_0) >= func_alpha(alpha_L) + \
            (1 - self.rho) * (alpha_0 - alpha_L) * grad_alpha(alpha_L))
        RC = (func_alpha(alpha_0) <= func_alpha(alpha_L) + \
            self.rho*(alpha_0 - alpha_L)*grad_alpha(alpha_L))
        
        return (LC[0],RC[0])
#        LC = (func_alpha(alpha_0) >= func_alpha(self.alpha_L) + \
#            (1 - self.rho) * (alpha_0 - self.alpha_L) * grad_alpha(self.alpha_L))
#        RC = (func_alpha(alpha_0) <= func_alpha(self.alpha_L) + \
#        self.rho*(alpha_0 - alpha_L))
#        LC = (self.function(alpha_0) >= self.function(self.alpha_L) \
#            + (1 - self.rho) * (alpha_0 - self.alpha_L) * self.fgradient(alpha_L))
#        RC = (self.function(alpha_0) <= self.function(alpha_L) \
#            + self.rho * (alpha_0 - alpha_L) * self.fgradient(self.alpha_L))
            
    