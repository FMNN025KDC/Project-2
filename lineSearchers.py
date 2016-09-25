# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:05:20 2016
@author:  tfy12dol
"""
from  scipy import *
from  pylab import *
import scipy.optimize as optimize

def exactLineSearch(func_alpha):
    alpha = 1
    alpha = optimize.minimize(func_alpha, alpha, method='Nelder-Mead')
    return alpha.x   

def goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha, rho = 0.1):
    LC = (func_alpha(alpha_0) >= func_alpha(alpha_L) + \
        (1 - rho) * (alpha_0 - alpha_L) * grad_alpha(alpha_L))
    RC = (func_alpha(alpha_0) <= func_alpha(alpha_L) + \
        rho*(alpha_0 - alpha_L)*grad_alpha(alpha_L))
    return (LC,RC)
    
def wolfePowellCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha, rho = 0.1, sigma = 0.7):
    LC = (grad_alpha(alpha_0) >= sigma * grad_alpha(alpha_L))
    RC = (func_alpha(alpha_0) <= func_alpha(alpha_L) + rho * \
        (alpha_0 - alpha_L) * grad_alpha(alpha_L))
        
    return (LC,RC)    
    
def inexactLineSearch(func_alpha, grad_alpha, condition = "goldstein", rho = 0.1, sigma = 0.7, tau = 0.1, xi = 9):
    alpha_U = 1e99
    alpha_L = 0
    alpha_0 = 1
    
    if (condition == "goldstein"):
        LC,RC = goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha, rho)
    elif (condition == "wolfePowell"):
        LC,RC = wolfePowellCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha, rho, sigma)
    else:
        print("Unknown condition. Using 'goldstein'.")
        LC,RC = goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha, rho)
        condition = "goldstein"
    
    while (not (LC and RC)):
        if (not LC):
            Dalpha_0 = (alpha_0 - alpha_L) * \
            grad_alpha(alpha_0) / (grad_alpha(alpha_L) - grad_alpha(alpha_0))
            Dalpha_0 = max(Dalpha_0, tau * (alpha_0 - alpha_L))
            Dalpha_0 = min(Dalpha_0, xi * (alpha_0 - alpha_L))
            alpha_L = alpha_0
            alpha_0 = alpha_0 + Dalpha_0
        else:
            alpha_U = min(alpha_0, alpha_U)
            Balpha_0 = (alpha_0 - alpha_L)**2 * grad_alpha(alpha_0) / \
                (2 * (func_alpha(alpha_L) - func_alpha(alpha_0) + (alpha_0 - alpha_L) * grad_alpha(alpha_L)))
            Balpha_0 = max(Balpha_0, alpha_L + tau * (alpha_U - alpha_L))
            Balpha_0 = min(Balpha_0, alpha_U - tau * (alpha_U - alpha_L))
            alpha_0 = Balpha_0
        
        if (condition == "goldstein"):
            LC,RC = goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha, rho)
        else:
            LC,RC = wolfePowellCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha,rho,sigma)
        
    return alpha_0
    
 