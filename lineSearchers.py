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
    alpha = optimize.minimize(func_alpha, alpha)
    return alpha.x   

def eval_LC(f, fp, alpha_0, alpha_l, rho = 0.1):
    return f(alpha_0) >= f(alpha_l) + (1 - rho)*(alpha_0 - alpha_l)*fp(alpha_l)

def eval_RC(f, fp, alpha_0, alpha_l, rho = 0.1):
    return f(alpha_0) <= f(alpha_l) + rho*(alpha_0 - alpha_l)*fp(alpha_l)    
    
def extrapolate(fp, alpha_0, alpha_l):
    return (alpha_0 - alpha_l)*fp(alpha_0)/(fp(alpha_l) - fp(alpha_0))

def interpolate(f, fp, alpha_0, alpha_l):
    num = fp(alpha_l)*(alpha_0 - alpha_l)**2
    denom = 2*(f(alpha_l) - f(alpha_0) + (alpha_0 - alpha_l)*fp(alpha_l))
    return num/denom    
    
#def inexactLineSearch(f, fp):
#    alpha_0 = 1
#    tau = 0.1
#    chi = 9
#    alpha_l = 0
#    alpha_u = 1e99
#    LC = eval_LC(f, fp, alpha_0, alpha_l)
#    RC = eval_RC(f, fp, alpha_0, alpha_l)
#    while not (LC and RC):
#        if not LC:
#            delta_alpha_0 = extrapolate(fp, alpha_0, alpha_l)
#            delta_alpha_0 = max(delta_alpha_0, tau*(alpha_0 - alpha_l))
#            delta_alpha_0 = min(delta_alpha_0, chi*(alpha_0 - alpha_l))
#            alpha_l = alpha_0
#            alpha_0 += delta_alpha_0
#        else:
#            alpha_u = min(alpha_0, alpha_u)
#            alpha_0_bar = interpolate(f, fp , alpha_0, alpha_l)
#            alpha_0_bar = max(alpha_0_bar, alpha_l + tau*(alpha_u - alpha_l))
#            alpha_0_bar = min(alpha_0_bar, alpha_u - tau*(alpha_u - alpha_l))
#            alpha_0 = alpha_0_bar
#        LC = eval_LC(f, fp, alpha_0, alpha_l)
#        RC = eval_RC(f, fp, alpha_0, alpha_l)
#    return alpha_0    

def goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha, rho = 0.1):
    LC = (func_alpha(alpha_0) >= func_alpha(alpha_L) + \
        (1 - rho) * (alpha_0 - alpha_L) * grad_alpha(alpha_L))
    RC = (func_alpha(alpha_0) <= func_alpha(alpha_L) + \
        rho*(alpha_0 - alpha_L)*grad_alpha(alpha_L))
#    print(LC,RC)
    return (LC,RC)
    
def inexactLineSearch(func_alpha, grad_alpha, rho=0.1, sigma=0.7, tau=0.1, xi=9):
    alpha_U = 1e99
    alpha_L = 0
    alpha_0 = 1

    LC,RC = goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha, rho)
        
    while (not (LC and RC)):
        if (not LC):
#                print("LC")
            Dalpha_0 = (alpha_0 - alpha_L) * \
            grad_alpha(alpha_0) / (grad_alpha(alpha_L) - grad_alpha(alpha_0))
            Dalpha_0 = max(Dalpha_0, tau * (alpha_0 - alpha_L))
            Dalpha_0 = min(Dalpha_0, xi * (alpha_0 - alpha_L))
            alpha_L = alpha_0
            alpha_0 = alpha_0 + Dalpha_0
        else:
#                print("RC")
            alpha_U = min(alpha_0, alpha_U)
#                print("Alpha:")
#                print(alpha_0)
            Balpha_0 = (alpha_0 - alpha_L)**2 * grad_alpha(alpha_0) / \
                (2 * (func_alpha(alpha_L) - func_alpha(alpha_0) + (alpha_0 - alpha_L) * grad_alpha(alpha_L)))
            Balpha_0 = max(Balpha_0, alpha_L + tau * (alpha_U - alpha_L))
            Balpha_0 = min(Balpha_0, alpha_U - tau * (alpha_U - alpha_L))
            alpha_0 = Balpha_0
        
        LC,RC = goldsteinCondition(alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha)
        
    return alpha_0
    
 