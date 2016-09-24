# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:59:40 2016
@author: bas11ksv
"""

from  scipy import *
from  pylab import *
import scipy.optimize as optimize
import time


class Optimizer():
    def __init__(self, function, fgradient = None, rho=0.1, sigma=0.7, tau=0.1, xi=9):
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.xi = xi    
        self.function = function
        
        if fgradient is None:     
            self.fgradient = Optimizer.calculateDifference(function)
        else:
            self.fgradient = fgradient
            
    @classmethod
    def calculateDifference(cls,function,Hessian = False):
        if Hessian:
            function = cls.calculateDifference(function)
        
        def differentiate(x):
            epsilon = 1e-8
            if (not isinstance(x,ndarray)) and (not isinstance(x,list)): 
                point = array([x]) +0.0
                res = array([x]) +0.0
            else:
                point = array(x) +0.0
                res = array(x) +0.0
                
            if Hessian:
                epsilon = 1e-5
                res = zeros((len(point),len(point)))
            
            for i in range(0,len(point)):
                xf = array(point)
                xf[i] = xf[i] +epsilon
                xb = array(point)
                xb[i] = xb[i] -epsilon
                res[i] = (function(xf) - function(xb)) / (2 * epsilon)

            return res
        return differentiate     




       
    def function_alpha(self,x_k,s_k):
        
        def func_alpha(alpha):
            return self.function(x_k + alpha * s_k)
            
        return func_alpha


        
    def inexactLineSearch(self,x_k,s_k):
        alpha_U = 1e99
        alpha_L = 0
        alpha_0 = 1
        
        func_alpha = self.function_alpha(x_k,s_k)
        grad_alpha = self.calculateDifference(func_alpha)

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

    def exactLineSearch(self,x_k,s_k):
        print('x_k',x_k)
        print('s_k',s_k)        
        alpha=1
        alpha_func=self.function_alpha(x_k,s_k)
        alpha=optimize.minimize(alpha_func, alpha)
        alpha=alpha.x
        return alpha
           
    def goldsteinCondition(self,alpha_0,alpha_U,alpha_L,func_alpha,grad_alpha):
        LC = (func_alpha(alpha_0) >= func_alpha(alpha_L) + \
            (1 - self.rho) * (alpha_0 - alpha_L) * grad_alpha(alpha_L))
        RC = (func_alpha(alpha_0) <= func_alpha(alpha_L) + \
            self.rho*(alpha_0 - alpha_L)*grad_alpha(alpha_L))
            
        return (LC[0],RC[0])
            
            
        
    def newton(self, x, Inexact=True, method='classic'):
        x=array(x,dtype=float)          
        g=self.fgradient(x)

        run_once=True
        
        while True:
                            
            
            gk0=g            
            
            g=self.fgradient(x) 
    
            gk1=g            
          
            Gbar=Optimizer.calculateDifference(self.function, True)             
            G=0.5*(Gbar(x)+transpose(Gbar(x)))
            
            if method=='classic':
            
                try:
                    L = cholesky(G)
                except LinAlgError:
                    print('matrix no psd')
                    break
                
                y=solve(L,-g)
                s=solve(L.conj().T,y)
                
            if method !='classic':
                if run_once==True:
                    Gbar=Optimizer.calculateDifference(self.function, True)             
                    G=0.5*(Gbar(x)+transpose(Gbar(x)))
                    H=inv(G)

                    s=-dot(H,g)
#                    print('s',s)
                    xk0=array(x)
                    print('xk0',xk0)
                    for i in range(0, len(x)):
                        x[i]=x[i]+1*s[i]
                    xk1=array(x)
                    g=self.fgradient(x)    
                    gk1=g       
                    print('xk1',xk1)                                      
                    run_once=False
                    
                gam=gk1-gk0
                print('gamma',gam)                
                delta=xk1-xk0
                print('delta',delta)                

                
                if method=='GB':               
                    H = GoodBroyden.updateHess(delta,gam,H)
                elif method=='DFP':

                    H = DFP.updateHess(delta,gam,H)
                    
                print('H',H,'g',g)
                print(shape(H),type(H))
                s=-dot(H,g)
                print('s',s)
                print('x',x)
                

            time.sleep(5)
                    
            if Inexact:
                alpha=self.inexactLineSearch(x,s)
            else:
                alpha=self.exactLineSearch(x,s)
            
            xk0=x

            for i in range(0, len(x)):
                x[i]=x[i]+alpha*s[i]
            
            xk1=x
            
            
            

            
        
            
            
            print(x,alpha)
            print('normg',norm(g))
            if norm(g) < 1e-5:
                print('x=',x)
                break
