# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:41:09 2016

@author: bas11ksv
"""

from  scipy import *
from  pylab import *
import scipy.optimize as optimize


class Optimizer():
    def __init__(self, function, fgradient = None):
        self.function = function
        if fgradient is None:     
            self.fgradient = Optimizer.calculateDifference(function)
        else:
            self.fgradient = fgradient
            
            
    def function_alpha(self,x_k,s_k):
        
        def func_alpha(alpha):
            return self.function(x_k + alpha * s_k)
            
        return func_alpha

            
            
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
#                print(array(point))
                xf = array(point)
                xf[i] = xf[i] +epsilon
                xb = array(point)
                xb[i] = xb[i] -epsilon
                res[i] = (function(xf) - function(xb)) / (2 * epsilon)

            return res
        return differentiate
    
    
#    @classmethod
#    def calculateHessian(cls, function,epsilon=1e-8):
        
        
        

                

    def newton(self, f, x,dx=0.00001):
        x=array(x,dtype=float)
        alpha=1
        
    
        g=self.fgradient(x)  
        
        
        while True:                
            
            gk0=g            
            
            g=self.fgradient(x) 

            gk1=g              
            
#            H11=(f([x[0]+dx,x[1]])+f([x[0]-dx, x[1]])-2*f([x[0],x[1]]))/(dx**2)   
#            H22=(f([x[0],x[1]+dx])+f([x[0], x[1]-dx])-2*f([x[0],x[1]]))/(dx**2)
#            H12=(f([x[0]+dx,x[1]+dx])-f([x[0]+dx,x[1]-dx])-f([x[0]-dx,x[1]+dx])+f([x[0]-dx,x[1]-dx]))/(4*dx**2)
            
#            Bbar=array([[H11, H12], [H12, H22]])
            
            Gbar=Optimizer.calculateDifference(self.function, True) 
            
            G=0.5*(Gbar(x)+transpose(Gbar(x)))
            
            print(G)

            try:
                L = cholesky(G)
            except LinAlgError:
                print('matrix no psd')

            y=solve(L,-g)
            s=solve(L.conj().T,y)
            
            alpha_func=self.function_alpha(x,s)
            alpha=optimize.minimize(alpha_func, 1)
            alpha=alpha.x
            
            xk0=x
            
            x[0]=x[0]+alpha*s[0]
            x[1]=x[1]+alpha*s[1]
            
            xk1=x

#
#            gamma=gk1-gk0
#            delta=xk1-xk0
#            u=delta-H*gamma
#            a=1/(u.T*gamma)
#            
#            H=H+a*u*u.T
        
            print('normg',norm(g))
            if norm(g) < 1e-9:
                
                print('x=',x)
                break


def Rosenbrock():
    def RBFunc(x):
        x = array(x)
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    return RBFunc

f=Rosenbrock()    
testopt=Optimizer(f)
x0=array([4,-4])
G=testopt.newton(f,x0)

