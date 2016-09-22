# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:41:09 2016

@author: bas11ksv
"""

from  scipy import *
from  pylab import *

class Optimizer():
    def __init__(self, function, fgradient = None):
        self.function = function
        if fgradient is None:     
            self.fgradient = Optimizer.calculateGradient(function)
        else:
            self.fgradient = fgradient
            
            
            
    @classmethod
    def calculateGradient(cls,function,epsilon = 1e-8):
        def grad(x):
            point = array(x) +0.0
            res = array(x) +0.0
            
            for i in range(0,len(point)):
                xf = array(point)
                xf[i] = xf[i] +epsilon
                xb = array(point)
                xb[i] = xb[i] -epsilon
                res[i] = (function(xf) - function(xb)) / (2 * epsilon)

            return res
        return grad



    def newton(self, f, x,dx=0.00001):

        alpha=1
            
        while True:                
#            gx1=(f(x1+dx,x2)-f(x1-dx, x2))/(2*dx)   
#            gx2=(f(x1,x2+dx)-f(x1, x2-dx))/(2*dx)
#            g=array([gx1,gx2])
            g=self.fgradient(x)       
            H11=(f(x[0]+dx,x[1])+f(x[0]-dx, x[1])-2*f(x[0],x[1]))/(dx**2)   
           
           
            H22=(f(x[0],x[1]+dx)+f(x[0], x[1]-dx)-2*f(x[0],x[1]))/(dx**2)
            
            
            H12=(f(x[0]+dx,x[1]+dx)-f(x[0]+dx,x[1]-dx)-f(x[0]-dx,x[1]+dx)+f(x[0]-dx,x[1]-dx))/(4*dx**2)
            
            Bbar=array([[H11, H12], [H12, H22]])
    
            B=0.5*(Bbar+transpose(Bbar))
     
    #            print(B,x1,x2)
            try:
                L = cholesky(B)
            except LinAlgError:
                print('matrix no psd')
            y=solve(L,-g)
            s=solve(L.conj().T,y)
            
 
                
               
            print(alpha)
    #            print("s=",s)
    #            G=inv(B)
    #            s=-dot(G,g)
            
     #           print(shape(G))
     #           print(shape(s))
    #            print(shape(g))
    
            x[0]=x[0]+alpha*s[0]
            x[1]=x[1]+alpha*s[1]
    
            print('f=',f(x1,x2))
            '''break criterion here...?'''
    #            i+=1
    #            j+=1
    
            if norm(g)< 0.000001:
                print(x1,x2)
                break

def Rosenbrock():
    def RBFunc(x):
        x = array(x)
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    return RBFunc

f=Rosenbrock()    
testopt=Optimizer(f)
x0=array([-1,1])
G=testopt.newton(f,x0)
