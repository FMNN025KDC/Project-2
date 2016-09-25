# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:19:33 2016
@author:  tfy12dol
"""
from  scipy import *
from  pylab import *
import scipy.linalg as cho
import differentiation
import lineSearchers
import hessianUpdaterMethods

class OptimizationProblem():
    def __init__(self, function, fgradient = None):
        self.function = function
        if fgradient is None:        
            self.fgradient = differentiation.getGradient(function)
        else:
            self.fgradient = fgradient
            
        
class OptimizationMethod():
    def __init__(self, problem, condition = "goldstein", rho = 0.1, sigma = 0.7, tau = 0.1, xi = 9):
        self.function = problem.function
        self.fgradient = problem.fgradient
        
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.xi = xi
        self.condition = condition
    
    def newton(self, x_0, lineSearchMethod = "exact", tol = 1e-3):
        x_k = x_0
        G_k = self.initialHessian(x_0)
        lineSearcher = self.getLineSearcher(lineSearchMethod)
#        xp = [] 
#        yp = []
        
        while True:
            s_k = self.calculateDirection(x_k, G_k)
            alpha_k = lineSearcher(self.function_alpha(x_k, s_k), self.fgradient_alpha(x_k, s_k), self.condition, self.rho, self.sigma, self.tau, self.xi)
            x_k1 = x_k + alpha_k * s_k
            print("-----------------------------------")
            print("alpha:",alpha_k,"x_k:",x_k,"x_k1:",x_k1,"s_k:",s_k)
#            xp.append(x_k[0])
#            yp.append(x_k[1])
            G_k = self.updateHessian(x_k, x_k1, G_k)
            x_k = x_k1
            print("G_k, or H_k for QuasiNewton:")
            print(G_k)
            if (norm(self.fgradient(x_k)) < tol):
                break
        
        print("-----------------------------------")
#        plot(xp,yp,'--o')
        print("Found point: ",x_k)
        print("Function value: ",self.function(x_k))
        return x_k,self.function(x_k)
      
    def function_alpha(self, x_k, s_k): 
        def func_alpha(alpha):
            return self.function(x_k + alpha * s_k)
        return func_alpha
        
    def fgradient_alpha(self, x_k, s_k):
        def fgrad_alpha(alpha):
            return self.fgradient(x_k + alpha * s_k).dot(s_k)
        return fgrad_alpha
    
    def getLineSearcher(self,lineSearchMethod):
        if lineSearchMethod == "exact":
            def lineSearcher(func_alpha, fgrad_alpha, *args):
                return lineSearchers.exactLineSearch(func_alpha)
            return lineSearcher
        if lineSearchMethod == "inexact":
            def lineSearcher(func_alpha, fgrad_alpha, condition, rho, sigma, tau, xi):
                return lineSearchers.inexactLineSearch(func_alpha, fgrad_alpha, condition, rho, sigma, tau, xi)
            return lineSearcher
        if lineSearchMethod == "None":
            def lineSearcher(*args):
                return 1
            return lineSearcher
               
    def initialHessian(self, x_0):
        """Will be shadowed in sub class"""        
        return None
               
    def calculateDirection(self, x_k, G_k):
        """Will be shadowed in sub class"""        
        return None      
        
    def updateHessian(self, x_k, x_k1, G_k):
        """Will be shadowed in sub class"""
        return None
        
        
        
class ClassicalNewton(OptimizationMethod):
    
    def initialHessian(self, x_0):
        return differentiation.evaluateHessian(self.fgradient, x_0)    
    
    def calculateDirection(self, x_k, G_k):
#        G_k = differentiation.evaluateHessian(self.fgradient, x_k)
        G_k = 0.5 * (G_k + G_k.T)
        
        while True:
            try:
                L = cho.cho_factor(G_k)
                break
            except cho.LinAlgError:
#                raise ValueError("Cholesky factorization failed. Possibly non PSD.")
                G_k += identity(len(x_k))
                
        
        s_k = -cho.cho_solve(L, self.fgradient(x_k))
        return s_k
        
    def updateHessian(self, x_k, x_k1, G_k):
        return differentiation.evaluateHessian(self.fgradient, x_k1)
        
    

class QuasiNewton(OptimizationMethod):
    def __init__(self, problem, method = "GB", condition = "goldstein",rho = 0.1, sigma = 0.7, tau = 0.1, xi = 9):
        super().__init__(problem, condition, rho, sigma, tau, xi)
        
        acceptedMethods=['DFP','GB','BFGS','BB']        
        if not method in acceptedMethods:            
            print('Not acceptable method.')
            print('Acceptable methods are: ',acceptedMethods)            
            sys.exit()
            
        if (method == 'DFP'):
            self.updateHess = hessianUpdaterMethods.DFPmethod
        elif (method == 'BB'):
            self.updateHess = hessianUpdaterMethods.BBmethod
        elif (method == 'BFGS'):
            self.updateHess = hessianUpdaterMethods.BFGSmethod
        else:
            self.updateHess = hessianUpdaterMethods.GBmethod


    
    def initialHessian(self, x_0):
        return identity(len(x_0)) 
        
    def calculateDirection(self, x_k, H_k):
        s_k = -H_k.dot(self.fgradient(x_k))
        return s_k
    
    def updateHessian(self, x_k, x_k1, H_k):
        delta = x_k1 - x_k
        gamma = self.fgradient(x_k1) - self.fgradient(x_k)
        H_k1 = self.updateHess(H_k, delta, gamma)
#        print("H_k1")
#        print(H_k1)
#        print("delta",delta)
#        print("gamma",gamma)
        while True:
            try:
                L = cho.cho_factor(H_k1)
                break
            except cho.LinAlgError:
#                    raise ValueError("Cholesky factorization failed. Possibly non PSD.")
                H_k1 += identity(len(x_k))        
        
        return H_k1
        









            
            
            
            
            
            