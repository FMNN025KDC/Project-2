# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:19:33 2016
@author:  David Olsson, Kristoffer Svendsen, Claes Svensson
"""
from  scipy import *
from  pylab import *
import scipy.linalg as cho
import differentiation
import lineSearchers
import hessianUpdaterMethods

class OptimizationProblem():
    """Class containing problem to optimize.
    
        Args:
            function: function object of which to find minimum.
            fgradient: {Optional} {Default = None} gradient of function, if no
                is given the gradient is numerically approximated using 
                differentiation.getGradient(function).
    """
    def __init__(self, function, fgradient = None):
        self.function = function
        if fgradient is None:        
            self.fgradient = differentiation.getGradient(function)
        else:
            self.fgradient = fgradient
            
        
class OptimizationMethod():
    """Superclass for optimization.
    
        Args:
            problem: OptimizationProblem object.
            condition: {Optional} {Default = "goldstein"} String containing 
                type of condition for inexact line search.
            rho: {Optional} {Default = 0.1} Parameter for inexact line search
            sigma: {Optional} {Default = 0.7} Parameter for inexact line search
            tau: {Optional} {Default = 0.1} Parameter for inexact line search
            xi: {Optional} {Default = 9} Parameter for inexact line search
    """
    def __init__(self, problem, condition = "goldstein", rho = 0.1, sigma = 0.7, tau = 0.1, xi = 9):
        self.function = problem.function
        self.fgradient = problem.fgradient
        
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.xi = xi
        self.condition = condition
    
    def newton(self, x_0, lineSearchMethod = "exact", tol = 1e-3):
        """Finds minimum of function using newton or quasi-newton method
            depending on the line search method used.
            
            Args:
                x_0: nparray object containing initial guess of point 
                    minimizing function.
                lineSearchMethod: {Optional} {Default = "exact"} String containing 
                    type of line search to be used.
                tol: {Optional} {Default = 1e-3} Tolerance of norm of gradient
                    difference from zero.
                    
            Returns:
                x_k: nparray object containing point which minimizes function.
                f_k: function value in point x_k
        """
        
        x_k = x_0
        # Initial Hessian is calculated differently for different methods.
        G_k = self.initialHessian(x_0)
        # Gets the correct line searcher method depending on input.
        lineSearcher = self.getLineSearcher(lineSearchMethod)
        # For plotting how our solver steps when solving Rosenbrock function.
#        xp = [] 
#        yp = []
        
        while True:
            # Calculates direction to step.
            s_k = self.calculateDirection(x_k, G_k)
            # Calculate step length
            alpha_k = lineSearcher(self.function_alpha(x_k, s_k), self.fgradient_alpha(x_k, s_k), self.condition, self.rho, self.sigma, self.tau, self.xi)
            # Find new point
            x_k1 = x_k + alpha_k * s_k
            # Some print outs to verify that the code does what it's supposed to.
#            print("-----------------------------------")
#            print("alpha:",alpha_k,"x_k:",x_k,"x_k1:",x_k1,"s_k:",s_k)
            # For plotting how our solver steps when solving Rosenbrock function.
#            xp.append(x_k[0])
#            yp.append(x_k[1])
            
            # Update Hessian.
            G_k = self.updateHessian(x_k, x_k1, G_k)
            x_k = x_k1
#            print("G_k, or H_k for QuasiNewton:")
#            print(G_k)
            
            # If condition is fulfilled, break while loop.
            if (norm(self.fgradient(x_k)) < tol):
                break
        
#        print("-----------------------------------")
        # For plotting how our solver steps when solving Rosenbrock function.
#        plot(xp,yp,'--o')
#        print("Found point: ",x_k)
        f_k = self.function(x_k)
#        print("Function value: ",f_k)
        return x_k,f_k
      
    def function_alpha(self, x_k, s_k): 
        """Defines a new function object 
            function_alpha(alpha) = function(x_k + alpha * s_k)
            
            Args:
                x_k: nparray object
                s_k: nparray object
            
            Returns:
                Function object
                function_alpha(alpha) = function(x_k + alpha * s_k)
        """
        def func_alpha(alpha):
            return self.function(x_k + alpha * s_k)
        return func_alpha
        
    def fgradient_alpha(self, x_k, s_k):
        """Defines a new function object 
            fgradient_alpha(alpha) = fgradient(x_k + alpha * s_k)
            
            Args:
                x_k: nparray object
                s_k: nparray object
                
            Returns:
            Function object
            fgradient_alpha(alpha) = fgradient(x_k + alpha * s_k)
        """
        def fgrad_alpha(alpha):
            return self.fgradient(x_k + alpha * s_k).dot(s_k)
        return fgrad_alpha
    
    def getLineSearcher(self,lineSearchMethod):
        """Returns line searcher method function.
        
            Args:
                lineSearchMethod: String object containing name of method to be 
                    used. {"exact","inexact","None"}
        """
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
        """Abstract. Will be shadowed in sub class"""        
        return None
               
    def calculateDirection(self, x_k, G_k):
        """Abstract. Will be shadowed in sub class"""        
        return None      
        
    def updateHessian(self, x_k, x_k1, G_k):
        """Abstract. Will be shadowed in sub class"""
        return None
        
        
        
class ClassicalNewton(OptimizationMethod):
    """Subclass to OptimizationMethod. Uses classical newton as default 
        stepping. Calculates the step direction using cholesky factorization.
        
        Args:
            problem: OptimizationProblem object.
            condition: {Optional} {Default = "goldstein"} String containing 
                type of condition for inexact line search.
            rho: {Optional} {Default = 0.1} Parameter for inexact line search
            sigma: {Optional} {Default = 0.7} Parameter for inexact line search
            tau: {Optional} {Default = 0.1} Parameter for inexact line search
            xi: {Optional} {Default = 9} Parameter for inexact line search
    """
    
    def initialHessian(self, x_0):
        """Calculates the initial hessian through finite difference 
            approximation in the point x_0.
            
            Args:
                x_0: nparray object contaning point of evaluation.
                
            Returns:
                nparry object containing hessian evaluated in point x_0 
        """
        return differentiation.evaluateHessian(self.fgradient, x_0)    
    
    def calculateDirection(self, x_k, G_k):
        """Calculates the step direction using cholesky factorization.
        
            Args:
                x_k: nparray object containing current point.
                G_k: nparray object containing hessian evaluated in current 
                    point.
                    
            Returns:
                nparray object containing the direction to step.
        """
        
        # Make sure the hessian is symmetric.
        G_k = 0.5 * (G_k + G_k.T)
        
        # While the hessian is non PSD, keep adding identity. For a matrix with
        # large enough elements in the diagonal (compared to off-diagonal) the 
        # diagonal elements are approximately the eigenvalues. 
        while True:
            try:
                # Use cholesky factorization to check if PSD.
                L = cho.cho_factor(G_k)
                break
            # We assume that all LinAlgError raised by cho_factor are due to 
            # the matrix being non PSD.
            except cho.LinAlgError:
                print("LinAlgError: Matrix was possibly non PSD. Adding identity to compensate.")
                G_k += identity(len(x_k))
                
        # Solve cholesky.
        s_k = -cho.cho_solve(L, self.fgradient(x_k))
        return s_k
        
    def updateHessian(self, x_k, x_k1, G_k):
        """Updates the hessian by calculating a new hessian in the new point.
            
            Args:
                x_k: nparray object contaning previous point.
                x_k1: nparray object contaning current point.
                G_k: nparray object contaning previous hessian.                
                
            Returns:
                nparry object containing hessian evaluated in point x_k1 
        """
        return differentiation.evaluateHessian(self.fgradient, x_k1)
        
    

class QuasiNewton(OptimizationMethod):
    """Subclass to OptimizationMethod. Uses quasi-newton solver.
        
        Args:
            problem: OptimizationProblem object.
            method: {Optional} {Default = "GB"} String containing name of 
            method to be uesd for updating hessian.
            condition: {Optional} {Default = "goldstein"} String containing 
                type of condition for inexact line search.
            rho: {Optional} {Default = 0.1} Parameter for inexact line search
            sigma: {Optional} {Default = 0.7} Parameter for inexact line search
            tau: {Optional} {Default = 0.1} Parameter for inexact line search
            xi: {Optional} {Default = 9} Parameter for inexact line search
    """
    def __init__(self, problem, method = "GB", condition = "goldstein",rho = 0.1, sigma = 0.7, tau = 0.1, xi = 9):
        # Send args to super class.
        super().__init__(problem, condition, rho, sigma, tau, xi)
        
        # Checks whether the method is in the list of accepted methods.
        acceptedMethods=['DFP','GB','BFGS','BB']        
        if not method in acceptedMethods:            
            print('Not acceptable method.')
            print('Acceptable methods are: ',acceptedMethods)            
            sys.exit()
            
        # Assigna updating method depending on input string.
        if (method == 'DFP'):
            self.updateHess = hessianUpdaterMethods.DFPmethod
        elif (method == 'BB'):
            self.updateHess = hessianUpdaterMethods.BBmethod
        elif (method == 'BFGS'):
            self.updateHess = hessianUpdaterMethods.BFGSmethod
        else:
            self.updateHess = hessianUpdaterMethods.GBmethod


    
    def initialHessian(self, x_0):
        """Returns the identity matrix as the initial hessian.
            
            Args:
                x_0: nparray object contaning point of evaluation.
                
            Returns:
                nparry object identity matrix with size len(x_0)xlen(x_0) 
        """
        
        # Identity is its own inverse, G = H.
        return identity(len(x_0)) 
        
    def calculateDirection(self, x_k, H_k):
        """Calculates the step direction.
        
            Args:
                x_k: nparray object containing current point.
                G_k: nparray object containing the inverse of the hessian
                    evaluated in current point.
                    
            Returns:
                nparray object containing the direction to step.
        """
        s_k = -H_k.dot(self.fgradient(x_k))
        return s_k
    
    def updateHessian(self, x_k, x_k1, H_k):
        """Updates the hessian using the assigned method.
            
            Args:
                x_k: nparray object contaning previous point.
                x_k1: nparray object contaning current point.
                G_k: nparray object contaning previous hessian.                
                
            Returns:
                nparry object containing hessian evaluated in point x_k1 
        """
        delta = x_k1 - x_k
        gamma = self.fgradient(x_k1) - self.fgradient(x_k)
        # Calls function depending on method chosen in __init__
        H_k1 = self.updateHess(H_k, delta, gamma)

        # While the hessian is non PSD, keep adding identity. For a matrix with
        # large enough elements in the diagonal (compared to off-diagonal) the 
        # diagonal elements are approximately the eigenvalues. 
        while True:
            try:
                # Use cholesky factorization to check if PSD.                
                L = cho.cho_factor(H_k1)
                break
            # We assume that all LinAlgError raised by cho_factor are due to 
            # the matrix being non PSD.
            except cho.LinAlgError:
                print("LinAlgError: Matrix was possibly non PSD. Adding identity to compensate.")
                H_k1 += identity(len(x_k))        
        
        return H_k1
        









            
            
            
            
            
            