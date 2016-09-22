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