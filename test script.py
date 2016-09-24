# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 20:17:30 2016
@author: poff
"""
from  scipy import *
from  pylab import *


def Rosenbrock():
    def RBFunc(x):
        x = array(x)
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    return RBFunc
    
    
func=Rosenbrock()
'''this point is not PSD after the first step'''
x0=array([-5,-10])
'''---------------------------------------'''


opt=Optimizer(func)
opt.newton(x0,False,'BFGS')
