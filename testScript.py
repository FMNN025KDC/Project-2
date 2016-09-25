# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:43:18 2016
@author: tfy12dol
"""
from  scipy import *
from  pylab import *
from scipy import optimize as so

import Optimization as Op
import chebyquad_problem_NG4oWEq as ch
import RosenbrockFunction as RB
import differentiation as df

ProblemCQ = Op.OptimizationProblem(ch.Chebyquad(),ch.ChebyquadGradient())

GoodCh = Op.QuasiNewton(ProblemCQ)

nValues = array([4,8,11])
optPoints = []
sciPoints = []
optValues = []
sciValues = []

for n in nValues:
    guess = rand(n)     
    optPoint,optValue = GoodCh.newton(guess,"inexact", 1e-5)
    optPoints.append(optPoint)
    optValues.append(optValue)    
    
    sciPoint = so.fmin_bfgs(ch.Chebyquad(),guess,ch.ChebyquadGradient())
    sciValue = ch.chebyquad(sciPoint)
    sciPoints.append(sciPoint)
    sciValues.append(sciValue)
    
for i in range(0,len(optPoints)):
    print("--------------------------")
    print("Our class: ")
    print("Point: ",optPoints[i])
    print("Value: ",optValues[i])
    print(" ")
    print("Scipy class: ")
    print("Point: ",sciPoints[i])
    print("Value: ",sciValues[i])
    print("--------------------------")
    
    
ProblemRB = Op.OptimizationProblem(RB.Rosenbrock(),RB.RosenbrockGradient())

BFGSrb = Op.QuasiNewton(ProblemRB,"BFGS")

BFGSPoint,BFGSValue = BFGSrb.newton(array([2,5]),"inexact")

Hess = df.getHessian(RB.RosenbrockGradient())
print(inv(Hess(BFGSPoint)))

    



