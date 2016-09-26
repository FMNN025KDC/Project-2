# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:43:18 2016
@author: David Olsson, Kristoffer Svendsen, Claes Svensson
"""
from  scipy import *
from  pylab import *
from scipy import optimize as so

import Optimization as Op
import chebyquad_problem_NG4oWEq as ch
import RosenbrockFunction as RB
import differentiation as df
import time
start_time = time.time()

#ProblemCQ = Op.OptimizationProblem(ch.Chebyquad(),ch.ChebyquadGradient())
#
#GoodCh = Op.QuasiNewton(ProblemCQ)
#
#nValues = array([4,8,11])
#optPoints = []
#sciPoints = []
#optValues = []
#sciValues = []
#
#for n in nValues:
#    guess = rand(n)     
#    optPoint,optValue = GoodCh.newton(guess,"inexact", 1e-7)
#    optPoints.append(optPoint)
#    optValues.append(optValue)    
#    
#    sciPoint = so.fmin_bfgs(ch.Chebyquad(),guess,ch.ChebyquadGradient())
#    sciValue = ch.chebyquad(sciPoint)
#    sciPoints.append(sciPoint)
#    sciValues.append(sciValue)
    
#for i in range(0,len(optPoints)):
#    print("--------------------------")
#    print("Our class: ")
#    print("Point: ",optPoints[i])
#    print("Value: ",optValues[i])
#    print(" ")
#    print("Scipy class: ")
#    print("Point: ",sciPoints[i])
#    print("Value: ",sciValues[i])
#    print("--------------------------")



acceptedMethods = ['DFP','GB','BFGS','BB']
lineSearchMethods = ['inexact','exact']
conditions = ['goldstein']
x0 = rand(2)*10

ProblemRB = Op.OptimizationProblem(RB.Rosenbrock(),RB.RosenbrockGradient())

for i in acceptedMethods:
    for j in lineSearchMethods:
        for k in conditions:
            solver = Op.QuasiNewton(ProblemRB,i,condition = k)
            point,value = solver.newton(x0,j)
            assert norm(point - array([1,1])) < 0.1
            
            
#ProblemRB = Op.OptimizationProblem(RB.Rosenbrock(),RB.RosenbrockGradient())
#solver = Op.QuasiNewton(ProblemRB,'BB', condition = 'wolfePowell')
#point,value = solver.newton(x0,'inexact')


#assert norm(point - array([1,1])) < 0.1
            



 
    
#ProblemRB = Op.OptimizationProblem(RB.Rosenbrock(),RB.RosenbrockGradient())
#
#BFGSrb = Op.QuasiNewton(ProblemRB,"BFGS")
#
#BFGSPoint,BFGSValue = BFGSrb.newton(array([2,5]),"inexact")

#Hess = df.getHessian(RB.RosenbrockGradient())
#print(inv(Hess(BFGSPoint)))

print("--- %s seconds ---" % (time.time() - start_time))


