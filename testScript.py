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
import unittest


class TestOptimizer(unittest.TestCase):

    def testAllCombinations(self):
        acceptedMethods = ['DFP','GB','BFGS','BB']
        lineSearchMethods = ['inexact','exact'] # 'None' is left out due to time consumption 
        conditions = ['goldstein'] # WolfePowell left, does not seem to work
        x_0 = rand(2)*10

        ProblemRB = Op.OptimizationProblem(RB.Rosenbrock(),RB.RosenbrockGradient())

        for i in acceptedMethods:
            for j in lineSearchMethods:
                for k in conditions:
                    solver = Op.QuasiNewton(ProblemRB,i,condition = k)
                    point,value = solver.newton(x_0,j,1e-8)
                    self.assertAlmostEqual(norm(point - array([1.,1.])), 0)
         

    def testChebyquad(self):
        ProblemCQ = Op.OptimizationProblem(ch.Chebyquad(),ch.ChebyquadGradient())
        
        BFGSCh = Op.QuasiNewton(ProblemCQ,"BFGS")
        
        nValues = array([4,8,11])
        optPoints = []
        sciPoints = []
        optValues = []
        sciValues = []
        
        for n in nValues:
            guess = rand(n)     
            optPoint,optValue = BFGSCh.newton(guess,"inexact", 1e-8)
            optValue = ch.chebyquad(optPoint)            
            optPoints.append(optPoint)
            optValues.append(optValue)    
            
            sciPoint = so.fmin_bfgs(ch.Chebyquad(),guess,ch.ChebyquadGradient(),gtol = 1e-8)
            sciValue = ch.chebyquad(sciPoint)
            sciPoints.append(sciPoint)
            sciValues.append(sciValue)
#            self.assertAlmostEqual(norm(sciPoint - optPoint), 0)
            
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

if __name__ == '__main__':
    unittest.main()
      
      
 
#ProblemRB = Op.OptimizationProblem(RB.Rosenbrock(),RB.RosenbrockGradient())
#
#BFGSrb = Op.QuasiNewton(ProblemRB,"BFGS")
#
#BFGSPoint,BFGSValue = BFGSrb.newton(array([2,5]),"inexact")

#Hess = df.getHessian(RB.RosenbrockGradient())
#print(inv(Hess(BFGSPoint)))

#print("--- %s seconds ---" % (time.time() - start_time))


