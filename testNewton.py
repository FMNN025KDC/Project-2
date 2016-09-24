'''This file is under development'''

import unittest
import numpy as np
from optimizationNewton import *
import linesearch
from rosenbrock import *

class TestEverything(unittest.TestCase):
    def setUp(self):
        function = rosenbrock
        self.guess = [10,10]
        self.problem = OptimizationProblem(function, x0 = self.guess)
  
    def test_exact_line_search(self):
        f = lambda x: (x-8)**6
        alpha = linesearch.exact_line_search(f,0)
        self.assertAlmostEqual(alpha,8,4)
    
    '''def test_inexact_line_search(self):
        f = lambda x: (x-8)**6
        fp = lambda x: 6*(x-8)**5
        alpha = linesearch.inexact_line_search(f, fp, 1)
        self.assertAlmostEqual(alpha,0) #How can the inexact line search be tested?'''

    def test_original_newton(self):
        solve = OriginalNewton(self.problem)
        xstar, fmin = solve.newton_procedure(par_line_search = "exact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)
        xstar, fmin = solve.newton_procedure(par_line_search = "inexact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)
    
    def test_quasi_DFP(self):
        solve = OptimizationMethodsQuasi(self.problem, hessupdate = "DFP")
        xstar, fmin = solve.newton_procedure(par_line_search = "exact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)
        xstar, fmin = solve.newton_procedure(par_line_search = "inexact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)
    
    def test_quasi_BFGS(self):
        solve = OptimizationMethodsQuasi(self.problem, hessupdate = "DFP")
        xstar, fmin = solve.newton_procedure(par_line_search = "exact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)
        xstar, fmin = solve.newton_procedure(par_line_search = "inexact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)
        
    def test_quasi_GB(self):
        solve = OptimizationMethodsQuasi(self.problem, hessupdate = "GB")
        '''xstar, fmin = solve.newton_procedure(par_line_search = "exact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)'''
        xstar, fmin = solve.newton_procedure(par_line_search = "inexact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)
        
    def test_quasi_BB(self):
        solve = OptimizationMethodsQuasi(self.problem, hessupdate = "BB")
        '''xstar, fmin = solve.newton_procedure(par_line_search = "exact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)'''
        xstar, fmin = solve.newton_procedure(par_line_search = "inexact")
        self.assertAlmostEqual(xstar.all(),1 , 4)
        self.assertAlmostEqual(fmin , 0, 4)
        
if __name__ == '__main__':
	unittest.main()

'''def test_exception(self):self.assertRaises(ValueError, self.s, self.grid[1])'''