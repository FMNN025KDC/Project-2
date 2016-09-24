'''Set FMNN25Project2 as consoles working directory!!!'''

import scipy.optimize as so
from rosenbrock import *
from optimizationNewton import *
from linesearch import *
import numpy as np
import chebyquad as cq

#%%
'''Test with easy function'''
def a(x):
    return x[0]**2 + 2*x[1]**4
def b(x):
    return np.array([2*x[0], 8*x[1]**3])
def c(alpha):
	x0 = np.array([3,3])
	s = np.array([-1,-1])
	return a(x0 + alpha*s)

#print(type(a), type(b))
#print(line_search(c, 0))
#print(exact_line_search(c, 0))



#prob = OptimizationProblem(a, np.array([7,11]), b)

#solver = OriginalNewton(prob)
#print("Linus testfunc")
#print(solver.newton_procedure(par_line_search="None"))
#print("test on a,b thing", solver.newton_procedure())

'''No gradient inserted'''
#prob = OptimizationProblem(a, x0 = np.array([7,11]))

#%%
'''Test rosenbrock'''

#prob2 = OptimizationProblem(rosenbrock, np.array([5,6]))

#solver = OriginalNewton(prob)

#print("Rosenbrock test")
#print(solver.newton_procedure(par_line_search= "exact"))

#solver = OptimizationMethodsQuasi(prob, "BB")
#print(solver.newton_procedure(par_line_search = "exact"))

#%%
'''Test on chebyquad ...'''
x0 = np.linspace(0, 1, 11)
print(cq.chebyquad(x0))
print(cq.gradchebyquad(x0))
prob3 = OptimizationProblem(cq.chebyquad, x0, cq.gradchebyquad)
solver3 = OptimizationMethodsQuasi(prob3, 'BFGS')
print(solver3.newton_procedure(par_line_search = 'inexact'))
print('BLANK LINE!!')
print(so.fmin_bfgs(cq.chebyquad, x0, cq.gradchebyquad))
