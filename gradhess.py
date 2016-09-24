import numpy as np
from scipy.misc import derivative

def calc_gradient(f, x0, spacing = 1e-5):
    n = len(x0)
    g = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        def ff(xi):
            return f(x0 + xi*ei)
        g[i] = derivative(ff, 0, spacing)
    return g

def calc_hessian(g, x0, spacing = 1e-5):
    n = len(x0)
    G = np.zeros((n, n))
    for i in range(n):
        def gi(x):
            return g(x)[i]
        col = calc_gradient(gi, x0)
        G[:, i] = col
    return G

def get_gradient(f):
    def g(x):
        return calc_gradient(f, x)
    return g

def get_hessian(g):
    def G(x):
        return calc_hessian(g, x)
    return G
