import numpy as np
import scipy.linalg as sl

def exact_line_search(f, alpha_0, tol = 1e-5):
    # dichotomus line search
    eps = min(1e-8, tol/50)
    a = alpha_0
    b = alpha_0 + 1
    f1 = f(a)
    f2 = f(b)
    c = 1;
    while f1 > f2:
        f1 = f(b)
        c *= 4
        b += c
        f2 = f(b)

    while abs(b - a) > tol:
        mid = (a + b)/2
        f1 = f(mid - eps)
        f2 = f(mid + eps)
        if (f1 < f2):
            b = mid
        else:
            a = mid
    return a

def eval_LC(f, fp, alpha_0, alpha_l, rho = 0.1):
    return f(alpha_0) >= f(alpha_l) + (1 - rho)*(alpha_0 - alpha_l)*fp(alpha_l)

def eval_RC(f, fp, alpha_0, alpha_l, rho = 0.1):
    return f(alpha_0) <= f(alpha_l) + rho*(alpha_0 - alpha_l)*fp(alpha_l)

def goldstein_extrapolate(fp, alpha_0, alpha_l):
    return (alpha_0 - alpha_l)*fp(alpha_0)/(fp(alpha_l) - fp(alpha_0))

def goldstein_interpolate(f, fp, alpha_0, alpha_l):
    num = fp(alpha_l)*(alpha_0 - alpha_l)**2
    denom = 2*(f(alpha_l) - f(alpha_0) + (alpha_0 - alpha_l)*fp(alpha_l))
    return num/denom

def inexact_line_search(f, fp, alpha_0):
    # goldstein
    alpha_0 += 1
    tau = 0.1
    chi = 9
    alpha_l = 0
    alpha_u = 10**4
    LC = eval_LC(f, fp, alpha_0, alpha_l)
    RC = eval_RC(f, fp, alpha_0, alpha_l)
    while not (LC and RC):
        if not LC:
            delta_alpha_0 = goldstein_extrapolate(fp, alpha_0, alpha_l)
            delta_alpha_0 = max(delta_alpha_0, tau*(alpha_0 - alpha_l))
            delta_alpha_0 = min(delta_alpha_0, chi*(alpha_0 - alpha_l))
            alpha_l = alpha_0
            alpha_0 += delta_alpha_0
        else:
            alpha_u = min(alpha_0, alpha_u)
            alpha_0_bar = goldstein_interpolate(f, fp , alpha_0, alpha_l)
            alpha_0_bar = max(alpha_0_bar, alpha_l + tau*(alpha_u - alpha_l))
            alpha_0_bar = min(alpha_0_bar, alpha_u - tau*(alpha_u - alpha_l))
            alpha_0 = alpha_0_bar
        LC = eval_LC(f, fp, alpha_0, alpha_l)
        RC = eval_RC(f, fp, alpha_0, alpha_l)
    return alpha_0
