import numpy as np

def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    
def rosenbrock_grad(x):
    # Detta är nu rätt
    return np.array([-400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]), 200*(x[1] - x[0]**2)])
    
if __name__ == "__main__":
    print(rosenbrock((1,2)))
    print(rosenbrock_grad((1,2)))
