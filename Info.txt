This is project 2 in FMNN25 about Quasi-Newton optimisation methods

generic structure 

x0,f,g
G0 = inital_hessian(g,xk)
while (cond):
    1 sk = inv(Gk)gk
    2 a) if line_search -> compute alphak
      b) else alphak = 1
    3 x_k+1 = update formula
    4 update G
    
Newton orig
    1) sk = solve_che..(G, )
    2) a) exact/ inexact
        b) alpha_k = 1
    3) bla bla 
    4) initial_hessian(g, x_k+1) 
    
Newton Q 
inital_hessian = eye 
    1) sk = -inv(Gk)gk # eventuellt är Gk inv(Gk)
    2) samma som i newton orig 
    3) update x_k+1
    4) update_hessian
