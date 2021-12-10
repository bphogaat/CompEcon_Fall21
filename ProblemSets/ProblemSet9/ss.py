import necessary_equation as ne
import scipy.optimize as opt
import numpy as np

def SS_solver(S, b, v, alpha, delta, A, sigma, l_tilde, beta):
    '''
    Solves for economy when in steady state
    '''
    xi = 0.5          #weight on the current guess
    tol = 1e-8        #tolerance
    max_iter = 500    #maximum number of iterations
    abs_ss = 1        #difference between initial and updated value of r
    ss_iter = 0
    r_guess = 0.05    #initial guess for r
    while abs_ss > tol and ss_iter < max_iter:
        ss_iter += 1
        r_guess = r_guess * np.ones(S)
        print(A,alpha, delta)
        w_old = ne.get_w(r_guess, (alpha, delta, A)) * np.ones(S)
    
        # Household decisions
        c1_guess = 0.1
        c1_args = (r_guess, w_old, beta, sigma, l_tilde, b, v, S)
        sol_c1 = opt.root(ne.get_last_period, c1_guess, args = (c1_args))
        c1 = sol_c1.x
    
        # Supplies for labor
        
        c = ne.get_c(c1, (r_guess, beta, sigma, S))
        n = ne.get_n((c, sigma, l_tilde, b, v, w_old))
        b = ne.get_b(c, n, (r_guess, w_old, S))
        K = ne.get_K(b)
        L = ne.get_L(n)
        r_prime = ne.get_r(K, L, (alpha, delta, A))
        
        # Change the initial guess
        abs_ss = ((r_prime - r_guess) ** 2).max()
        r_guess = xi * r_prime + (1 - xi) * r_guess

    r_ss = r_guess * np.ones(S)
    w_ss = ne.get_w(r_ss, (alpha, delta, A)) * np.ones(S)
 
    return n, b, c, r_ss, w_ss