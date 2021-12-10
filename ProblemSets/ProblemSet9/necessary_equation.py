import numpy as np
import scipy.optimize as opt

def get_L(n):
    '''
    Function to compute aggregate labor supplied
    '''
    L = n.sum()
    return L

def get_K(b):
    '''
    Funstion to compute aggregate capital supplied
    '''
    K = b.sum()
    return K


def get_r(K, L, params):
    '''
    Compute the interest rate from the firm's FOC
    '''
    alpha, delta , A=params
    r = alpha*A*(L/K)**(1-alpha)-delta
    return r

def get_w(r, params):
    '''
    Solve for the w that is consistent with r from the firm's FOC
    '''
    alpha, delta, A=params
    w = (1-alpha)*A*((alpha*A)/(r+delta))**(alpha/(1-alpha))
    return w

def get_c(c1, params): 
    '''
    Find consumption using the budget constraint
    '''
    r, beta, sigma, period= params
    c = np.zeros(period)
    c[0] = c1  #initial guess
    cs = c1
    s = 0
    while s < period - 1:
        c[s + 1] = cs * (beta * (1 + r[s + 1])) ** (1 / sigma)
        cs = c[s + 1]
        s += 1
    return c


def mu_c_func(c, sigma):
    '''
    Marginal utility of consumption
    '''
    mu_c = np.power(c, (-sigma))
    epsilon = 1e-6
    b1 = (-sigma) * np.power(epsilon, (-sigma - 1))
    b2 = np.power(epsilon,(-sigma)) - b1 * epsilon
    c_constr = c < epsilon
    mu_c[c_constr] = b1 * c[c_constr] + b2
    return mu_c


def mu_d_func(n, params):
    l_tilde, b, v = params
    epsilon_lb = 1e-6
    epsilon_ub = l_tilde - epsilon_lb
    
    '''
    Marginal disutility of labor
    '''
    mu_d = ((b / l_tilde) * ((n / l_tilde) ** (v - 1)) * (1 - ((n / l_tilde) ** v)) **\
           ((1 - v) / v))

    m1 = (b * (l_tilde ** (-v)) * (v - 1) * (epsilon_lb ** (v - 2)) * \
         ((1 - ((epsilon_lb / l_tilde) ** v)) ** ((1 - v) / v)) * \
         (1 + ((epsilon_lb / l_tilde) ** v) * ((1 - ((epsilon_lb / l_tilde) ** v)) ** (-1))))
    m2 = ((b / l_tilde) * ((epsilon_lb / l_tilde) ** (v - 1)) * \
         ((1 - ((epsilon_lb / l_tilde) ** v)) ** ((1 - v) / v)) - (m1 * epsilon_lb))

    q1 = (b * (l_tilde ** (-v)) * (v - 1) * (epsilon_ub ** (v - 2)) * \
         ((1 - ((epsilon_ub / l_tilde) ** v)) ** ((1 - v) / v)) * \
         (1 + ((epsilon_ub / l_tilde) ** v) * ((1 - ((epsilon_ub / l_tilde) ** v)) ** (-1))))

    q2 = ((b / l_tilde) * ((epsilon_ub / l_tilde) ** (v - 1)) * \
         ((1 - ((epsilon_ub / l_tilde) ** v)) ** ((1 - v) / v)) - (q1 * epsilon_ub))

    nl_constr = n < epsilon_lb
    nu_constr = n > epsilon_ub

    mu_d[nl_constr] = m1 * n[nl_constr] + m2
    mu_d[nu_constr] = q1 * n[nu_constr] + q2
    return mu_d

def get_b(c, n, params_b): 
    '''
    Function to calculate lifetime savings
    '''
    r, w, period = params_b
    bs = 0.0
    b = np.zeros(period)
    b[0] = bs
    s = 0
    while s < period - 1:
        b[s + 1] = (1 + r[s]) * bs + w[s] * n[s] - c[s]
        bs = b[s + 1]
        s += 1
    return b


def get_errors(n, *args):
    '''
    Function to euler error
    '''
    c, sigma, l_tilde, x, v, w = args
    mu_c = mu_c_func(c, sigma)
    mu_d = mu_d_func(n, (l_tilde, x, v))
    euler_error = w * mu_c - mu_d
    return euler_error


def get_n(params): 
    '''
    Function to calculate optimal root value
    '''
    c, sigma, l_tilde, b, v, w=params
    n_args = params
    S = 100
    n_guess = 0.5 * l_tilde * np.ones(S)
    sol = opt.root(get_errors, n_guess, args = (n_args), method = 'lm')
    n = sol.x
    return n


def get_last_period(c1, *args): 
    '''
    Function to last-period savings, given intial guess c1
    '''
    r, w, beta, sigma, l_tilde, b, v, period = args
    c = get_c(c1, (r, beta, sigma, period))
    n = get_n((c, sigma, l_tilde, b, v, w))
    b = get_b(c, n, (r, w, period))
    b_final = (1 + r[-1]) * b[-1] + w[-1] * n[-1] - c[-1]
    return b_final

