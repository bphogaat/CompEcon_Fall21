#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import numpy as np
import matplotlib.pyplot as plt

# to print plots inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


beta = 0.95
sigma = 1.0
rho = 1.0


# In[ ]:


def SS_grid(lb_C, ub_C, size_C):
    '''
    Creates grid for state space
    lb_C      = scalar, lower bound of customer grid
    ub_C      = scalar, upper bound of customer grid 
    size_C    = integer, number of grid points in customer state space
    C_grid    = vector, size_C x 1 vector of customer grid points 

    '''
    C_grid = np.linspace(lb_C, ub_C, size_C)
    return C_grid


# In[ ]:


def UT_grid(size_C, C_grid):
    '''
    Creates grid of current utility values  
    a        = matrix, current spending on advertising (a=exp(C'-(rho*C)))
    U        = matrix, current period count of number of customers of C and C' (rows are C', columns C)
    '''
    a = np.zeros((size_C, size_C))
    b = np.zeros((size_C)) 
    for i in range(size_C): # loop over C
        for j in range(size_C): # loop over C'
            a[i, j] = np.exp((C_grid[i] - (rho*C_grid[j]))) 
            b[j] = (rho*C_grid[j])

    if sigma == 1:
        U =  b - a
    else:
        U = (a ** (1 - sigma)) / (1 - sigma)
    U[a<0] = -9999999
    return U


# In[ ]:


def VFI(VFtol,VFdist,VFmaxiter,size_C,U):
    '''
    ------------------------------------------------------------------------
    Value Function Iteration    
    ------------------------------------------------------------------------
    VFtol     = scalar, tolerance required for value function to converge
    VFdist    = scalar, distance between last two value functions
    VFmaxiter = integer, maximum number of iterations for value function
    V         = vector, the value functions at each iteration
    Vmat      = matrix, the value for each possible combination of C and C'
    Vstore    = matrix, stores V at each iteration 
    VFiter    = integer, current iteration number
    TV        = vector, the value function after applying the Bellman operator
    PF        = vector, indicies of choices of C' for all C 
    VF        = vector, the "true" value function
    ------------------------------------------------------------------------
    '''
    V = np.zeros(size_C) # initial guess at value function
    Vmat = np.zeros((size_C, size_C)) # initialize Vmat matrix
    Vstore = np.zeros((size_C, VFmaxiter)) #initialize Vstore array
    VFiter = 1 
    while VFdist > VFtol and VFiter < VFmaxiter:  
        for i in range(size_C): # loop over C
            for j in range(size_C): # loop over C'
                Vmat[i, j] = U[i, j] + beta * V[j] 

        Vstore[:, VFiter] = V.reshape(size_C,) # store value function at each iteration for graphing later
        TV = Vmat.max(1) # apply max operator to Vmat (to get V(w))
        PF = np.argmax(Vmat, axis=1)
        VFdist = (np.absolute(V - TV)).max()  # check distance
        V = TV
        VFiter += 1 



    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')


    VF = V # solution to the functional equation
    return PF, VF


# In[ ]:

def opt(C_grid,PF):
    '''
    ------------------------------------------------------------------------
    Find consumption and savings policy functions   
    ------------------------------------------------------------------------
    optC  = vector, the optimal choice of C' for each C
    opta  = vector, the optimal choice of a for each a
    ------------------------------------------------------------------------
    '''
    optC = C_grid[PF] # tomorrow's optimal customer size 
    opta = optC - (rho*C_grid) # optimal advertisement spending - get spending through the transition eqn
    return opta


