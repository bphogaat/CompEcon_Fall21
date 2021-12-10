import ss as SS
import numpy as np
import matplotlib.pyplot as plt

# Household Parameters
S = 100
beta = 0.9 
sigma = 2.0
l_tilde = 1.0
b = 0.501
v = 1.554

# Firms Parameters
alpha = 0.35
A = 1.0
delta = 0.6

# SS Solver
n, b, c, r_ss, w_ss = SS.SS_solver(S, b, v, alpha, delta, A, sigma, l_tilde, beta)

# 1. Plot of savings for each age
plt.plot ( np.arange(100), b, 'g.-', color = 'blue', label = 'savings')
plt.title('Savings distribution for each age', fontsize=15)
plt.xlabel('Age')
plt.ylabel('Savings')
plt.legend()
plt.show()
    
# 2. Plot of labor supply for each age
plt.plot ( np.arange(100), n, 'g.-', color = 'magenta', label = 'labor supply')
plt.title('Labor Supply distribution for each age', fontsize=15)
plt.xlabel('Age')
plt.ylabel('labor supply')
plt.legend()  
plt.show() 
    
# 3. Plot of consumption for each age
plt.plot (np.arange(100), c, 'g.-', color = 'green', label = 'consumption')
plt.title('Consumption distribution for each age', fontsize=15)
plt.xlabel('Age')
plt.ylabel('Consumption')
plt.legend()
plt.show()










