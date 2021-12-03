#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from functions import SS_grid, UT_grid, VFI, opt
import matplotlib.pyplot as plt

# In[ ]:


size_C = 500


# In[ ]:


C_grid = SS_grid(0.4, 2.0, size_C)


# In[ ]:


U = UT_grid(size_C, C_grid)


# In[ ]:


PF, VF = VFI(1e-8, 7.0, 3000, size_C, U)

opta = opt(C_grid,PF)

# In[ ]:


# Plot value function 
plt.figure()
plt.scatter(C_grid[1:], VF[1:])
plt.xlabel('Customers')
plt.ylabel('Value Function')
plt.title('Value Function - deterministic advertisement spending')
plt.show()


#Plot optimal advertising spending rule as a function of customers
plt.figure()
fig, ax = plt.subplots()
ax.plot(C_grid[3:], opta[3:], label='Spending')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Proportion of customers')
plt.ylabel('Optimal Spending on Advertising')
plt.title('Policy Function, customers - deterministic advertisement spending')
plt.show()