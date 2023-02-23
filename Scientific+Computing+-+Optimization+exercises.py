
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:




# In[ ]:

#Excercise 2


# In[2]:

W = 10
p_P = 1
p_C = 2


# In[3]:

from scipy.optimize import minimize
import numpy as np

# Define the utility function
def utility(x):
    return -((x[0]-20)**2) - 2*((x[1]-1)**2)

# Define the constraint equation
def constraint(x):
    return x[0] + x[1] - W

# Define the bounds for the variables
bounds = ((0, None), (0, None))

# Define the initial guess for the variables
x0 = np.array([10, 10])


# In[4]:

# Define the value of W
W = 10

# Define the optimization problem
problem = {'fun': utility, 'x0': x0, 'bounds': bounds, 'constraints': {'type': 'ineq', 'fun': constraint}}

# Solve the optimization problem
result = minimize(**problem)

# Print the solution
print(result)


# In[4]:

#W=50

W = 50

# Define the optimization problem
problem = {'fun': utility, 'x0': x0, 'bounds': bounds, 'constraints': {'type': 'ineq', 'fun': constraint}}

# Solve the optimization problem
result = minimize(**problem)

# Print the solution
print(result)


# In[6]:

#W=150

W = 150

# Define the optimization problem
problem = {'fun': utility, 'x0': x0, 'bounds': bounds, 'constraints': {'type': 'ineq', 'fun': constraint}}

# Solve the optimization problem
result = minimize(**problem)

# Print the solution
print(result)


# In[ ]:



