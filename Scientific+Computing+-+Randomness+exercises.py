
# coding: utf-8

# In[22]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[11]:

import seaborn as sns


# In[ ]:

#Excercise 1


# In[8]:

#Generating sets of random numbers between 0,1 with increasing sample sizes

mean_list = []
variance_list = []
for i in range(0,20):
    vars()["draws"+str(i)] = np.random.rand(50+250*i)
    mean_list.append(np.mean(vars()["draws"+str(i)]))
    variance_list.append( np.var((vars()["draws"+str(i)])))

# print the mean and variance of these set of random numbers with differing sample sizes
# If the mean and variance harmnize towards (1/2,1/12) respecively, we can conclude that the thoery holds good.


# In[9]:

mean_list


# In[10]:

variance_list


# In[12]:

# As we see in the above samples, the mean and variance harmnize towards (1/2,1/12) respecively.


# In[ ]:

#Excercise 2


# In[23]:

# 1. State description
state_values = ["repaying", "delinquency", "default"]

# 2. Transition probabilities: encoded in a matrix (2d-array) where element [i, j]
# is the probability of moving from state i to state j
P = np.array([[0.85, 0.1, 0.05], [0.25, 0.6, 0.15], [0, 0, 1]])

# 3. Initial distribution: assume loans start in repayment
pi = np.array([1, 0, 0])


# In[24]:

# Define the tolerance for convergence
tolerance = 1e-6

# Define the maximum number of iterations
max_iterations = 1000


# In[25]:

# Perform the power iteration method
for i in range(max_iterations):
    # Compute the next distribution vector
    pi_next = np.dot(pi, P)
    # Check for convergence
    if np.max(np.abs(pi_next - pi)) < tolerance:
        print("Converged after {} iterations".format(i+1))
        break
    # Update the distribution vector
    pi = pi_next

# Normalize the stationary distribution vector
pi /= np.sum(pi)

# Print the stationary distribution vector
print("Stationary distribution: ", pi)


# In[ ]:

#Excercise 3


# In[27]:

# define components here

# Define the transition matrix
P = np.array([[0.95, 0.05], [0.1, 0.9]])

# Define the initial distribution
pi_0 = np.array([0.9, 0.1])



# In[29]:

# construct Markov chain
from quantecon import MarkovChain

# Create an instance of the Markov chain
mc = MarkovChain(P, state_values=('E', 'U'))


# In[34]:

import matplotlib.pyplot as plt

# Simulate the chain 30 times for 50 time periods
T = 50
N = 30
sim_data = mc.simulate(ts_length=T, init=pi_0, num_reps=N)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(50), sim_data.T, alpha=0.4)


# In[32]:

#4

# Compute the stationary distribution
stationary_distributions = mc.stationary_distributions
pi_star = stationary_distributions[0]

# Compute the average long run payment
avg_payment = pi_star @ np.array([10, 1])
print(f"Average long run payment: {avg_payment:.2f}")


# In[ ]:

#Excercise 4


# In[26]:

import scipy.stats as stats

# Asset 1: Normal distribution with mean = 10, standard deviation = 5
asset1 = stats.norm(loc=10, scale=5)

# Asset 2: Gamma distribution with k = 5.3, theta = 2
asset2 = stats.gamma(a=5.3, scale=2)

# Asset 3: Gamma distribution with k = 5, theta = 2
asset3 = stats.gamma(a=5, scale=2)

# Calculate mean, median, and coefficient of variation for each asset
mean_returns = [asset1.mean(), asset2.mean(), asset3.mean()]
median_returns = [asset1.median(), asset2.median(), asset3.median()]
cv_returns = [asset1.std() / asset1.mean(), asset2.std() / asset2.mean(), asset3.std() / asset3.mean()]

print(f"Mean returns: {mean_returns}")
print(f"Median returns: {median_returns}")
print(f"Coefficient of variation: {cv_returns}")


# In[ ]:

#Asset 2 has the highest average returns (mean = 10.6).
#Asset 1 and Asset 2 both have the highest median returns (median = 10.0).
#Asset 3 has the lowest coefficient of variation (CV = 0.447).

#The choice of asset depends on the investor's risk preference and investment goals. 
#If the investor wants the highest average returns, they should choose Asset 2. 
#If they want a lower risk investment with lower variability in returns, they should choose Asset 3. 
#If they have no preference for risk and want a balanced investment with moderate variability in returns, 
    #they should choose Asset 1.


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



