
# coding: utf-8

# In[2]:

# import numpy to prepare for code below
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[ ]:

#Excercise 1


# In[37]:

N_A = 100
N_B = 50
i = 0.05
p_A_y = 1500
p_B_y = 500
t_A = 6
t_B = 4
target = 500000
N_A


# In[51]:

N = np.array([N_A,N_B])
p = np.array([p_A_y,p_B_y])
t = np.array([t_A,t_B])


# In[54]:

#Cash flow calculation

Cashflow = np.zeros(np.size(N))
NPV =  np.zeros(np.size(N))

for j in range(0,np.size(N)):
    Cashflow[j] = p[j] * N[j]
    NPV[j] = np.npv(i,[Cashflow[j]]*t[j])/1.05 #adjusting for NPV in python to match excel definition
    
amt = np.ones(np.size(N)) @ NPV
print(amt)


# In[55]:

if amt > target:
    print("Alice can retire")
else:
    print("Alice cannot retire yet")


# In[ ]:

#Excercise 2


# In[56]:

x1 = np.reshape(np.arange(6), (3, 2))
x2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
x3 = np.array([[2, 5, 2], [1, 2, 1]])
x4 = np.ones((2, 3))

y1 = np.array([1, 2, 3])
y2 = np.array([0.5, 0.5])


# In[57]:

# won't work
x1 @ x2


# In[58]:

# won't work
x2 @ x1


# In[59]:

#works
x2 @ x3


# In[60]:

# won't work
x3 @ x2


# In[61]:

# works
x1 @ x3


# In[62]:

# works
x4 @ y1


# In[63]:

# won't work
x4 @ y2


# In[65]:

# won't work
y1 @ x4


# In[66]:

# works
y2 @ x4


# In[ ]:

#Excercise 3


# In[3]:

# Long simulation

phi = 0.1
alpha = 0.05

x0 = np.array([900_000, 100_000])

A = np.array([[1-alpha, alpha], [phi, 1-phi]])

def simulate(x0, A, T=10):
    """
    Simulate the dynamics of unemployment for T periods starting from x0
    and using values of A for probabilities of moving between employment
    and unemployment
    """
    nX = x0.shape[0]
    out = np.zeros((T, nX))
    out[0, :] = x0

    for t in range(1, T):
        out[t, :] = A.T @ out[t-1, :]

    return out


# In[4]:

def plot_simulation(x0, A, T=100):
    X = simulate(x0, A, T)
    fig, ax = plt.subplots()
    ax.plot(X[:, 0])
    ax.plot(X[:, 1])
    ax.set_xlabel("t")
    ax.legend(["Employed", "Unemployed"])
    return ax

plot_simulation(x0, A, 5000)


# In[5]:

#Eigen values and Eigen vectors

eigvals, eigvecs = np.linalg.eig(A.T)
for i in range(len(eigvals)):
    if eigvals[i] == 1:
        which_eig = i
        break

print(f"We are looking for eigenvalue {which_eig}")


# In[6]:

dist = eigvecs[:, which_eig]

# need to divide by sum so it adds to 1
dist /= dist.sum()

print(f"The distribution of workers is given by {dist}")


# In[ ]:

# Multiplying the no. of workers (1000000) with the distribution


# In[14]:

No_workers = 1000000
Dist_mul = No_workers*dist
Dist_mul


# In[9]:

#long simulation figures
X = simulate(x0, A, T=500)


# In[15]:

Sim_compare = X-Dist_mul


# In[16]:

# The values of the simulation eventually converge toward Eigen value associated with the Eigen vectors
# the values of the simulation differ in the initial periods and eventually match the Eigen Values

Sim_compare


# In[ ]:



