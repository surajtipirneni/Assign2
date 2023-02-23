
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np


# In[ ]:

#Excercise 1


# In[2]:

get_ipython().magic('matplotlib inline')


# In[3]:

# Step 1
fig, ax = plt.subplots()

# Step 2
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Step 3
ax.plot(x, y)


# In[4]:

fig, ax = plt.subplots()

fig.set_facecolor("red")
ax.set_facecolor("blue")


# In[5]:

# We specified the shape of the axes -- It means we will have two rows and three columns
# of axes on our figure
fig, axes = plt.subplots(2, 3)

fig.set_facecolor("gray")

# Can choose hex colors
colors = ["#065535", "#89ecda", "#ffd1dc", "#ff0000", "#6897bb", "#9400d3"]

# axes is a numpy array and we want to iterate over a flat version of it
for (ax, c) in zip(axes.flat, colors):
    ax.set_facecolor(c)

fig.tight_layout()


# In[11]:

N = 50

np.random.seed(42)

x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

fig, ax = plt.subplots()

ax.scatter(x, y, s=area, c=colors, alpha=0.5)

ax.annotate(
    "First point", xy=(x[0], y[0]), xycoords="data",
    xytext=(25, -25), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.6")
)


# In[12]:

x = np.linspace(0, 1, 500)
y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

fig, ax = plt.subplots()

ax.grid(True)
ax.fill(x, y)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



