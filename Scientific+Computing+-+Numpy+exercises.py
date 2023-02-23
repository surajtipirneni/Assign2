
# coding: utf-8

# In[3]:

import numpy as np


# In[1]:

#Excercise 1


# In[3]:

x_3d_list = [[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]]
x_3d = np.array(x_3d_list)
print(x_3d)


# In[8]:

print(x_3d[1,1,2])


# In[ ]:

#Excercise 2


# In[9]:

x_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x_2d)


# In[ ]:

# the inner most index correspods to row and the outermost index is columns 
# This is demonstrated below


# In[13]:

# Inner most index
x_2d[0]


# In[14]:

# Outer most index
x_2d[0]


# In[15]:

#Combination
# Inner most index
x_2d[0,1]


# In[ ]:

#Excercise 3


# In[4]:

ar = np.array([[5, 6], [50, 60]])


# In[7]:

# By extract, I am assuming slicing the array, if not, please provide clearer instructions next time.

# To slice thise array or to extract elements from this array, I have to first create the array. Second, call upon the
    #respective indices to extract the elements required as shown in the examples below.

print(ar[0,:])
print(ar[:,0])


# In[ ]:

#Excercise 4


# In[11]:

# Multiplying lists with integers results in duplicating the list elements and appending the duplicate list to the original
    #It is demonstrated below
a = [1,2,3]
a1 = a*2
print(a1)


# In[13]:

# Multiplying array with integer results in actually multiplying the elements of the array with the integer.
    #Ex:
ar2= ar*2
ar2


# In[ ]:

#Excercise 5


# In[54]:

i = 0.03
M = 100
C = 5

#Defining maturities array 'N' - 1 to 10
N = np.arange(1,11)
Maturities = np.size(N)


# In[84]:

#Bond Prices
p = np.zeros(Maturities)

for j in range(0,Maturities):
    p[j] = C * ((1-pow((1+i),-N[j]))/i) + M * pow(1+i,-N[j])
    print('for Maturity ' + str(N[j]) + ', Price = ' + str(p[j]))


# In[ ]:



