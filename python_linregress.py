#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the dataset
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys


dataset = pd.read_csv(sys.argv[1])

#Plotting Data set in Scatter Plot
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('dataset.png')
plt.clf()



# In[2]:


# Fitting Linear Regression to the Dataset
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


#model = LinearRegression()
#model.fit(dataset[['y']], dataset[['x']])
#plt.scatter(dataset[['y']], dataset[['x']])
#plt.show()


# In[3]:


#Visualizing the Linear Regression results

plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
#plt.plot(dataset[['y']], model.predict(dataset[['y']]), color = 'blue')
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('linearmodel.png')
plt.clf()


# In[ ]:




