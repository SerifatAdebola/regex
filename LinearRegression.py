#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy 
import sklearn 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
import sys

# In[3]:


data= pd.read_csv(sys.argv[1])
data.plot.scatter(x='x', y='y', title="Scatter plot")
plt.show()
plt.savefig('py_orig2.png')

# In[4]:


reg= sklearn.linear_model.LinearRegression()
X= data['x'].values.reshape(-1,1)
y= data['y'].values.reshape(-1,1)
X_train=X
X_test=X
y_train=y
y_test=y
regressor =LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[6]:


plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.show()
plt.savefig('py_lm.png')


# In[ ]:




