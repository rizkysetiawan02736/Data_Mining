#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


dataset = pd.read_csv('Downloads/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[4]:


print(x)


# In[5]:


print(y)


# In[6]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


# In[7]:


print(x)


# In[9]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# In[10]:


print(x)


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[12]:


print(y)


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[14]:


print(x_train)


# In[15]:


print(x_test)


# In[16]:


print(y_train)


# In[17]:


print(y_test)


# In[19]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])


# In[21]:


print(x_train)


# In[22]:


print(x_test)


# In[ ]:




