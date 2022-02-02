#!/usr/bin/env python
# coding: utf-8

# In[212]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[213]:


dataset = pd.read_csv("diabetes.csv")


# In[214]:


dataset.shape


# In[215]:


dataset.head()


# In[216]:


dataset.tail()


# In[217]:


dataset.plot(x="Pregnancies", y="Age", style="o" )
plt.xlabel("Pregnancies")
plt.ylabel("age")
plt.show()


# In[218]:


dataset.plot(x="Glucose", y="Age", style="o" )
plt.xlabel("glucose")
plt.ylabel("age")
plt.show()


# In[219]:


dataset.plot(x="BloodPressure", y="Age", style="o" )
plt.xlabel("BloodPressure")
plt.ylabel("age")
plt.show()


# In[220]:


dataset.plot(x="SkinThickness", y="Age", style="o" )
plt.xlabel("SkinThickness")
plt.ylabel("age")
plt.show()


# In[221]:


dataset.plot(x="Insulin", y="Age", style="o" )
plt.xlabel("Insulin")
plt.ylabel("age")
plt.show()


# In[222]:


dataset.plot(x="BMI", y="Age", style="o" )
plt.xlabel("BMI")
plt.ylabel("age")
plt.show()


# In[223]:


dataset.plot(x="DiabetesPedigreeFunction", y="Age", style="o" )
plt.xlabel("DiabetesPedigreeFunction")
plt.ylabel("age")
plt.show()


# In[224]:


dataset.describe()


# In[225]:


X = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']]

y = dataset[['Age']]


# In[226]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[227]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[228]:


print(regressor.coef_)


# In[229]:


y_pred = regressor.predict(X_test)


# In[230]:


import pandas as pd
df = pd.DataFrame([{'Actual': y_test, 'Predicted': y_pred}])
df


# In[231]:


df.head()


# In[232]:


from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[ ]:




