#!/usr/bin/env python
# coding: utf-8

# ## Regression Project
# 
# #### Edit By : Mahmoud Sabry
# 
# 
# ## 1) importing libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns' , None)
import seaborn as sns
sns.set(style = 'whitegrid')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest , SelectPercentile , f_classif
import os
os.chdir(r'I:\machine-learning2\projects\first regression')


# ## 2) Read Data

# In[2]:


data = pd.read_csv('Ecommerce Customers.csv')
data.head()


# ## 3) Data Analysis

# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


lb = LabelEncoder()
new_email = lb.fit_transform(data['Email'] )
new_email  = pd.DataFrame(new_email , columns  = {'New_Email'})
new_email


# In[7]:


lb = LabelEncoder()
New_Address = lb.fit_transform(data['Address'])
New_Address = pd.DataFrame(New_Address , columns = {'New_Address'})
New_Address


# In[8]:


lb = LabelEncoder()
New_Avatar = lb.fit_transform(data['Avatar'])
New_Avatar = pd.DataFrame(New_Avatar , columns = {'New_Avatar'})
New_Avatar


# In[9]:


data = pd.concat([data ,new_email , New_Address , New_Avatar] , axis = 1)
data.head()


# In[10]:


data = data.drop(['Email' , 'Address' , 'Avatar'] , axis = 1)
data.head()


# ## 4) Data cleaning

# In[11]:


data.isna().sum()


# ## 5) Outliers

# In[12]:


data.boxplot('New_Email')
plt.show()


# In[13]:


data.boxplot('New_Address')
plt.show()


# In[14]:


data.boxplot('New_Avatar')
plt.show()


# In[15]:


data.boxplot('Length of Membership')
plt.show()


# In[16]:


data.boxplot('Avg. Session Length')
plt.show()


# In[17]:


data.boxplot('Time on App')
plt.show()


# In[18]:


data.boxplot('Time on Website')
plt.show()


# In[19]:


data.boxplot('Yearly Amount Spent')
plt.show()


# In[20]:


data.info()


# In[ ]:





# In[21]:


data.info()


# In[23]:


for i in ('Avg. Session Length' , 'Time on App' , 'Time on Website' , 'Length of Membership' , 'Yearly Amount Spent' , 'New_Avatar' , 'New_Email' , 'New_Address'):
    q75 , q25 = np.percentile(data.loc[:,i] , [75 , 25])
    intr_qr = q75-q25
    max = q75+(1*intr_qr)
    min = q25-(1*intr_qr)
    print(f'for {i} has min outliers {data.loc[data[i]<min , i].shape[0]} rows and max has {data.loc[data[i]>max , i].shape[0]} rows')
    data.loc[data[i]<min ,i] = np.nan
    data.loc[data[i]>max , i] = np.nan

data = data.dropna(axis = 0)
data.reset_index(inplace = True)
data.drop(['index'] , axis = 1 , inplace = True)


# ## 6) Feature Extraction

# In[24]:


for i in ('Avg. Session Length' , 'Time on App' , 'Time on Website' , 'Length of Membership' , 'Yearly Amount Spent' , 'New_Avatar' , 'New_Email' , 'New_Address'):
    q75 , q25 = np.percentile(data.loc[:,i] , [75,25])
    intr_qr = q75-q25
    max = q75+(1*intr_qr)
    min = q25-(1*intr_qr)
    print(f'for {i} has min outliers {data[data.loc[i]<min , i].shape[0]} rows and max has {data[data.loc[i]>max , i].shape[0]} rows')


# ## 7)Feature Selection

# In[28]:


x = data.drop(['Yearly Amount Spent'] , axis = 1)
y = data['Yearly Amount Spent']
x


# In[29]:


y


# In[40]:


FeatureSelection = SelectPercentile(score_func = f_classif , percentile = 0.05)
x_Selected = FeatureSelection.fit_transform(x,y)
NewData = pd.DataFrame(x_Selected , columns = [i for i,j in zip(x.columns , FeatureSelection.get_support()) if j])
NewData


# In[42]:


FeatureSelection = SelectKBest(score_func = f_classif , k = 5)
x_Selected = FeatureSelection.fit_transform(x,y)
NewData = pd.DataFrame(x_Selected , columns = [i for i,j in zip(x.columns , FeatureSelection.get_support()) if j])
NewData


# ## 8)Visualization The Data

# In[49]:


sns.jointplot(x = 'Time on Website' , y = 'Yearly Amount Spent' , data =data ,kind = 'scatter')
plt.show()


# In[51]:


sns.jointplot(x = 'Length of Membership' , y = 'Yearly Amount Spent' , data = data , kind = 'scatter' )
plt.show()


# In[52]:


sns.jointplot(x = 'New_Email' , y = 'Yearly Amount Spent' , data = data , kind = 'scatter' )
plt.show()


# In[53]:


sns.jointplot(x = 'New_Address' , y = 'Yearly Amount Spent' , data = data , kind = 'scatter' )
plt.show()


# In[54]:


sns.jointplot(x = 'New_Avatar' , y = 'Yearly Amount Spent' , data = data , kind = 'scatter' )
plt.show()


# In[56]:


sns.pairplot(data)
plt.show()


# ## 9) Building Model

# In[58]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size = 0.2 , random_state = 33)
print('x_train shape is : ' , x_train.shape)
print('y_train shape is : ' , y_train.shape)
print('x_test shape is : ' , x_test.shape)
print('y_test shape is : ' , y_test.shape)


# In[90]:


from sklearn.ensemble import RandomForestRegressor
RandomForestModel = RandomForestRegressor(criterion = 'squared_error' , n_estimators = 500  , max_depth = 30 , random_state = 33)
RandomForestModel.fit(x_train , y_train)
y_pred = RandomForestModel.predict(x_test)


# In[91]:


from sklearn.metrics import mean_absolute_error , mean_squared_error
MAE = mean_absolute_error(y_test , y_pred)
MSE = mean_squared_error(y_test , y_pred)
RMSE = np.sqrt(mean_squared_error(y_test , y_pred))

print('MAE : ' , MAE)
print('MSE : ' , MSE)
print('RMSE : ' , RMSE)


# In[92]:


sns.distplot((y_test-y_pred),bins=50)
plt.show()


# In[74]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor


# In[108]:


LinearModel = LinearRegression()
SVRmodel = SVR(kernel = 'rbf' , max_iter =100 ,C= 0.5 , gamma = 'auto')
RandomRegressor = RandomForestRegressor(criterion = 'squared_error' , n_estimators = 300 , max_depth = 7 , random_state = 33)
GradientRegressor = GradientBoostingRegressor(criterion = 'squared_error' , n_estimators = 300 , max_depth = 7 , random_state = 33)

Models = [LinearRegression , SVR , RandomForestRegressor ,GradientBoostingRegressor ] 


# In[113]:


ModelScoreAllData = {}
for model in Models:
    print(f'for model {str(model).split("(")[0]}')
    m = model()
    m.fit(x_train , y_train)
    print(f'train score is : {m.score(x_train , y_train)}')
    print(f'test score is : {m.score(x_test , y_test)}')
    y_pred = m.predict(x_test)
    MAE = mean_absolute_error(y_test , y_pred)
    MSE = mean_squared_error(y_test , y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test , y_pred))
    print('MAE : ' , MAE)
    print('MSE : ' , MSE)
    print('RMSE : ' , RMSE)


# In[ ]:




