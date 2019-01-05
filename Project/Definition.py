#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('loan.csv')
pd.options.display.max_columns = None
df.head()


# In[18]:


total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
PercentofNaNValues= pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

PercentofNaNValues


# In[19]:


df.drop(['dti_joint', 'verification_status_joint', 'annual_inc_joint', 'il_util', 
'mths_since_rcnt_il', 'all_util', 'max_bal_bc','url'],axis=1,inplace=True) 


"""
open_rv_24m
open_rv_12m
total_cu_tl
total_bal_il
open_il_24m
open_il_12m
open_il_6m
open_acc_6m
inq_fi
inq_last_12m
desc
mths_since_last_record
mths_since_last_major_derog
mths_since_last_delinq
"""


# In[30]:


df.head()


# In[28]:


df.describe()


# In[29]:


df.info()


# In[32]:


df['loan_status'].unique()


# In[36]:


#creating the target variable based of loan_status

df['BadLoan'] = np.where(np.isin(df['loan_status'],['Charged Off','Default','Late (31-120 days)', 
                                    'In Grace Period', 'Late (16-30 days)',
                                   'Does not meet the credit policy. Status:Charged Off']), 1, 0)
df.drop(['loan_status'],axis=1,inplace=True) 


# In[37]:


df.describe()


# In[40]:


#Correlation
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True)
fig = plt.figure(figsize = (20,20))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .8})
plt.show()


# In[72]:


#finding correlation with target variable
x=corr["BadLoan"].sort_values(ascending=False)

# plotting with variables with too many nas
from pandas.plotting import scatter_matrix
attributes = ['open_rv_24m','open_rv_12m','total_cu_tl','total_bal_il','open_il_24m','open_il_12m','open_il_6m']
attributes2=['open_acc_6m','inq_fi','inq_last_12m','mths_since_last_record','mths_since_last_major_derog','mths_since_last_delinq']


# In[65]:


scatter_matrix(df[attributes], alpha=0.5,figsize=(24, 16))


# In[66]:


scatter_matrix(df[attributes2], alpha=0.5,figsize=(24, 16))


# In[73]:


print(x)


# In[ ]:




