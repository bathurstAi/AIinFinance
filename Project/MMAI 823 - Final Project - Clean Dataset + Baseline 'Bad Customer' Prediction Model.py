#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:24:25 2018

@author: hasham.javaid@ibm.com
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import missingno as msno 
from keras.layers import Dense 
from keras.models import Sequential
from keras.callbacks import EarlyStopping 
pd.set_option('display.max_columns', 500)

#Read Data
data_df = pd.read_csv('loan.csv')

#How Many Observations are NaN?
total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
PercentofNaNValues= pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

#One-Hot Encoding of Categorical Features 
onehot_columns = data_df[['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 
 'pymnt_plan', 'purpose', 'initial_list_status', 'application_type']]
onehot_values = pd.get_dummies(onehot_columns)
data_df = pd.concat([data_df, onehot_values], axis=1)

#Dropping columns with more then 75% NaN Values 
data_df.drop(['mths_since_last_record', 'inq_last_12m', 'inq_fi', 'open_acc_6m', 
'open_il_6m', 'open_il_12m', 'open_il_24m', 'total_bal_il'],axis=1,inplace=True) 

data_df.drop(['total_cu_tl', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 
'all_util', 'mths_since_rcnt_il', 'il_util', 'annual_inc_joint'],axis=1,inplace=True) 

data_df.drop(['dti_joint', 'verification_status_joint'],axis=1,inplace=True) 

#Dropping other features that may not be useful
data_df.drop(['id', 'member_id', 'emp_title', 'issue_d', 'pymnt_plan', 'url', 'purpose',
'title'],axis=1,inplace=True)

data_df.drop(['zip_code', 'addr_state', 'earliest_cr_line', 'pub_rec', 'initial_list_status', 
'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'],axis=1,inplace=True)

data_df.drop(['policy_code', 'application_type', 'loan_amnt', 'desc'],axis=1,inplace=True)

#Dropping features that have been one-hot encoded  
data_df.drop(['verification_status', 'grade', 'sub_grade', 'emp_length', 'home_ownership','term'],axis=1,inplace=True)

print('Data rows:',data_df.shape[0], 'Columns:',data_df.shape[1])

#Replace NaN values in mths_since_last_major_derog column with 0 
data_df['mths_since_last_major_derog'] = data_df['mths_since_last_major_derog'].fillna(0)

#Replace NaN values in mths_since_last_delinq column with median 
delinq_mnths_median = data_df['mths_since_last_delinq'].median()
data_df['mths_since_last_delinq'] = data_df['mths_since_last_delinq'].fillna(delinq_mnths_median)

#Replace NaN values in Annual Income column with median
income_median = data_df['annual_inc'].median()
data_df['annual_inc'] = data_df['annual_inc'].fillna(income_median)

#Replace NaN values in 'delinq_2yrs' column with 0
data_df['delinq_2yrs'] = data_df['delinq_2yrs'].fillna(0)

#Replace NaN values in 'inq_last_6mths' column with 0
data_df['inq_last_6mths'] = data_df['inq_last_6mths'].fillna(0)

#Replace NaN values in 'open_acc' column with median 
openaccounts_median = data_df['open_acc'].median()
data_df['open_acc'] = data_df['open_acc'].fillna(openaccounts_median)

#Replace NaN values in 'revol_util' column with median 
revolutil_median = data_df['revol_util'].median()
data_df['revol_util'] = data_df['revol_util'].fillna(revolutil_median)

#Replace NaN values in 'total_acc' column with median 
acc_median = data_df['total_acc'].median()
data_df['total_acc'] = data_df['total_acc'].fillna(acc_median)

#Replace NaN values in 'collections_12_mths_ex_med' column with 0 
data_df['collections_12_mths_ex_med'] = data_df['collections_12_mths_ex_med'].fillna(0)

#Replace NaN values in 'acc_now_delinq' column with 0 
data_df['acc_now_delinq'] = data_df['acc_now_delinq'].fillna(0)

#Replace NaN values in tot_coll_amt' column with median 
collamt_median = data_df['tot_coll_amt'].median()
data_df['tot_coll_amt'] = data_df['tot_coll_amt'].fillna(collamt_median)

#Replace NaN values in 'tot_cur_bal' column with median 
curbal_median = data_df['tot_cur_bal'].median()
data_df['tot_cur_bal'] = data_df['tot_cur_bal'].fillna(curbal_median)

#Replace NaN values in 'total_rev_hi_lim' column with median 
revlimit_median = data_df['total_rev_hi_lim'].median()
data_df['total_rev_hi_lim'] = data_df['total_rev_hi_lim'].fillna(revlimit_median)

#Bad Customer Definition 
data_df['BadLoan'] = np.where(np.isin(data_df['loan_status'],['Charged Off','Default','Late (31-120 days)', 
                                    'In Grace Period', 'Late (16-30 days)',
                                   'Does not meet the credit policy. Status:Charged Off']), 1, 0)
data_df.drop(['loan_status'],axis=1,inplace=True) 

#Lets see if there are any missing values left
plt.figure(figsize=(16,6))
msno.matrix(data_df,labels = True, color = (0.2,0.15,0.45))

#Correlation Matrix for new dataset 
fig,ax = plt.subplots(figsize =(8,8))
corr = data_df.corr()

mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr,mask=mask,square = False, linewidths = .5,cbar_kws={"shrink": .5})

#Lets assign input and output values to the data 
y = data_df['BadLoan'].values 
x = data_df.drop(['BadLoan'], axis=1).values 
input_cols = x.shape[1]
model_inputvalue = (input_cols,)

#Lets build the model! 
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = model_inputvalue))
model.add(Dense(1))

#Lets Compile the model! 
model.compile(optimizer='adam', loss='mean_squared_error')

#Lets define training time, and then fit the model! 
early_stopping_monitor = EarlyStopping(patience = 3)
model.fit(x, y, validation_split=0.3, epochs=30, callbacks = [early_stopping_monitor])





