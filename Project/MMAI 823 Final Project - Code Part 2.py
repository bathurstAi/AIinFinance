#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 23:19:13 2018

@author: hasham.javaid@ibm.com
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import missingno as msno 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import linkage, dendrogram 
from sklearn.preprocessing import normalize 
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

#Lets import the clean dataset
df = pd.read_csv('/Users/hasham.javaid@ibm.com/Desktop/loanclean.csv')

#Lets check out the correlations between the features and the 'BadLoan' variable
corr = df.corr()
x=corr["BadLoan"].sort_values(ascending=False)
print(x) 

#Dropping features that have less then a 1% correlation with the 'BadLoan' variable 
df.drop(['dti', 'purpose_wedding', 'purpose_house', 'home_ownership_OTHER', 'emp_length_< 1 year',
'purpose_medical', 'emp_length_7 years', 'application_type_INDIVIDUAL'], axis=1, inplace=True)

df.drop(['emp_length_5 years', 'pymnt_plan_y', 'purpose_renewable_energy', 'sub_grade_C4', 
'grade_C', 'funded_amnt', 'delinq_2yrs', 'emp_length_1 year'], axis=1, inplace=True)

df.drop(['purpose_vacation', 'home_ownership_NONE', 'emp_length_4 years', 'emp_length_3 years', 
'sub_grade_C3', 'emp_length_9 years', 'emp_length_2 years', 'funded_amnt_inv'], axis=1, inplace=True)

df.drop(['acc_now_delinq', 'home_ownership_ANY', 'purpose_major_purchase', 'emp_length_8 years', 
'tot_coll_amt', 'purpose_car', 'sub_grade_C2', 'home_ownership_OWN'], axis=1, inplace=True)

df.drop(['pymnt_plan_n', 'application_type_JOINT', 'collections_12_mths_ex_med', 'purpose_home_improvement', 
'sub_grade_C1', 'mths_since_last_delinq', 'sub_grade_B5', 'open_acc'], axis=1, inplace=True)

df.drop(['sub_grade_B4', 'mths_since_last_major_derog', 'total_acc', 'revol_bal', 
'emp_length_10+ years', 'sub_grade_B3', 'verification_status_Source Verified', 'verification_status_Not Verified'], axis=1, inplace=True)

df.drop(['sub_grade_B2', 'sub_grade_B1', 'sub_grade_A3', 'home_ownership_MORTGAGE', 
'annual_inc', 'sub_grade_A2', 'sub_grade_A4', 'sub_grade_A5'], axis=1, inplace=True)

df.drop(['purpose_credit_card', 'term_ 36 months', 'sub_grade_A1', 'total_rev_hi_lim', 
'total_pymnt', 'total_pymnt_inv', 'tot_cur_bal', 'grade_B'], axis=1, inplace=True)

df.drop(['grade_A', 'initial_list_status_w', 'total_rec_prncp', 'last_pymnt_amnt', 
'out_prncp_inv', 'out_prncp'], axis=1, inplace=True)

#Whats the shape of our new dataset? 
print('Data rows:',df.shape[0], 'Columns:',df.shape[1])

#Lets normalize the data 
normalized_data = normalize(df)

#Whats the optimal number of clusters? 
ks = range(1,10)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(normalized_data)
    inertias.append(model.inertia_)
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#PCA 
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(normalized_data)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA Feature')
plt.show()

real_pca = PCA(n_components=2)
real_pca.fit(normalized_data)
transformed_data = real_pca.transform(normalized_data)
print(transformed_data.shape)




