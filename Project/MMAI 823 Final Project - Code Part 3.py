#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 16:42:35 2018

@author: hasham.javaid@ibm.com
"""
#KMeans Clustering 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import missingno as msno 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 

#KMeans Clustering 
kmeans = KMeans(n_clusters=4)
kmeans.fit(transformed_data)
k_means_labels = kmeans.predict(transformed_data)
print(k_means_labels)

#Lets visualize our clusters 
plt.scatter(transformed_data[k_means_labels == 0,0], transformed_data[k_means_labels == 0,1], s=25, c='red', label='Cluster 1')
plt.scatter(transformed_data[k_means_labels == 1,0], transformed_data[k_means_labels == 1,1], s=25, c='green', label='Cluster 2')
plt.scatter(transformed_data[k_means_labels == 2,0], transformed_data[k_means_labels == 2,1], s=25, c='magenta', label='Cluster 3')
plt.scatter(transformed_data[k_means_labels == 3,0], transformed_data[k_means_labels == 3,1], s=25, c='cyan', label='Cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=25, c='blue', label='Centroid')
plt.title('KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

#Including the clustering labels in the dataset 
kmeans_labels = pd.Series(k_means_labels)
new_data = pd.concat([data_df,kmeans_labels], axis=1)

#Dropping grade and sub_grade column 
new_data.drop(['grade_D', 'grade_E', 'grade_F', 'grade_G', 
'sub_grade_C5', 'sub_grade_D1', 'sub_grade_D2' ], axis=1, inplace=True)

new_data.drop(['sub_grade_D3', 'sub_grade_D4', 'sub_grade_D5', 'sub_grade_E1',
'sub_grade_E2', 'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5'], axis=1, inplace=True)

new_data.drop(['sub_grade_F1', 'sub_grade_F2', 'sub_grade_F3', 'sub_grade_F4',
'sub_grade_F5', 'sub_grade_G1', 'sub_grade_G2', 'sub_grade_G3'], axis=1, inplace=True)

new_data.drop(['sub_grade_G4', 'sub_grade_G5', 'grade_A', 'grade_B', 'grade_C',
'sub_grade_A1', 'sub_grade_A2', 'sub_grade_A3'], axis=1, inplace=True)

new_data.drop(['sub_grade_A4', 'sub_grade_A5','sub_grade_B1', 'sub_grade_B2', 
'sub_grade_B3','sub_grade_B4', 'sub_grade_B5', 'sub_grade_C1'], axis=1, inplace=True)

new_data.drop(['sub_grade_C2', 'sub_grade_C3','sub_grade_C4'], axis=1, inplace=True)

#Lets rename the column containing the cluster labels 
new_data.rename(columns={'0':'Cluster'},inplace = True) 

#Lets see if there are any missing values left
plt.figure(figsize=(16,6))
msno.matrix(new_data,labels = True, color = (0.2,0.15,0.45))

#Correlation Matrix for new dataset 
fig,ax = plt.subplots(figsize =(8,8))
corr = new_data.corr()

mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr,mask=mask,square = False, linewidths = .5,cbar_kws={"shrink": .5})