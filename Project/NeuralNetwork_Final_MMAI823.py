#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 18:13:56 2019

@author: hasham.javaid@ibm.com
"""

import numpy as np 
import pandas as pd 
from keras.layers import Dense 
from keras.models import Sequential
from keras.callbacks import EarlyStopping 
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
from keras.layers import Dropout 
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import normalize


real_data = new_data.drop(['BadLoan'], axis=1)
minmaxscaler = MinMaxScaler()
final_data = minmaxscaler.fit_transform(real_data)
X_imputed_df = pd.DataFrame(final_data, columns = real_data.columns)

#Lets assign input and output values to the data 
output_value = new_data['BadLoan'].values
input_value = X_imputed_df.values
X_train,X_test,y_train,y_test = train_test_split(input_value, output_value, test_size = 0.25, random_state = 0)

#SMOTE to deal with imbalanced dataset 
Smt = SMOTE(random_state=0) 
Smote_X,Smote_Y=Smt.fit_sample(X_train,y_train)
Smote_labels = Smote_Y
Smote_y = to_categorical(Smote_Y)
input_cols = Smote_X.shape[1]
input_layer = (input_cols,)

#Lets build the model! 
Neural_Network = Sequential()
Neural_Network.add(Dense(200, activation='relu', input_shape = input_layer))
Neural_Network.add(Dense(150, activation='relu'))
Neural_Network.add(Dense(120, activation='relu'))
Neural_Network.add(Dense(2, activation = 'softmax'))

#Lets Compile the model! 
Neural_Network.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#Lets define training time, and then fit the model! 
early_stopping_monitor = EarlyStopping(patience = 3)
Neural_Network.fit(Smote_X, Smote_y , validation_split = 0.25, epochs=20, callbacks = [early_stopping_monitor])
predicting = Neural_Network.predict(X_test) 

probas = predicting[:,1]
labels = (probas > 0.5).astype(np.int)

cnf_matrix=confusion_matrix(y_test,labels)
sns.heatmap(cnf_matrix,annot=True,linewidths=0.5,fmt="d")
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()

roc_auc = roc_auc_score(y_test, labels)
print(roc_auc)
GINI = (2 * roc_auc) - 1
print(GINI)
print(classification_report(y_test, labels))