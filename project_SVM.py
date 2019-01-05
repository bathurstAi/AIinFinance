# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:52:21 2019

@author: kishite
"""
from sklearn.preprocessing import StandardScaler 
#from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import numpy as np  

import matplotlib.pyplot as plt  
import pandas as pd 

import seaborn as sns

from sklearn import metrics

from scipy.stats import kstest

from sklearn.decomposition import PCA  
from sklearn.metrics import classification_report, confusion_matrix  
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc

#Load data into dataframe
df_Loan_Data = pd.read_excel('LoanDataFinalVersion.xlsx')

#Describe the dataset
df_Loan_Data.describe()

#Describe the dataset
df_Loan_Data.info()

X = df_Loan_Data.loc[:, df_Loan_Data.columns != 'BadLoan']
y = df_Loan_Data.loc[:, df_Loan_Data.columns == 'BadLoan']
#Split dat into features and class labels
#X = df_Loan_Data.iloc[:, :-1].values  
#y = df_Loan_Data.iloc[:, 72].values 

os = SMOTE(random_state=0)

#Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)  
columns = X_train.columns
print(y_train.shape)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['BadLoan'])

#plot number of compnents versus culmative variance
pca = PCA(72)
pca_full = pca.fit(os_data_X)

plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('# of components')
plt.ylabel('Cumulative explained variance')

print(pca.explained_variance_ratio_)

pca = PCA(n_components=10)
X_train_transformed = pca.fit_transform(os_data_X)
X_test_transformed = pca.transform(X_test)

#Perform Feature Scaling
scaler = StandardScaler()  
scaler.fit(X_train_transformed)

X_train_sc = scaler.transform(X_train_transformed)  
X_test_sc = scaler.transform(X_test_transformed)

# Create SVM classification object 
model = SVC(kernel='linear', C=1, gamma=1).fit(X_train_sc, os_data_y) 

# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.score(X_train_sc, os_data_y)

#Predict Output
predicted= model.predict(X_test_sc)

#Evaluate with confusion matrix and classification report
print("Accuracy: ", model.score(X_test_sc,y_test))

#Confusion Matrix
cm = confusion_matrix(y_test,predicted)
sns.heatmap(cm,annot=True,linewidths=0.5,fmt="d")
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()

print(confusion_matrix(y_test, predicted))  
print(classification_report(y_test, predicted)) 

y_score = metrics.accuracy_score(y_test, predicted)

# Compute ROC curve and ROC area for each class
y_scores = model.predict_proba(X_test_sc)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

#ROC
print(roc_auc)

#GINI
GINI = (2 * roc_auc) - 1
print(GINI)

#KS STAT
KS=kstest(model.predict(X_test_sc),'norm')
print(KS)