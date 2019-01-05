# -*- coding: utf-8 -*-
"""
@author: kishite
"""
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 

import seaborn as sns

from sklearn import metrics

from scipy.stats import kstest

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc

#%matplotlib inline

#Load data into dataframe
df_Loan_Data = pd.read_excel('LoanDataFinalVersion.xlsx')

##Describe the dataset
#df_Loan_Data.describe()

##Describe the dataset
#df_Loan_Data.info()

X = df_Loan_Data.loc[:, df_Loan_Data.columns != 'BadLoan']
y = df_Loan_Data.loc[:, df_Loan_Data.columns == 'BadLoan']
#Split dat into features and class labels
#X = df_Loan_Data.iloc[:, :-1].values  
#y = df_Loan_Data.iloc[:, 72].values 

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
columns = X_train.columns
print(y_train.shape)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['BadLoan'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of good loans oversampled data",len(os_data_y[os_data_y["BadLoan"]==0]))
print("No.of Bad Loans",len(os_data_y[os_data_y["BadLoan"]==1]))
print("Proportion of Non-Bankrupt data in oversampled data is ",len(os_data_y[os_data_y["BadLoan"]==0])/len(os_data_X))
print("Proportion of Bankrupt in oversampled data is ",len(os_data_y[os_data_y["BadLoan"]==1])/len(os_data_X))

#pca = PCA(n_components=72)
#pca_result = pca.fit_transform(os_data_X)
 
#print(os_data_X.shape)
#print(pca_result.shape)

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

#Split data into training and test
#X_train, X_test, y_train, y_test = train_test_split(X_train_transformed, y, test_size=0.40)  

#Perform Feature Scaling
scaler = StandardScaler()  
scaler.fit(X_train_transformed)

X_train_sc = scaler.transform(X_train_transformed)  
X_test_sc = scaler.transform(X_test_transformed)

print(y_train.shape)

#KNN Classification, fit to training data
classifier = KNeighborsClassifier(n_neighbors=5)  
#classifier.fit(X_train, y_train)  
classifier.fit(X_train_sc, os_data_y) 

#Predict on test data
y_pred = classifier.predict(X_test_sc)

#Evaluate with confusion matrix and classification report
print("Accuracy: ", classifier.score(X_test_sc,y_test))

#Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,linewidths=0.5,fmt="d")
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

y_score = metrics.accuracy_score(y_test, y_pred)

# Compute ROC curve and ROC area for each class
y_scores = classifier.predict_proba(X_test_sc)
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
KS=kstest(classifier.predict(X_test_sc),'norm')
print(KS)

# try K=1 through K=25 and record testing accuracy
k_range = range(1, 26)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sc, np.ravel(os_data_y))
    y_pred = knn.predict(X_test_sc)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)


#PLot KNN versus accuracy 
# allow plots to appear within the notebook
#%matplotlib inline

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

#Compre Error Rate and K-Value
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 10):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_sc, np.ravel(os_data_y))
    pred_i = knn.predict(X_test_sc)
    error.append(np.mean(pred_i != y_test))

#Plot error rate versus K-value
plt.figure(figsize=(12, 6))  
plt.plot(40, error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')
