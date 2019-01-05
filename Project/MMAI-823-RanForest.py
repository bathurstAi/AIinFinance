
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score 


df = pd.read_csv("LoanDataFinalVersion.csv")



# In[3]:


def data_prepration(x): # preparing data for training and testing as we are going to use different data 
    #again and again so make a function
    X= x.ix[:,x.columns != "BadLoan"]
    y = x.ix[:,x.columns=="BadLoan"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    return(X_train,X_test,y_train,y_test)


# In[4]:


def model(model,features_train,features_test,labels_train,labels_test):
    clf= model
    clf.fit(features_train,labels_train.values.ravel())
    pred=clf.predict(features_test)
    
    cnf_matrix=confusion_matrix(labels_test,pred)
    #print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    #print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
    #print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal
    #print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
    #print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal
    sns.heatmap(cnf_matrix,annot=True,linewidths=0.5,fmt="d")
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print(cnf_matrix)
    return clf,classification_report(labels_test,pred)


# In[4]:


from sklearn.model_selection import train_test_split 
osSmote = SMOTE(random_state=0) #   We are using SMOTE as the function for oversampling
# now we can devided our data into training and test data
# Call our method data prepration on our dataset
X_train,X_test,y_train,y_test = data_prepration(df)
osSmote_X,osSmote_y=osSmote.fit_sample(X_train,y_train)
osSmote_X = pd.DataFrame(data=osSmote_X,columns=X_train.columns )
osSmote_y= pd.DataFrame(data=osSmote_y,columns=["BadLoan"])
# we can Check the numbers of our data
print("length of oversampled data is ",len(osSmote_X))
print("Number of Non-Default transcation in oversampled data",len(osSmote_y[osSmote_y["BadLoan"]==0]))
print("No.of Default transcation",len(osSmote_y[osSmote_y["BadLoan"]==1]))
print("Proportion of Non-Default data in oversampled data is ",len(osSmote_y[osSmote_y["BadLoan"]==0])/len(osSmote_X))
print("Proportion of Default in oversampled data is ",len(osSmote_y[osSmote_y["BadLoan"]==1])/len(osSmote_X))


# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(osSmote_X)
X_test_sc = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,max_depth=20)
classifier.fit(X_train_sc, osSmote_y)

#Prediction
y_pred = classifier.predict(X_test_sc)




# In[6]:


roc_auc = roc_auc_score(y_test,y_pred)
print(roc_auc)
GINI = (2 * roc_auc) - 1
print(GINI)


# In[7]:


from scipy.stats import kstest
KS=kstest(y_pred,'norm')
print(KS)


# In[8]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
fpr, tpr, thresh = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.show()


# In[9]:


from sklearn.metrics import confusion_matrix

print("Accuracy: ", classifier.score(X_test_sc,y_test))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,linewidths=0.5,fmt="d")
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()
print(classification_report(y_test,y_pred))


# In[26]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())

