
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score 


df = pd.read_csv("LoanDataFinalVersion.csv")


# In[15]:


df.head()


# In[16]:


def stress_update(x,stresscolumn,stressvalue):
    x[stresscolumn]= x[stresscolumn] + stressvalue
    return x

    


# In[17]:


def data_prepration(x): # preparing data for training and testing as we are going to use different data 
    #again and again so make a function
    X= x.ix[:,x.columns != "BadLoan"]
    y = x.ix[:,x.columns=="BadLoan"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    return(X_train,X_test,y_train,y_test)


# In[18]:


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


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = data_prepration(df)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,max_depth=20)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train_sc, y_train)

#Prediction
y_pred = classifier.predict(X_test_sc)




# In[20]:


roc_auc = roc_auc_score(y_test,y_pred)
print(roc_auc)
GINI = (2 * roc_auc) - 1
print(GINI)


# In[21]:


from scipy.stats import kstest
KS=kstest(y_pred,'norm')
print(KS)


# In[9]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
fpr, tpr, thresh = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.show()


# In[22]:


from sklearn.metrics import confusion_matrix

print("Accuracy: ", classifier.score(X_test_sc,y_test))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,linewidths=0.5,fmt="d")
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()
print(classification_report(y_test,y_pred))


# In[24]:


X= df.iloc[:,df.columns != "BadLoan"]
y = df.iloc[:,df.columns=="BadLoan"]
for i in range(1 , 6):
    X = stress_update(X,"int_rate",i)
    predicitons = classifier.predict(X)
    df["intrate_stressed_with_"+str(i)] = X["int_rate"]
    df["prediction_with_"+str(i)] = predicitons
    X["int_rate"] = df.iloc[:,df.columns == "int_rate"]
df.head
    


# In[25]:


df.head()


# In[26]:


#df[df["BadLoan"]==1].count()[0]
df[df["prediction_with_5"]==1].count()[0]


# In[79]:


df.to_csv("stresstest.csv")

