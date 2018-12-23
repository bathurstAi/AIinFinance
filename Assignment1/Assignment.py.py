# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:33:40 2018

@author: Muhammad Shahbaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # to split the data
from sklearn.model_selection import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import warnings
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings('ignore')




dfBank = pd.read_csv("Bankruptcy_final.csv")
nansum = dfBank.isna().sum()
summary = dfBank.describe()
info = dfBank.info()
dftest = dfBank[dfBank['BK']==1]

#Outliers Check
dfBank.boxplot(column=['EPS','Liquidity','Profitability','Productivity','Leverage Ratio'])
dfBank.boxplot(column=['Operational Margin','Return on Equity','Market Book Ratio','Assets Growth','Sales Growth'])

#Removing nan-Values from Sales Growth,No Effect on Bankruptcy
#dfBank=dfBank.loc[dfBank['Sales Growth'].isna() != True]
#nansum = dfBank.isna().sum()
#dftest = dfBank[dfBank['BK']==1]

#Correlation
corr = dfBank.iloc[:,1:].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#Unbalanced Data
# now let us check in the number of Percentage
countNonBankrupt = len(dfBank[dfBank["BK"]==0]) # normal transaction are repersented by 0
countBankrupt = len(dfBank[dfBank["BK"]==1]) # fraud by 1
PerNonBankrupt = countNonBankrupt/(countNonBankrupt+countBankrupt)
print("percentage of Non Bankruptcy is",PerNonBankrupt*100)
PerBankrupt= countBankrupt/(countBankrupt+countNonBankrupt)
print("percentage of Bankruptcy",PerBankrupt*100)


#Data Being highly unbalanced, we shall start with UnderSampling
# now lets us see the index of fraud cases
#now let us a define a function for make undersample data with different proportion
#different proportion means with different proportion of normal classes of data

#Undersampling function
def undersample(nonBankrupt_indices,Bankrupt_indices,times):
    #Randomly picking the sample from nonBankrupt data
    nonBankrupt_indices_undersample = np.array(np.random.choice(nonBankrupt_indices,(times*len(Bankrupt_indices)),replace=False))
    undersample_data= np.concatenate([Bankrupt_indices,nonBankrupt_indices_undersample])
    undersample_dataset = dfBank.ix[undersample_data,:]
    
    print("the Non Bankrupt Data proportion is :",len(undersample_dataset[undersample_dataset.BK==0])/len(undersample_dataset))
    print("the Bankrupt Data proportion is :",len(undersample_dataset[undersample_dataset.BK==1])/len(undersample_dataset))
    print("total number of record in resampled data is:",len(undersample_dataset))
    return(undersample_dataset)

#Modeling Function
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
    
#Prepare Data    
def data_prepration(x): # preparing data for training and testing as we are going to use different data 
    #again and again so make a function
    X= x.ix[:,x.columns != "BK"]
    y = x.ix[:,x.columns=="BK"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    print("length of training data")
    print(len(X_train))
    print("length of test data")
    print(len(X_test))
    print("length of validation data")
    print(len(X_val))
    
    return(X,y, X_train,X_test,y_train,y_test, X_val,y_val)
 
#OverSampling Function
def over_sample(nonBankrupt_train,Bankrupt_train,times):
    for i in range (times): 
        nonBankrupt_train= nonBankrupt_train.append(Bankrupt_train)
    os_data = nonBankrupt_train.copy() 
    print("length of oversampled data is ",len(os_data))
    print("Number of Non Bankruptcy in oversampled data",len(os_data[os_data["BK"]==0]))
    print("No.of Bankruptcy",len(os_data[os_data["BK"]==1]))
    print("Proportion of Non Bankruptcy in oversampled data is ",len(os_data[os_data["BK"]==0])/len(os_data))
    print("Proportion of Bankruptcy in oversampled data is ",len(os_data[os_data["BK"]==1])/len(os_data))
    return os_data
   
    
#Basic RF
X,y, features_train,features_test,labels_train,labels_test=data_prepration(dfBank)
clf= RandomForestClassifier(n_estimators=100)# here we are just changing classifier
clf, result = model(clf,features_train,features_test,labels_train,labels_test)
scores = cross_val_score(clf,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())
  
    
#Under Sampling    
X,y, X_train,X_test,y_train,y_test,X_val,y_val=data_prepration(dfBank)
#X_train["BK"]= y_train["BK"]
rus = RandomUnderSampler(sampling_strategy=.4,random_state=0)
X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)

y_train_undersampled = pd.DataFrame(y_train_undersampled)
clfUnder = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,max_depth=3)# here we are just changing classifier
clfUnderval, resultUnderval = model(clfUnder, X_train_undersampled, X_val, y_train_undersampled ,y_val)
clfUndertest, resultUndertest = model(clfUnder, X_train_undersampled, X_test, y_train_undersampled,y_test)

    

#Over Sampling
X,y, X_train,X_test,y_train,y_test,X_val,y_val=data_prepration(dfBank)
X_train["BK"]= y_train["BK"] # combining class with original data

print("length of training data",len(X_train))
# Now make data set of normal transction from train data
nonBankrupt_data = X_train[X_train["BK"]==0]
print("length of normal data",len(nonBankrupt_data))
Bankrupt_data = X_train[X_train["BK"]==1]
print("length of fraud data",len(Bankrupt_data))

oversampled_data6030 = over_sample(nonBankrupt_data,Bankrupt_data,62)    
oversampled_data5050 = over_sample(nonBankrupt_data,Bankrupt_data,140) 

clf6030= RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,max_depth=3)
clf5050 = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,max_depth=3)

#Training with oversampled data and testing on original data
clf6030,result6030 = model(clf6030, oversampled_data6030.iloc[:,:-1], X_test, oversampled_data6030.iloc[:,-1],y_test)
clf5050,result5050 = model(clf6030, oversampled_data5050.iloc[:,:-1], X_test, oversampled_data5050.iloc[:,-1],y_test)



#SMOTE
osSmote = SMOTE(random_state=0) #   We are using SMOTE as the function for oversampling
# now we can devided our data into training and test data
# Call our method data prepration on our dataset
X,y, X_train,X_test,y_train,y_test, X_val, y_val = data_prepration(dfBank)
osSmote_X,osSmote_y=osSmote.fit_sample(X_train,y_train)
osSmote_X = pd.DataFrame(data=osSmote_X,columns=X_train.columns )
osSmote_y= pd.DataFrame(data=osSmote_y,columns=["BK"])
# we can Check the numbers of our data
print("length of oversampled data is ",len(osSmote_X))
print("Number of Non-Bankrupt transcation in oversampled data",len(osSmote_y[osSmote_y["BK"]==0]))
print("No.of Bankrupt transcation",len(osSmote_y[osSmote_y["BK"]==1]))
print("Proportion of Non-Bankrupt data in oversampled data is ",len(osSmote_y[osSmote_y["BK"]==0])/len(osSmote_X))
print("Proportion of Bankrupt in oversampled data is ",len(osSmote_y[osSmote_y["BK"]==1])/len(osSmote_X))

#Modeling with SMOTE Data
clfSmote= RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,max_depth=3)
# train data using oversampled data and predict for the test data

smoteCLF, smoteResult_val = model(clfSmote,osSmote_X,X_val,osSmote_y,y_val)
smoteCLF_val, smoteResult_test = model(clfSmote,osSmote_X,X_test,osSmote_y,y_test)
