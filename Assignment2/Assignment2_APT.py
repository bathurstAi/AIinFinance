# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:58:34 2018

@author: kishite
"""

import tensorflow as tf
import keras

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import the SGD optimizer
#from keras.optimizers import SGD
from keras import optimizers
import numpy as np

#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.pipeline import Pipeline
import seaborn as sb

RANDOM_SEED = 7

#read in data
dfA2_train = pd.read_excel('A2trainData_MMAI.xlsx')
dfA2_test = pd.read_excel('A2testData_MMAI.xlsx')

#correlation between features
C_mat = dfA2_train.corr()
fig = plt.figure(figsize = (10,10))
sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()

#create matrix of features and target variable 94859 rows
data_X = dfA2_train.iloc[:, 1:16].values
data_Y = dfA2_train.iloc[:, -1].values
data_X_test = dfA2_train.iloc[:, 1:16].values
data_Y_test = dfA2_train.iloc[:, -1].values
#dfA2_train= dfA2_train.replace(0, np.nan)
#dfA2_test= dfA2_test.replace(0, np.nan)

#only replace return and output return columns
dfA2_train['Returns']= dfA2_train['Returns'].replace(0, np.nan)
dfA2_test['Returns']= dfA2_test['Returns'].replace(0, np.nan)

#SAME for Output Return %
dfA2_train['Output Return %']= dfA2_train['Output Return %'].replace(0, np.nan)
dfA2_test['Output Return %']= dfA2_test['Output Return %'].replace(0, np.nan)


#drop rows with zeros 91709 rows (94859 - 91709 = 3150 rows)
dfA2_train_d=dfA2_train.dropna(how='any', axis=0)
dfA2_test_d=dfA2_test.dropna(how='any', axis=0)
#dfA22
#data_X_2 = dfA22.iloc[:, 1:16].values
#data_Y_2 = dfA22.iloc[:, -1].values

#normalize data (scale data), notice that 
#mean_returns = dfA2_train_d['Returns'].mean(axis=0)
#std_returns = dfA2_train_d['Returns'].std(axis=0)
#dfA2_train_d['Returns'] = (dfA2_train_d['Returns'] - mean_returns) / std_returns
##data_Y_3 = (data_Y_2 - mean) / std
#mean_output_return = dfA2_train_d['Output Return %'].mean(axis=0)
#std_output_return = dfA2_train_d['Output Return %'].std(axis=0)
#dfA2_train_d['Output Return %'] = (dfA2_train_d['Output Return %'] - mean_output_return) / std_output_return
#
#mean_returns = dfA2_test_d['Returns'].mean(axis=0)
#std_returns = dfA2_test_d['Returns'].std(axis=0)
#dfA2_test_d['Returns'] = (dfA2_test_d['Returns'] - mean_returns) / std_returns
#data_Y_3 = (data_Y_2 - mean) / std
#mean_output_return = dfA2_train_d['Market Book Ratio'].mean(axis=0)
#std_output_return = dfA2_train_d['Market Book Ratio'].std(axis=0)
#dfA2_train_d['Market Book Ratio'] = (dfA2_train_d['Market Book Ratio'] - mean_output_return) / std_output_return
#
#mean_output_return = dfA2_test_d['Market Book Ratio'].mean(axis=0)
#std_output_return = dfA2_test_d['Market Book Ratio'].std(axis=0)
#dfA2_test_d['Market Book Ratio'] = (dfA2_test_d['Market Book Ratio'] - mean_output_return) / std_output_return

#normalize data (scale data),_n notice that 
#mean = dfA2_train_d.mean(axis=0)
#std = dfA2_train_d.std(axis=0)
#dfA2_train_d = (dfA2_train_d - mean) / std
#
##normalize data (scale data),_n notice that 
#mean = dfA2_test_d.mean(axis=0)
#std = dfA2_test_d.std(axis=0)
#dfA2_test_d = (dfA2_test_d - mean) / std
##data_Y_3 = (data_Y_2 - mean) / std
#
##normalize data (scale data),_n notice that only scale features
#mean = dfA2_train_d.mean(axis=0)
#std = dfA2_train_d.std(axis=0)
#dfA2_train_d = (dfA2_train_d - mean) / std
#
##normalize data (scale data),_n notice that only scale features
#mean_t = dfA2_test_d.mean(axis=0)
#std_t = dfA2_test_d.std(axis=0)
#dfA2_test_d = (dfA2_test_d - mean_t) / std_t
##data_Y_3 = (data_Y_2 - mean) / std

#Section to data and label
data_X_2 = dfA2_train_d.iloc[:, 1:16].values
data_Y_2 = dfA2_train_d.iloc[:, -1].values
data_X_3 = dfA2_test_d.iloc[:, 1:16].values
data_Y_3 = dfA2_test_d.iloc[:, -1].values





#Split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data_X_2, data_Y_2, test_size = 0.2, random_state=RANDOM_SEED)

#instanstiate optimizers
#use sgd optimizer
def optimize(op, lr):
    if op == 0:
        model_opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif op == 1:
        #rms prop
        model_opt=keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
    elif op == 2:
        #Adagrad
        model_opt=keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
    elif op == 3:
        #Adadelta
        model_opt=keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
    elif op == 4:
        #Adam
        model_opt=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif op == 5:
        #Adamax
        model_opt=keras.optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif op == 6:
        #Nadam
        model_opt=keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    return model_opt


#define model
def b_model():
#    model = Sequential()
#    #Add the input layer and first hidden layer, use ReLu for the activation
#    model.add(Dense(128, init = 'normal', activation = 'sigmoid', input_dim = 15))
#    #Add the first hidden layer
#    model.add(Dense(128, init = 'normal', activation = 'sigmoid'))
#    #Add the second hidden layer
#    model.add(Dense(128, init = 'normal', activation = 'sigmoid'))
#    #Add the third hidden layer
#    model.add(Dense(128, init = 'normal', activation = 'sigmoid'))
#    #Add output layer
#    model.add(Dense(1, init= 'normal'))
    model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.sigmoid,
                       input_shape=(X_train.shape[1],)),
    keras.layers.ActivityRegularization(l1=0.01, l2=0.001),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    Dropout(0.25),
    keras.layers.Dense(1)
  ])
    #Compile model
    #model.compile(optimizer = rms, loss = 'mean_squared_error', metrics=['mae'])
   
    return model
#x = 1
#while (x < 950):
model = b_model()
model.summary()
#    x +=1

# Create list of learning rates: lr_to_test 
lr_to_test = [0.000001, 0.001, 0.01, 0.002, 1]
# Create list of optimizers
opt = [0,1,2,3,4,5,6]
result_array = np.array([])
# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    # loop over all optimizers
    for op in opt:
        print('\n\nTesting model with optimizer: %f\n'%op )
        # Create optimizer with specified learning rate: my_optimizer
        my_optimizer = optimize(op, lr)
    
        # Compile the model
        model.compile(optimizer=my_optimizer, loss='mean_squared_error', metrics=['mae'])
    
        #define Checkpoint callback
        checkpoint_name = 'test\Weights--{epoch:03d}-{val_loss:.5f}.hdf5' 
        #checkpoint_name = 'test\weights.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
        callbacks_list = [checkpoint]
        result_array = np.append(result_array, checkpoint)

# evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=20, verbose=0)
#
#kfold = KFold(n_splits=10, random_state=RANDOM_SEED)
#results = cross_val_score(estimator, X_train, y_train, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

        # The patience parameter is the amount of epochs to check for improvement
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        
    
        #Fitting model
        history=model.fit(X_train, y_train, validation_split = 0.2, batch_size= 20, epochs= 1000, callbacks=[checkpoint, early_stop]) 
        result_array = np.append(result_array, model.save_weights('final weights123123.hdf5'))
        # plot validation and training loss
#        plt.plot(history.history['loss'], label = 'loss')
#        plt.plot(history.history['val_loss'], label = 'val_loss')
#        plt.xlabel('Epoch')
#        plt.ylabel('Loss')
#        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
        #Predict test results
        y_pred = model.predict(X_test).flatten()
        #y_pred = (y_pred + mean)*std 
        #print confusion_matrix
        #print(confusion_matrix(y_test, y_pred))
        #y_test = (y_test + mean)*std
        
        def plot_history(history):
          plt.figure()
          plt.xlabel('Epoch')
          plt.ylabel('Mean Abs Error')
          plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
                   label='Train Loss')
        #  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
        #           label = 'loss')
          plt.legend()
          plt.ylim([0, 5])
        
        plot_history(history)
        
        [loss, mae] = model.evaluate(X_test, y_test, verbose=0)
        print("Testing set Mean Abs Error: ${:7.2f}".format(mae))
        
        plt.scatter(y_test, y_pred)
        plt.xlabel('True Values ')
        plt.ylabel('Predictions ')
        plt.axis('equal')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        _ = plt.plot([-100, 100], [-20, 20])
        
        plt.plot(y_test, color = 'red', label = 'Real data')
        plt.plot(y_pred, color = 'blue', label = 'Predicted data')
        plt.title('Prediction')
        plt.legend()
        plt.show()
        
        error = y_pred - y_test
        plt.hist(error, bins = 50)
        plt.xlabel("Prediction Error ")
        _ = plt.ylabel("Count")

#load best result out of all epochs
model.load_weights('good_weights\Weights--008-126.35243.hdf5')   

#evaluate model
# Compile the model
model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])
        
#Predict test results from test data  
y_pred_test = model.predict(data_X_3) 
y_pred_test_f = model.predict(data_X_3).flatten()

plt.scatter(y_pred_test,data_Y_3, c='blue', alpha=0.5)
plt.title('Predicted vs Actual')
plt.ylabel('actual')
plt.xlabel('predicted')
plt.show()

df =pd.concat([y_pred_test_f,data_Y_3.flatten()], axis=1)
sb.pairplot(df, diag_kind = 'kde',
            plot_kws = {'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
            size = 3)
plt.suptitle('Predicted vs Actual',size = 18)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.show()


plt.scatter(data_Y_3, y_pred_test)
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-1, 1], [-1, 1])
        
plt.plot(data_Y_3, color = 'red', label = 'Real data')
plt.plot(y_pred_test, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()
        
error = y_pred_test - data_Y_3
#print(error.mean)
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error ")
_ = plt.ylabel("Count")       

def printMatrix(s):

    for i in range(len(s)):
        for j in range(len(s[0])):
            print("%5f-- " % (s[i][j]), end="")
 
        
printMatrix(y_pred_test)
  
len(y_pred_test)
#print(y_pred_test_f)
#len(data_Y_3)
     
#Predict test results
#y_pred = model.predict(data_X_3).flatten()
#
#[loss, mae] = model.evaluate(data_X_3, data_Y_3, verbose=0)
#print("Testing set Mean Abs Error: ${:7.2f}".format(mae))
