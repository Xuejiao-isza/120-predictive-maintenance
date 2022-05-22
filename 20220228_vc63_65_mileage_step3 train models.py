#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import sklearn.preprocessing as preprocessing
from random import shuffle

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from livelossplot import PlotLossesKeras
import matplotlib.pyplot as plt

from numpy import random


# In[2]:


#column names all
column_names = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/0.data/4.column_names/1.VC63_ENG_columns_eng.csv", header=None).iloc[:,2].values
#stationary features, all y
step1_stat_features = np.array(pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/step1_stationary_features.csv").iloc[:, 1])
#x for each stationary y
x_y_index = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/step2_ys_x_before_pearson.csv")
#final selected X for each y
x_y_qualf = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/step2_y_xs_qualified_pearson.csv")


# In[3]:


step1_stat_features


# In[4]:


x_y_qualf.head()


# In[5]:


x_y_index.head()


# In[6]:


##1. get x index for each y
y_x_list = []

for i in range(len(step1_stat_features)):
    
    print ("y index is: ", column_names[step1_stat_features[i]])
    x_index = np.array(x_y_index.iloc[i, 2:])
    x_qualf = np.array(x_y_qualf.iloc[i, 1:])
    
    x_train_index = []
    for j in range(len(x_qualf)):
        if x_qualf[j] == 1:
            x_train_index.append(x_index[j])
    x_train_index.append(6) ## add mileage to x      
    print (column_names[x_train_index])
    print ("x number is: ", len(x_train_index))
    
    y_x_list.append(x_train_index)   


# In[18]:


int(len(data_path)*0.7)


# In[20]:


##get low mileage data
data_path = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/m10k.csv").iloc[:, 1].values
random.shuffle(data_path)

data_list = []
cnt = 0
row_num = 160 #vc63_eng
col_num = 336 #vc63_eng

for i in range(int(len(data_path)*0.7)):
    try:
        data1 = pd.read_csv(data_path[i], header=None, encoding = 'Shift-JIS')
        if data1.shape[0]==row_num and data1.shape[1]>10:
            data = data1.iloc[:, :col_num]
            data_list.append(np.array(data))
            print (data.shape)
            
    except UnicodeDecodeError:
        data1 = pd.read_csv(data_path[i])
        if data1.shape[0]==row_num and data1.shape[1]>10:
            data = data1.iloc[:, :col_num]
            data_list.append(np.array(data))
            print (data.shape)

data_np = np.array(data_list)


# In[21]:


data_np = np.array(data_list)
data_np.shape


# In[21]:


step1_stat_features[1]


# In[22]:


for m in range(x_y_index.shape[0]):
    y_index = step1_stat_features[m]
    x_index = y_x_list[m]
    
    print (len(x_index))
    # process data and train data for one y
    if len(x_index) > 3 :
        
        data_x = data_np[:, :, x_index]
        data_y = data_np[:, :, y_index]

        print (data_x.shape)
        print (data_y.shape)
        
        x_train = data_np[:10000, :, x_index].reshape(-1, len(x_index))
        y_train = data_np[:10000, :, y_index].reshape(-1, 1)
        print ("3.1 get data")
        
        ## get the scaler
        scaler_X = preprocessing.MaxAbsScaler().fit(x_train)
        scaler_y = preprocessing.MaxAbsScaler().fit(y_train)

        ## scale train data
        X_training = scaler_X.transform(x_train)
        y_training = scaler_y.transform(y_train)

        X_train_3d = X_training.reshape(-1, row_num, len(x_index))
        y_train_3d = y_training.reshape(-1, row_num, 1) 

        x_test = data_np[-600:, :, x_index].reshape(-1, len(x_index))
        y_test = data_np[-600:, :, y_index].reshape(-1, 1)

        x_testing = scaler_X.transform(x_test)
        x_test_3d = x_testing.reshape(-1, row_num, len(x_index))
        
        from pickle import dump
        dump(scaler_X, open('/data/0.long_term_projects/Arika-san/DTC_time_series/20220310/vc63_eng_10/scalers1/'+str(y_index)+'_x.pkl', 'wb'))
        dump(scaler_y, open('/data/0.long_term_projects/Arika-san/DTC_time_series/20220310/vc63_eng_10/scalers1/'+str(y_index)+'_y.pkl', 'wb'))
        print ("3.2 scale data")
       
        #cce = keras.losses.CategoricalCrossentropy()


        ##training model
        model = Sequential()
        model.add(LSTM(1, return_sequences=True, input_shape=(row_num, len(x_index)))) # do not use "batch_input_shape"
        #model.add(Dense(1))
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

        optimizer = keras.optimizers.Adam(lr=0.1)
        model.compile(loss='mse', optimizer='adam')
        history = model.fit(X_train_3d, y_train_3d, epochs = 30, validation_split = 0.5, verbose=1, callbacks=[PlotLossesKeras(), es])


        plt.plot(history.history['loss'], label = 'train loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.title(str(y_index))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.savefig("/data/0.long_term_projects/Arika-san/DTC_time_series/20220310/vc63_eng_10/training_fig1/"+str(y_index)+".png")
        plt.show()
        plt.close()

        model.save("/data/0.long_term_projects/Arika-san/DTC_time_series/20220310/vc63_eng_10/models1/"+str(y_index)+".h5")
        
        #test accuracy and save
        y_test_pre = model.predict(x_test_3d)
        y_test_pre_2d = y_test_pre.reshape(-1, 1)
        y_test_pre_rev = scaler_y.inverse_transform(y_test_pre_2d)
        
        
        from sklearn.metrics import mean_absolute_percentage_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score
        test_accuracy = []

        for k in range(50):
            b = mean_absolute_percentage_error(y_test_pre_rev[row_num*k:row_num*(k+1)], y_test[row_num*k:row_num*(k+1)])
            a = mean_squared_error(y_test_pre_rev[row_num*k:row_num*(k+1)], y_test[row_num*k:row_num*(k+1)])
            r2 = r2_score(y_test_pre_rev[row_num*k:row_num*(k+1)], y_test[row_num*k:row_num*(k+1)])
            
            test_accuracy.append(1-b)
            
            plt.plot(np.arange(row_num), y_test_pre_rev[row_num*k:row_num*(k+1)], label = 'pre')
            plt.plot(np.arange(row_num), y_test[row_num*k:row_num*(k+1)], label = 'true')
            plt.title(str(y_index)+"_MAPE_"+str(1-b)+"_MSE_"+str(a))
            plt.legend()
            plt.savefig("/data/0.long_term_projects/Arika-san/DTC_time_series/20220310/vc63_eng_10/test_accuracy1/"+str(y_index)+"_"+str(k)+".png")
            plt.close()
        
        
        acc = np.mean(test_accuracy)
        print (acc, r2)
        pd.DataFrame({"test_accuracy":test_accuracy}).to_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220310/vc63_eng_10/test_accuracy1/"+str(y_index)+"_x_"+str(len(x_index))+"_"+str(acc)+".csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




