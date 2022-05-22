#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import random
import sklearn.preprocessing as preprocessing
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

from livelossplot import PlotLossesKeras
import matplotlib.pyplot as plt

from keras.models import load_model
from pickle import load

from datetime import datetime, timedelta


import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import statistics


# In[2]:


#column names all
column_names = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/0.data/4.column_names/1.VC65_ENG_columns_eng.csv", header=None).iloc[:,2].values
#stationary features, all y
step1_stat_features = np.array(pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc65_eng/step1_stationary_features.csv").iloc[:, 1])
#x for each stationary y
x_y_index = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc65_eng/step2_ys_x_before_pearson.csv")
#final selected X for each y
x_y_qualf = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc65_eng/step2_y_xs_qualified_pearson.csv")


# In[3]:


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


# In[20]:


####-------------2. normal accuracy critria-----------------should be saved AFTER ALL
nrm_critria = []
h_path = "/data/0.long_term_projects/Arika-san/DTC_time_series/20220413_model_with_mileage_scaler_cover_all_mileage_range_vc65eng"
nrm_accuracy_path = h_path+"/test_accuracy1"

files = os.listdir(nrm_accuracy_path)
y_list = []
accuracy_list = []

for file in files:
    index = []
    y = 0
    accuracy = 0
    
    if file[-3:] == 'csv':
        for i in range(len(file)):
            if file[i] == '_':
                index.append(i)
                
    if len(index) == 3:
        y = int(file[:index[0]])
        accuracy = float(file[index[2]+1:index[2]+7])
        y_list.append(y)
        accuracy_list.append(accuracy)

print (column_names[y_list],y_list, accuracy_list)
pd.DataFrame({"index":y_list, "y name":column_names[y_list], "accuracy":accuracy_list}).to_csv("/home/isuzu/Desktop/vc65_model_accuracy.csv")


# In[22]:


####-------------3.1 and 3.2 get normal vin in bin-----------------
vc65_eng_nrm_path = "/data/0.long_term_projects/Arika-san/DTC_time_series/0.data/2.VC65_ENG/0.VC65_ENG_NRM"
nrm_vin_list = os.listdir(vc65_eng_nrm_path)

nrm_vin_in_bin_list_0 = []
nrm_vin_in_bin_list_1 = []
nrm_vin_in_bin_list_2 = []
nrm_vin_in_bin_list_3 = []
nrm_vin_in_bin_list_4 = []

cnt = 0

for vin in nrm_vin_list:
    
    vin_folder = os.path.join(vc65_eng_nrm_path, vin) ##one vin path
    date_list = []
    one_vin_sorted_file_path_list = []
    nrm_mileage_list = []
    print (vin, cnt, len(nrm_vin_list))

    
    records = os.listdir(vin_folder)
    for record in records:
        if len(record) > 20:
            date_list.append(int(record[20:34]))
    
    date_list.sort()
    print ("0. sorted date")
    
    
    #sorted records path list
    for date in date_list:
        for record in records:
            if len(record) > 20 and int(record[20:34]) == date:
                one_vin_sorted_file_path_list.append(os.path.join(vin_folder, record))
    print ("1. get sorted path")

    #get mileage list
    for record in  one_vin_sorted_file_path_list:
        mileage = pd.read_csv(record, encoding='shift-jis').iloc[2, 6]
        nrm_mileage_list.append(mileage)
    print ("2. get mnileage list")

    
    mile_value = statistics.median(nrm_mileage_list)
    
    if mile_value < 25000:
        nrm_vin_in_bin_list_0.append(vin)
        print ("0 bin")
        
    if mile_value > 25000 and mile_value < 50000:
        nrm_vin_in_bin_list_1.append(vin)
        print ("1 bin")
        
    if mile_value > 50000 and mile_value < 75000:
        nrm_vin_in_bin_list_2.append(vin)
        print ("2 bin")
        
    if mile_value > 75000 and mile_value < 125000:
        nrm_vin_in_bin_list_3.append(vin)
        print ("3 bin")
        
    if mile_value > 125000:
        nrm_vin_in_bin_list_4.append(vin)
        print ("4 bin")
        
    cnt = cnt+1
    


# In[28]:


pd.DataFrame({"vin":nrm_vin_in_bin_list_4}).to_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220418_vc65eng_all_range_step4/bin4_vin.csv")


# In[17]:


####-------------3.2.1 sort dtc's nrm vin following date order-----------------
vc65_eng_nrm_target_path = "/data/0.long_term_projects/Arika-san/DTC_time_series/20220418_vc65eng_all_range_step4/nrm_results"

#create nrm target folders
for item in nrm_vin_list:
    try:
        os.mkdir(os.path.join(vc65_eng_nrm_target_path, item))
    except FileExistsError:
        pass

#sort each vin's records
for item in nrm_vin_list:
    date_list = []###
    records_path_list = []###
    one_vin_all_ys_accuracy = []###
    final_y_list = []###
    
    y_index_list = step1_stat_features
    y_x_list = y_x_list
    one_vin_folder = os.path.join(vc65_eng_nrm_path, item)
    records = os.listdir(one_vin_folder)
    
    for record in records:
        if len(record) > 20 and int(record[20:34]) == date:
            records_path_list.append(os.path.join(one_vin_folder, record))
    
    
    #-----------------3.2.1.1 process data, get sorted data list and mile list
    data_list, mile_list = process_data_for_testing(records_path_list)
    print ("3.2.1.1 processed data, get data and mile list", len(data_list), item)
    #-----------------3.2.1.1 process data, get sorted data list and mile list
    
    
    #-----------------3.2.1.2 get model, scalers
    for i in range(len(y_index_list)):
        y_index = y_index_list[i]
        x_index = y_x_list[i]
        scaler_x, scaler_y, model = get_model_scaler_index(y_index, h_path)
        print ("3.2.1.2 get model, scalers")
        if scaler_x == 0 and scaler_y ==0 and model ==0:#pass model not exist
            pass
    #-----------------3.2.1.2 get model, scalers
    
        
    #-----------------3.2.1.3 ge test results   
        else:
            final_y_list.append(y_index)
            one_vin_one_y_accuracy_list = get_accuracy_list(data_list, x_index, y_index, scaler_x, model, scaler_y)
            one_vin_all_ys_accuracy.append(one_vin_one_y_accuracy_list)
            print ("3.2.1.3 get one y accuracy", y_index, i,' of ', len(y_index_list))
            print ("elements in one y: ", len(one_vin_one_y_accuracy_list))
    #-----------------3.2.1.3 ge test results
    
    
    #----------------save final all ys accuracy to targte folder
    pd_ys_accuracy = pd.DataFrame(np.array(one_vin_all_ys_accuracy)).T
    print (pd_ys_accuracy.shape, pd_ys_accuracy.head(), len(final_y_list))
    
    pd_ys_accuracy.columns = final_y_list
    pd_ys_accuracy.insert(0, "miles", mile_list)
    pd_ys_accuracy.to_csv(os.path.join(vc65_eng_nrm_target_path, item, 'miles_all_ys_accuracy.csv'))
    print ("all ys accuracy for "+item, pd_ys_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#####-------------------------define functions-----------------------###


# In[5]:


def process_data_for_testing(sorted_path):
    data_list = []
    mile_list = []
    
    for file in sorted_path:
        data = pd.read_csv(file, header=None, encoding='shift-jis')
        data_list.append(np.array(data))
        mile_list.append(data.iloc[2, 6])
    
    return (data_list, mile_list)


# In[6]:


def get_model_scaler_index(y_index, h_path):
    
        model_path = h_path+"/models1/"+str(y_index)+".h5"
        x_scaler_path = h_path+"/scalers1/"+str(y_index)+"_x.pkl"
        y_scaler_path = h_path+"/scalers1/"+str(y_index)+"_y.pkl"

        # load model
        try:
            model = load_model(model_path)

            from pickle import dump
            scaler_x = load(open(x_scaler_path, 'rb'))
            scaler_y = load(open(y_scaler_path, 'rb'))

            return (scaler_x, scaler_y, model)
        
        except OSError:
            return (0, 0, 0)
        


# In[7]:


def get_accuracy_list(data_list, x_index, y_index, scaler_x, model, scaler_y):
    x_list = []
    y_list = []
        
    for i in range(len(data_list)):
        data_x = data_list[i][:, x_index]
        data_y = data_list[i][:, y_index]
        x_list.append(np.array(data_x))
        y_list.append(np.array(data_y))

    x_np = np.array(x_list)
    y_np = np.array(y_list)

    xn_2d = x_np.reshape(-1, len(x_index))
    yn_2d = y_np.reshape(-1, 1)
    #print (xn_2d)
    
    '''
    for i in range(len(x_index)):
        if xn_2d[:, i].dtype == np.dtype(object):
            new_array = np.zeros(xn_2d.shape[0])
            xn_2d[:, i] = new_array
    '''
    #prepare data
    xn_model_2d = scaler_x.transform(xn_2d)
    xn_model_3d = xn_model_2d.reshape(-1, 160, len(x_index))
    
    #predict x
    yn_pre = model.predict(xn_model_3d) #predict y
    yn_pre_2d = yn_pre.reshape(-1, 1)
    yn_pre_inv = scaler_y.inverse_transform(yn_pre_2d)
    yn_2d = yn_2d

    #get prediction accuracy
    accuracy_list = []
    from sklearn.metrics import mean_absolute_percentage_error
    for i in range(len(data_list)):
        b = mean_absolute_percentage_error(yn_pre_inv[160*i:160*(i+1)], yn_2d[160*i:160*(i+1)])
        accuracy_list.append(1-b)
            
    #print ("get predict accuracy matrix")
    return (accuracy_list)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




