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


# In[3]:


#column names all
column_names = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/0.data/4.column_names/1.VC63_ENG_columns_eng.csv", header=None).iloc[:,2].values
#stationary features, all y
step1_stat_features = np.array(pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/step1_stationary_features.csv").iloc[:, 1])
#x for each stationary y
x_y_index = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/step2_ys_x_before_pearson.csv")
#final selected X for each y
x_y_qualf = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/step2_y_xs_qualified_pearson.csv")


# In[4]:


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


# In[5]:


##original data1
data_path = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/m10k.csv").iloc[:, 1].values
random.shuffle(data_path)

data_list = []
cnt = 0
row_num = 160
col_num = 336
#row_num = 160 #vc65_eng
#col_num = 380
for i in range(10000):
    try:
        data = pd.read_csv(data_path[i], header=None, encoding = 'Shift-JIS').iloc[:, :col_num]
        if data.shape[0]==row_num:
            #artificial data
            #a = random.randint(500, 100000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)
    except UnicodeDecodeError:
        data = pd.read_csv(data_path[i]).iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(500, 100000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)

data_np_1 = np.array(data_list)


##artil data2
data_path = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/m25k.csv").iloc[:, 1].values
random.shuffle(data_path)

data_list = []
cnt = 0

for i in range(10000):
    try:
        data = pd.read_csv(data_path[i], header=None, encoding = 'Shift-JIS').iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(100000, 200000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)
    except UnicodeDecodeError:
        data = pd.read_csv(data_path[i]).iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(100000, 200000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)

data_np_2 = np.array(data_list)




##arti data3
data_path = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/m50k.csv").iloc[:, 1].values
random.shuffle(data_path)

data_list = []
cnt = 0

for i in range(10000):
    try:
        data = pd.read_csv(data_path[i], header=None, encoding = 'Shift-JIS').iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(200000, 300000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)
    except UnicodeDecodeError:
        data = pd.read_csv(data_path[i]).iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(200000, 300000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)

data_np_3 = np.array(data_list)



##arti data4
data_path = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/m70k.csv").iloc[:, 1].values
random.shuffle(data_path)

data_list = []
cnt = 0

for i in range(10000):
    try:
        data = pd.read_csv(data_path[i], header=None, encoding = 'Shift-JIS').iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(300000, 400000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)
    except UnicodeDecodeError:
        data = pd.read_csv(data_path[i]).iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(300000, 400000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)

data_np_4 = np.array(data_list)



##arti data5
data_path = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc63_eng/m99k.csv").iloc[:, 1].values
random.shuffle(data_path)

data_list = []
cnt = 0

for i in range(5000):
    try:
        data = pd.read_csv(data_path[i], header=None, encoding = 'Shift-JIS').iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(400000, 550000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)
    except UnicodeDecodeError:
        data = pd.read_csv(data_path[i]).iloc[:, :col_num]
        if data.shape[0]==row_num:
            #a = random.randint(400000, 550000)
            #data.iloc[:, 6] = np.repeat(a, 160)
            data_list.append(np.array(data))
            print (data.shape)

data_np_5 = np.array(data_list)
        


# In[9]:


target_path= "/data/0.long_term_projects/Arika-san/DTC_time_series/20220412_model_with_mileage_scaler_cover_all_mileage_range_vc63eng/0.trend_results"
accuracy_list_1 = []
accuracy_list_2 = []
accuracy_list_3 = []
accuracy_list_4 = []
accuracy_list_5 = []



for t in range(x_y_index.shape[0]):
    y_index = step1_stat_features[t]
    x_index = y_x_list[t]
    
    if len(x_index) >3:  ###x_index should higher than 3 features
        print (x_index)
        #step1: get models and scalers
        h_path = "/data/0.long_term_projects/Arika-san/DTC_time_series/20220412_model_with_mileage_scaler_cover_all_mileage_range_vc63eng"
        scaler_x, scaler_y, model = get_model_scaler_index(y_index, h_path)
        print ("get scalers and models")

        data1_accuracy = get_accuracy(data_np_1, x_index, y_index, scaler_x, model, len(data_np_1))
        data2_accuracy = get_accuracy(data_np_2, x_index, y_index, scaler_x, model, len(data_np_2))
        data3_accuracy = get_accuracy(data_np_3, x_index, y_index, scaler_x, model, len(data_np_3))
        data4_accuracy = get_accuracy(data_np_4, x_index, y_index, scaler_x, model, len(data_np_4))
        data5_accuracy = get_accuracy(data_np_5, x_index, y_index, scaler_x, model, len(data_np_5))
        
        mean_data1_accuracy = statistics.mean(data1_accuracy)
        mean_data2_accuracy = statistics.mean(data2_accuracy)
        mean_data3_accuracy = statistics.mean(data3_accuracy)
        mean_data4_accuracy = statistics.mean(data4_accuracy)
        mean_data5_accuracy = statistics.mean(data5_accuracy)

        
        fig_path = os.path.join(target_path, str(y_index)+'.png')
        print (fig_path)
        sns.distplot(data1_accuracy, hist=True, kde=True, label = 'original <100k  '+str(round(mean_data1_accuracy, 2)))
        sns.distplot(data2_accuracy, hist=True, kde=True, label = 'original 100k-200k  '+str(round(mean_data2_accuracy, 2)))
        sns.distplot(data3_accuracy, hist=True, kde=True, label = 'original 200k-300k  '+str(round(mean_data3_accuracy, 2)))
        sns.distplot(data4_accuracy, hist=True, kde=True, label = 'original 300k-400k  '+str(round(mean_data4_accuracy, 2)))
        sns.distplot(data5_accuracy, hist=True, kde=True, label = 'original >400k  '+str(round(mean_data5_accuracy, 2)))

        plt.title(column_names[y_index]) 
        plt.xlim((-2, 2))
        plt.legend()

        plt.savefig(fig_path)
        plt.show()
        plt.close()
        
        fig_path2 = os.path.join(target_path, str(y_index)+'_2.png')
        plt.plot(np.arange(5), [mean_data1_accuracy, mean_data2_accuracy, mean_data3_accuracy, mean_data4_accuracy, mean_data5_accuracy])
        plt.xticks(np.arange(5), ['<100k', '100k-200k', '200k-300k', '300k-400k', '>400k'])
        plt.savefig(fig_path2)
        plt.show()
        plt.close()


# In[6]:


#y_index = y_index_list[t]
def get_model_scaler_index(y_index, h_path):
    
    #step1: import model and scaler
        #import model
        model_path = h_path+"/models1/"+str(y_index)+".h5"
        x_scaler_path = h_path+"/scalers1/"+str(y_index)+"_x.pkl"
        y_scaler_path = h_path+"/scalers1/"+str(y_index)+"_y.pkl"

        # load model
        model = load_model(model_path)
        from pickle import dump
        scaler_x = load(open(x_scaler_path, 'rb'))
        scaler_y = load(open(y_scaler_path, 'rb'))

        return (scaler_x, scaler_y, model)


# In[8]:


def get_accuracy(nrm_list, x_index, y_index, scaler_x, model, sample_num):
    nrm_x_list = []
    nrm_y_list = []
    cnt = 0

    for i in range(len(nrm_list)):
        data_x = nrm_list[i][:, x_index]
        data_y = nrm_list[i][:, y_index]

        nrm_x_list.append(np.array(data_x))
        nrm_y_list.append(np.array(data_y))

        cnt = cnt+1
        #print (cnt)

    nrm_x_np = np.array(nrm_x_list)
    nrm_y_np = np.array(nrm_y_list)
    #print ("step 2.1 got x and y data")
    print (nrm_x_np.shape)
    print (nrm_y_np.shape)

    x_2d1 = nrm_x_np.reshape(-1, len(x_index))
    x_2d = x_2d1.astype(np.float)
    y_2d = nrm_y_np.reshape(-1, 1)
    
    
    x_model_2d = scaler_x.transform(x_2d)
    x_model = x_model_2d.reshape(-1, 160, len(x_index))
    #print ("step 2.2: normal data is ready")

    #2.3 predict nrm y
    y_pre = model.predict(x_model) #predict y
    y_pre_2d = y_pre.reshape(-1, 1)
    y_pre_inv = scaler_y.inverse_transform(y_pre_2d)
    y_2d = y_2d
    #print ("step 2.3: predict Y")

    #2.4 get normal prediction accuracy
    nrm_accuracy = []
    from sklearn.metrics import mean_absolute_percentage_error

    for i in range(sample_num):
        try:
            b = mean_absolute_percentage_error(y_pre_inv[160*i:160*(i+1)], y_2d[160*i:160*(i+1)])
            nrm_accuracy.append(1-b)
            #print (1-b)
        except ValueError:
            nrm_accuracy.append(0) 
                
    return(nrm_accuracy)


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




