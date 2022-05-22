#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import random


# In[2]:


## X_y doc, normal data path
X_Y_DOC = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/0.data/5.column_explain/X_Y_by_Yong_0131/4.VC63_ENG_X_y.csv")
nrm_path = "/data/0.long_term_projects/Arika-san/DTC_time_series/0.data/2.VC60/0.VC60_40s_NRM/"


# In[17]:


test_data = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/0.data/2.VC60/0.VC60_40s_NRM/CXZ77BT-7001494/VC60_191216163500.csv")
test_data.shape


# In[18]:


##0. read all nrm data
row_num = 160
col_num = 160
cnt = 0
data_path_list = []
data_list = []

#get all data path
for subdir, dirs, files in os.walk(nrm_path):
    
    for filename in files:
        filepath = subdir + os.sep + filename
        data_path_list.append(filepath)
        
#shuffle data path
random.shuffle(data_path_list)    


#get random 1000 file
for i in range(2000):
    if data_path_list[i][-3:] == 'csv':
        try:
            data = pd.read_csv(data_path_list[i], header=None, encoding = 'Shift-JIS').iloc[:, :col_num] ##none header
            if data.shape[0] == row_num:
                data_list.append(data)
                print (data.shape)
        except UnicodeDecodeError:
            data = pd.read_csv(data_path_list[i]).iloc[:, :col_num]
            if data.shape[0] == row_num:
                data_list.append(data)
                print (data.shape)
        
        
print ("0. get random data", len(data_list))


# In[19]:


#1. only keep float col
data_np = np.array(data_list)
print (data_np.shape)

#get all float index
float_index = [] #all float index
for i in range(data_np.shape[2]):
    try:
        data_np[:, :, i].astype(np.float)
        float_index.append(i)    
    except ValueError:
        pass
    
pd.DataFrame({"all float col":float_index}).to_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc60/step1_all_float_col.csv")
print ("1.all float col number is: ", len(float_index))
print ("all float col saved")   


# In[20]:


len(float_index)


# In[21]:


#2. get difference of all float data, prepare for stationary analysis
y_diff = []
for i in range(data_np.shape[0]):
    y = data_np[i][:, float_index].astype(np.float)## cannot use [:, :, :]format
    y_d = np.diff(y, axis = 0) ## get differ for each columns
    y_diff.append(np.array(y_d))

y_diff_np = np.array(y_diff)
y_diff_np[np.isnan(y_diff_np)] = 0 ## replace nan with 0
print ("2. y float sensors' diff shape is: ", y_diff_np.shape)


# In[60]:





# In[23]:


#2. get stationary analysis for all float columns
from statsmodels.tsa.stattools import adfuller
y_ADF_list = []
y_cite_value_list = []
y_p_list = []

for i in range(y_diff_np.shape[0]): ##~ 2000 sample
    y_adf = []
    y_cite = []
    y_p = []
    
    for m in range(y_diff_np.shape[2]): ## each float feature
        y_results = adfuller(y_diff_np[i, :, m])
        
        y_adf.append(y_results[0])
        y_cite.append(y_results[4])
        y_p.append(y_results[1])
        
        print (i, "th sample: ", m, "th  y: ", y_results[0], y_results[1], y_results[4])
    
    y_ADF_list.append(np.array(y_adf)) # ADF value
    y_cite_value_list.append(np.array(y_cite)) # criteria 5% of ADF
    y_p_list.append(np.array(y_p)) # lower than 5%

print ("2. get ADF and P ")


# In[24]:


# 2. fill nan
#y p value
y_p_pd = pd.DataFrame(np.array(y_p_list))
y_p_pd.fillna(1, inplace=True)
y_p_pd.head()

print (y_p_pd.shape)
print ("2. fill ADF and P nan ")


# In[25]:


##3. plot and get each feature's qualified ratio
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

p_ratio_list = []

for i in range(y_p_pd.shape[1]):
    p_ratio_bool= np.array(y_p_pd.iloc[:, i])<0.05
    p_ratio_np = y_p_pd.iloc[:, i][p_ratio_bool]
    p_ratio = len(p_ratio_np)/y_p_pd.shape[0]
    p_ratio_list.append(p_ratio)
    
print ("3. adf and p ratio list")
print (p_ratio_list, len(p_ratio_list))


# In[26]:


##4. features: more than 20% data meet stationary requirment
step1_stationary_index = []

for i in range(len(p_ratio_list)):
    if p_ratio_list[i] > 0.2:
        step1_stationary_index.append(float_index[i])

pd.DataFrame({"step1_stationary_index":step1_stationary_index}).to_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc60/step1_stationary_features.csv")
print ("4. step1_stationary_features is saved")


# In[ ]:




