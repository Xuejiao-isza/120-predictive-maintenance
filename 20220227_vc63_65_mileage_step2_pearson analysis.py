#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random


# In[14]:


#0. get Y index, and X range
step1_stationary_features = np.array(pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc60/step1_stationary_features.csv").iloc[:, 1])
print ("step1 stationary features: ", len(step1_stationary_features))


# In[15]:


nrm_path = "/data/0.long_term_projects/Arika-san/DTC_time_series/0.data/2.VC60/0.VC60_40s_NRM/"
##0. read all nrm data
cnt = 0
row_num = 160
col_num = 160
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
            data = pd.read_csv(data_path_list[i], header=None, encoding = 'Shift-JIS').iloc[:, :col_num]
            if data.shape[0] == row_num: #change row number based on data shape
                data_list.append(data)
                print (data.shape)
        except UnicodeDecodeError:
            data = pd.read_csv(data_path_list[i]).iloc[:, :col_num]
            if data.shape[0] == row_num:
                data_list.append(data)
                print (data.shape)
        
        
print ("0. get random data", len(data_list))


# In[16]:


all_x_list = []
for i in range(len(step1_stationary_features)):
    y_index = step1_stationary_features[i]
    x_index = [x for x in step1_stationary_features if x != step1_stationary_features[i]]
    all_x_list.append(x_index)

all_x_np = np.array(all_x_list)
print ("2.1 get X index for each y")
print (all_x_np.shape)

all_ys_x = pd.DataFrame(all_x_np)
all_ys_x.insert(0, 'y', step1_stationary_features)

all_ys_x.to_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc60/step2_ys_x_before_pearson.csv")
print ("save x_index and y_index")


# In[17]:


all_ys_x.shape


# In[18]:


#pearson correlation
from scipy.stats import pearsonr

all_results = [] 

for k in range(200):#200 samples
    one_p_coe = []
    print ("sample number is: ", k)
    
    for i in range(all_ys_x.shape[0]):#y index
        one_f = []
        for j in range(all_x_np.shape[1]):#x index
            y_index = all_ys_x.iloc[i, 0]
            x_index = all_ys_x.iloc[i, j+1]
            #print (data_list[k].iloc[:, y_index], data_list[k].iloc[:, x_index])
            
            try:
                data_list[k].iloc[:, y_index].astype(np.float32)
                data_list[k].iloc[:, x_index].astype(np.float32)
                corr, _ = pearsonr(data_list[k].iloc[:, y_index].astype(np.float32), data_list[k].iloc[:, x_index].astype(np.float32))
        
            except ValueError:
                corr = 0
              
            one_f.append(np.absolute(corr))
            print ("ith sample: ", k,  "y index: ", y_index, "  x_index: ", x_index, "cof: ", np.absolute(corr))
        
        print (one_f)
        one_p_coe.append(one_f)
    
    all_results.append(one_p_coe)
print ("2.2 get all results")


# In[19]:


all_np = np.array(all_results)
all_np_shape = all_np.shape
all_np_shape


# In[20]:


#fill nan to pearson results 
all_np = np.array(all_results)
p_2d = all_np.reshape(-1, all_np.shape[2])
p_pd = pd.DataFrame(p_2d)
p_pd.fillna(0, inplace=True)
print ("2.3 fill nan with 0", p_pd.shape)


# In[21]:


all_p = np.array(p_pd).reshape(all_np_shape[0], all_np_shape[1], all_np_shape[2])

all_p_f = []
for i in range(all_p.shape[1]):
    one_y = []
    for j in range(all_p.shape[2]):
        coe = np.mean(all_p[:, i, j])
        one_y.append(coe)
    
    print (i, ": ",  one_y)
    
    all_p_f.append(one_y)

print (all_p_f)


# In[22]:


all_p_np = np.array(all_p_f)
all_p_pd = pd.DataFrame(all_p_np)
all_p_pd.to_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc60/step2_y_xs_pearson_all.csv")


# In[23]:


#get top 20 x index for eahc y
select_y = []

for i in range(all_p.shape[1]):
    y_index = []
    for j in range(all_p.shape[2]):
        if all_p_np[i, j]>0.25 or all_p_np[i, j]<-0.25:
            y_index.append(1)
        else:
            y_index.append(0)
              
    select_y.append(y_index)

x_y_np = np.array(select_y)
pd.DataFrame(x_y_np).to_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220228/vc60/step2_y_xs_qualified_pearson.csv")
print ("2.5 save qualified x index")
print(x_y_np.shape)


# In[24]:


pd.DataFrame(x_y_np).head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




