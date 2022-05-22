#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import random


# In[11]:


nrm_res_path = "/data/0.long_term_projects/Arika-san/DTC_time_series/20220418_vc65eng_all_range_step4/nrm_results"
bin0_vin = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220418_vc65eng_all_range_step4/bin0_vin.csv").iloc[:, 1].values
bin1_vin = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220418_vc65eng_all_range_step4/bin1_vin.csv").iloc[:, 1].values
bin2_vin = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220418_vc65eng_all_range_step4/bin2_vin.csv").iloc[:, 1].values
bin3_vin = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220418_vc65eng_all_range_step4/bin3_vin.csv").iloc[:, 1].values
bin4_vin = pd.read_csv("/data/0.long_term_projects/Arika-san/DTC_time_series/20220418_vc65eng_all_range_step4/bin4_vin.csv").iloc[:, 1].values

#select high accuracy features
ori_index= ["24","29","32","48","50","51","57","82",
            "115","141","156","159","181","266","269",
           "279","292","325","326","327","329","334"]

##mileage bin infor
bin0_path = []
bin1_path = []
bin2_path = []
bin3_path = []
bin4_path = []

#accuracy path for each mileage bin
for i in range(len(bin0_vin)):
    path = os.path.join(nrm_res_path, bin0_vin[i],"miles_all_ys_accuracy.csv")
    bin0_path.append(path)
    
for i in range(len(bin1_vin)):
    path = os.path.join(nrm_res_path, bin1_vin[i],"miles_all_ys_accuracy.csv")
    bin1_path.append(path)
    
for i in range(len(bin2_vin)):
    path = os.path.join(nrm_res_path, bin2_vin[i],"miles_all_ys_accuracy.csv")
    bin2_path.append(path)
    
for i in range(len(bin3_vin)):
    path = os.path.join(nrm_res_path, bin3_vin[i],"miles_all_ys_accuracy.csv")
    bin3_path.append(path)
    
for i in range(len(bin4_vin)):
    path = os.path.join(nrm_res_path, bin4_vin[i],"miles_all_ys_accuracy.csv")
    bin4_path.append(path)
    


# In[13]:


bin0_data_list = []

for i in range(len(bin0_path)):
    data  = pd.read_csv(bin0_path[i]).loc[:, ori_index]
    #print (data.shape)
    for j in range(data.shape[0]):
        bin0_data_list.append(data.iloc[j, :])
                    
nrm_array = np.array(bin0_data_list)


# In[14]:


from sklearn.ensemble import IsolationForest
clf = IsolationForest(random_state=0).fit(nrm_array)


# In[21]:


#test bin
test_data_list = []

for i in range(len(bin0_path)): #change mileage range
    data  = pd.read_csv(bin0_path[i]).loc[:, ori_index] #change mileage range
    print (data.shape)
    for j in range(data.shape[0]):
        test_data_list.append(data.iloc[j, :])
                    
test_array = np.array(test_data_list)


# In[22]:


score_list = []
r = test_array.shape[0]
for i in range(r):
    print (i, r)
    test = np.array(test_array[i, :]).reshape(1, -1)
    y_pre = clf.predict(test)
    y_pre2 = clf.score_samples(test)
    score_list.append(y_pre2[0]+1)



plt.hist(score_list)
plt.title("0_bin score distribution")
plt.xlabel("iForest score")
plt.ylabel("test counts")


# In[23]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
n_95 = np.percentile(score_list, 5)

plt.hist(score_list)
plt.title("0_bin score distribution "+str(n_95))
plt.xlabel("iForest score")
plt.ylabel("test counts")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




