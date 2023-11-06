#!/usr/bin/env python
# coding: utf-8

# In[35]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import seaborn as sns
from matplotlib import pyplot as plt



data = pd.read_csv("Traffic.csv")
data


# # **Preparing dataset **

data.columns = data.columns.str.lower().str.replace(' ', '_')
data


# # Feature Engineering for Logistic Regression optimization

# Convert the 'Time' column to a datetime object
data['time_new'] = pd.to_datetime(data['time'], format='%I:%M:%S %p')

# Extract hour, minute, and second
data['hour'] = data['time_new'].dt.hour
data['minute'] = data['time_new'].dt.minute
data['second'] = data['time_new'].dt.second

#Add new feature Weekend (True, false)
data['weekend'] = data['day_of_the_week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

#Add new features for parts of a day 
data['is_morning'] = data['hour'].apply(lambda x: 1 if x in range(6,12) else 0)
data['is_night'] = data['hour'].apply(lambda x: 1 if x in range(0,6) else 0)
data['is_day'] = data['hour'].apply(lambda x: 1 if x in range(12,18) else 0)
data['is_evening'] = data['hour'].apply(lambda x: 1 if x in range(18,24) else 0)

data



traffic_situation_mapping = {'low': 0, 'normal': 1, 'high': 2, 'heavy':3}
data['traffic_situation'].replace(traffic_situation_mapping, inplace=True)




#selecting columns for the dataset
dataset = data[['date', 'day_of_the_week', 'traffic_situation', 'hour',	'minute', 'weekend', 'is_morning', 'is_night', 'is_day', 'is_evening']]
dataset





# ## Split the data

len(dataset)

df_full_train, df_test = train_test_split(dataset, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

len(df_train), len(df_val), len(df_test)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#For the final model
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)



y_train = df_train.traffic_situation.values
y_val = df_val.traffic_situation.values
y_test = df_test.traffic_situation.values

#For the final model
y_full_train = df_full_train.traffic_situation.values
y_test = df_test.traffic_situation.values


# In[484]:


del df_train['traffic_situation']
del df_val['traffic_situation']
del df_test['traffic_situation']

#For the final model
del df_full_train['traffic_situation']


# In[485]:


dv = DictVectorizer(sparse=True)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)


# # Training the model

# In[486]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
import xgboost as xgb



# # Using Random Forest for training the model



# # Training the final selected model
print("Training the final model")

# In[504]:


dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)


# In[505]:


rf = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=1, class_weight='balanced')
rf.fit(X_full_train, y_full_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy


# # Saving the Model to a Python file
print("Saving the final model to a file")

# In[506]:


import pickle


# In[507]:


output_file = 'model.bin'
output_file


# In[ ]:


f_out = open(output_file, 'wb')
pickle.dump((dv, rf), f_out)
f_out.close()


# In[508]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)


# # Load a model

# In[3]:


import pickle


# In[4]:


model_file = 'model.bin'


# In[5]:


with open(model_file, 'rb') as f_in:
    dv, rf = pickle.load(f_in)


# In[6]:


dv, rf


# In[66]:


input = {
    'hour': 14,
    'minute': 00,
    'day_of_the_week': "Wednesday"
}


# In[67]:


x = dv.transform([input])
x


# In[68]:


y = rf.predict(x)
y




