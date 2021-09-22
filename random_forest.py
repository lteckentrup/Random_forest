'''
Based on Sanaa Hobeichi's script
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import random as Rd
import tensorflow_decision_forests as tfdf
from wurlitzer import sys_pipes
import time
import netCDF4 as nc

from numpy.random import rand
import os, psutil
from datetime import datetime

startTime = datetime.now()

### names of CMIP6 models
models = ['CanESM5', 'CESM2-WACCM','CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg',
          'GFDL-CM4', 'IITM-ESM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
          'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
          'NorESM2-LM', 'NorESM2-MM']

### name of target (reanalysis)
target = 'CRUJRA'

### generate fake data
a = np.random.randn(20)
b = np.random.randn(20)
c = np.random.randn(20)
d = np.random.randn(20)
e = np.random.randn(20)
f = np.random.randn(20)
g = np.random.randn(20)
h = np.random.randn(20)
i = np.random.randn(20)
j = np.random.randn(20)
k = np.random.randn(20)
l = np.random.randn(20)
m = np.random.randn(20)
n = np.random.randn(20)
o = np.random.randn(20)
p = np.random.randn(20)
q = np.random.randn(20)

fake_data_train = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]

full_df = pd.DataFrame()
for m,fd in zip(models,fake_data_train):
    full_df[m] = fd

full_df[target] = q

### Split into to train and test data
train_df = full_df.head(15)
test_df = full_df.tail(5)

### Convert training data to tensorflow objects
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label='CRUJRA', task=tfdf.keras.Task.REGRESSION)

### Configure the model
model1 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION, num_candidate_attributes=5, num_trees = 100)
model1.compile(metrics=["mse"])

### Train the model
model1.fit(x=train_ds)
file_name = "model"

model1.save(file_name)
end_time = time.time()

process = psutil.Process(os.getpid())

### Get memory
print(process.memory_info().rss/(1024 ** 2))
## Get run time
print(datetime.now() - startTime)

#  TESTING
# Convert the testing dataset into tensorflow object
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label='CRUJRA', task=tfdf.keras.Task.REGRESSION)
pred = model1.predict(test_ds)
