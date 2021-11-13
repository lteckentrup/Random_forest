import tensorflow as tf
import pandas as pd
import numpy as np
import random as Rd
import tensorflow_decision_forests as tfdf
from wurlitzer import sys_pipes
import time
import netCDF4 as nc
import matplotlib.pyplot as plt
from numpy.random import rand
import os, psutil
from datetime import datetime
import math
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ntrees', type=int, required=True)
parser.add_argument('--nattr', type=int, required=True)
parser.add_argument('--fname', type=str, required=True)
parser.add_argument('--var', type=str, required=True)

args = parser.parse_args()

random.seed(42)

startTime = datetime.now()

target = 'CRUJRA'

models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
          'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-CM4', 'GFDL-ESM4',
          'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6',
          'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
          'NorESM2-MM']

def prepare(file,var):
    df = pd.read_csv('../../lpj_guess_australia/runs/CanESM5/'+file+'.out',
                     header=0, delim_whitespace=True)
    if file == 'cpool':
        df_CMIP6 = df.drop(columns=['VegC', 'LitterC', 'SoilC', 'Total'])
    elif file == 'cflux':
        df_CMIP6 = df.drop(columns=['Veg', 'Repr', 'Soil', 'Fire', 'Est', 'NEE'])
    elif file == 'fpc':
        df_CMIP6 = df.drop(columns=['BNE', 'BINE', 'BNS', 'TeNE', 'TeBS', 'IBS',
                                    'TeBE', 'TrBE', 'TrIBE', 'TrBR', 'C3G', 'C4G',
                                    'Total'])

    for m in models:
        CMIP6 = pd.read_csv('../../lpj_guess_australia/runs/'+m+'/'+file+'.out',
                            header=0, delim_whitespace=True)
        df_CMIP6[m] = CMIP6[var]*1000

    df_FULL = df_CMIP6.loc[df_CMIP6['Year'].isin(np.arange(1901,2019))].reset_index()
    df_CRUJRA = pd.read_csv('../../lpj_guess_australia/runs/CRUJRA/'+file+'.out',
                            header=0,delim_whitespace=True)

    df_FULL['CRUJRA'] = df_CRUJRA[var]*1000

    return(df_FULL)

full_df = prepare(args.fname, args.var)

def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return(dataset[~test_indices], dataset[test_indices])

train_df, test_df = split_dataset(full_df.drop(columns=['index', 'Year']))

# Convert my training data to tensorflow objects
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label='CRUJRA',
                                                 task=tfdf.keras.Task.REGRESSION)
# Configure the model.
model1 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION,
                                      num_candidate_attributes=args.nattr,
                                      num_trees=args.ntrees)
model1.compile(metrics=["mse"])

# Train the model.
start_time = time.time()
model1.fit(x=train_ds)
file_name = "model"

model1.make_inspector().evaluation()


#  TESTING
# Convert the testing dataset into tensorflow object
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label='CRUJRA',
                                                task=tfdf.keras.Task.REGRESSION)
# Evaluate the model on the test dataset.
evaluation = model1.evaluate(test_ds, return_dict=True)

print(evaluation)
print()
print(f"MSE: {evaluation['mse']}")
print(f"RMSE: {math.sqrt(evaluation['mse'])}")

full_ds = tfdf.keras.pd_dataframe_to_tf_dataset(full_df, label='CRUJRA',
                                                task=tfdf.keras.Task.REGRESSION)
# Make prediction
full = model1.predict(full_ds)
full_kg = full/1000

full_pred = full_df[['Lon','Lat','Year']].reset_index()
full_pred[args.var]=full_kg
full_pred_df = full_pred.drop(columns=['index'])

## save the results as .csv
full_pred_df.to_csv(args.var+'/PRED_'+args.var+'_'+
                    str(args.ntrees)+'_'+str(args.nattr)+'.csv',
                    index=False)
