#### Sanaa Hobeichi's code


import tensorflow as tf
import pandas as pd
import numpy as np
import random as Rd
import tensorflow_decision_forests as tfdf
from wurlitzer import sys_pipes
import time
import netCDF4 as nc
import os

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

models = ['CanESM5', 'CESM2-WACCM','CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg]
a = rand(20)
b = rand(20)
c = rand(20)
d = rand(20)
d = rand(20)
e = rand(20)

fake_data = [a,b,c,d,e]
train_df = pd.DataFrame()

for m,fd in zip(models,fake_data):
    train_df[m] = fd

train_df['CRUJRA'] = e

print(train_df.head())

# Convert my training data to tensorflow objects
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label='CRUJRA', task=tfdf.keras.Task.REGRESSION)
print(train_ds)

# Configure the model.
model1 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION, num_candidate_attributes=3, num_trees = 10)
model1.compile(metrics=["mse"])
print(model1)

# Train the model.
start_time = time.time()
with sys_pipes():
    model1.fit(x=train_ds)

file_name = "model.csv"
model1.save(file_name)
end_time = time.time()

time_need = end_time - start_time
print("time needed to run 10 forest is {}".format(time_need))

# ##############################################################################################
# ##############################################################################################

# ###################################################################################################################
# ###################################################################################################################
# #%% TESTING
# for bin in range(9,21):
#   print(bin)
#   file_name = "/mnt/e/Data/flux_Gab/half_hourly/csv/bins/test_bin{}.csv".format(bin)
#   test_site = pd.read_csv(file_name, index_col=0)

#   ## Prepare test dataset
#   test_df = pd.DataFrame()
#   sites_df = pd.DataFrame()
#   i = 0
#   for s in test_site.site_code:
#       print(i)
#       file_name = "/mnt/e/Data/flux_Gab/half_hourly/csv/sites_testing_Qle/{}_testing_half_hourly.csv".format(s)
#       s_df =  pd.read_csv(file_name, index_col=0)
#       #s_df = s_df.dropna()
#       s2_df = pd.DataFrame( data={'site_code':s_df['site_code']})
#       sites_df = sites_df.append(s2_df,ignore_index=True)

#       s_df = s_df.drop('site_code', 1)
#       test_df = test_df.append(s_df, ignore_index=True)

#       i = i+1

#   ## Convert the testing dataset into tensorflow object
#   test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label='Qle', task=tfdf.keras.Task.REGRESSION)

#   ## Load the saved model
#   file_name = "/mnt/e/Data/flux_Gab/half_hourly/RFTF/model_bin{}.csv".format(bin)
#   model2 = tf.keras.models.load_model(file_name)

#   pred = model2.predict(test_ds)

#   ## save the results as .csv
#   pred_df = pd.DataFrame(pred, columns=['pred'])
#   obs_df = pd.DataFrame( data={'obs':test_df['Qle']})
#   pred_df = pd.concat([sites_df, pred_df, obs_df], axis=1)
#   pred_df.insert(0, "bin", bin)

#   file_name = file_name = "/mnt/e/Data/flux_Gab/half_hourly/csv/bins/pred_bin{}.csv".format(bin)
#   pred_df.to_csv(file_name, index=False)


# ############################################################################################
# ############################################################################################

# #%% EVALUATION -- compare predicted Qle with the Observed Qle
# evaluation =model2.evaluate(test_ds, return_dict=True)
# print(evaluation)
# print(f"MSE:".format(evaluation['mean_squared_error']))
# #%%
# print(np.mean(pred))
# print(np.mean(test_df['Qle']))

# print(np.std(pred))
# print(np.std(test_df['Qle']))

# print(np.median(pred))
# print(np.median(test_df['Qle']))

# print(type(pred))

# ff = test_df['Qle'].to_numpy(dtype=type(pred))
# print(type(ff))
# print(ff[0])
# print(pred)
# print(np.corrcoef(pred, ff))

# #%%
# ## USE Gradient Boosted Tree Models from Tensorflow to test the importance of each feature
# # train_rank_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label='Qle', task=tfdf.keras.Task.RANKING)
# # model_rank = tfdf.keras.GradientBoostedTreesModel( task=tfdf.keras.Task.RANKING,ranking_group="climate", num_trees=50)

# # with sys_pipes():
# #   model_rank.fit(x=train_rank_ds)
