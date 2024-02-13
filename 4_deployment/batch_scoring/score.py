#!/usr/bin/env python
# coding: utf-8

# The idea here is to create a notebook first then turning it into a script,
# 
# //Afterwards we will pass arguments to run as it will run with arguments
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import mlflow
import pickle
import os
import uuid
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


# In[21]:


taxi_type = 'green'
year=2021
month='03'

input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month}.parquet"
output_file = f"output/{taxi_type}_{year}_{month}.parquet"


RUN_ID = '7f6c9b37be7041d89ed1abc9d4f14836'


# In[18]:


def generate_uuids(n):
    ride_ids =[]
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds()/60)


    df['ride_id'] = generate_uuids(len(df))
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical  = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


# In[27]:


def load_model(run_id):
    """Here we used the bucket address where the model is stored, since we are using locally no need to pass bucket address as params"""
    logged_model = f"s3://mlflow-artifacts-remote-11/mlflow_artifacts/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, RUN_ID, output_file):
    print("Reading the model")
    df  = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    print("loading the dictvecotrizer")
    with open('dv.bin','rb') as f_out:
        dv = pickle.load(f_out)

    X = dv.transform(dicts)
    print("Loading the model")
    model = load_model(RUN_ID)
    y_pred= model.predict(X)

    print("Creating the results dataframe")
    #modify the output dataframe
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = RUN_ID
    
    df_result.to_parquet(output_file, index=False)
    print("Model reading complete")


# In[28]:




# # Now we will turn this notebook into a script

# In[ ]:

def run():
    taxi_type = sys.argv[1] # green
    year = sys.argv[2] #2021
    month = sys.argv[3] #3


    apply_model(
        input_file=input_file,
        RUN_ID=RUN_ID,
        output_file=output_file
    )

if __name__ == "__main__":
    run()
