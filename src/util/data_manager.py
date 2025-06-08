import sys
import os
from io import StringIO
import json
import csv
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as cf


# read a csv file
def read_csv_file(file_path=None, file_name=None, type='s3'):
    if(type == 's3'):
        data = cf.S3_CLIENT.get_object(Bucket=cf.S3_DATA_PATH + '/' + bucket_name, Key=file_name)
        data = pd.read_csv(data['Body'])
    elif(type == 'local'):
        data = pd.read_csv(file_path + '/' + file_name)
        #data = pd.read_csv(cf.DATA_PATH + '/' + file_name)
    return data


# store a json file
def write_json_file(bucket_name, file_name, data, type='s3'):
    if(type == 's3'):
        cf.S3_CLIENT.put_object(Bucket=bucket_name, Key=file_name, Body = json.dumps(data).encode('UTF-8'))        
    elif(type == 'local'):
        with open(file_name, 'w') as outfile:
            json.dump(data, outfile)


# write a csv file
def write_csv_file(file_path, file_name, data, type='s3'):
    if(type == 's3'):
        csv_buffer = StringIO()
        data.to_csv(csv_buffer, index=False)
        cf.S3_CLIENT.put_object(Bucket=file_path, Key=file_name, Body = csv_buffer.getvalue())
    elif(type == 'local'):
        data.to_csv(file_path + '/' + file_name, index=False)


def split_data(X, y, test_size=0.2, random_state=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
    