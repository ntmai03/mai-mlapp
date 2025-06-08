# config
from pathlib import Path
import os
import sys

# 
import pandas as pd
import numpy as np
import pickle as p
import json
import joblib

import function
from function import HousePrice

# Define path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config.yml')
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
TRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw')
DATA_PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
PIPELINE_PATH = os.path.join(SRC_PATH, 'pipeline')
ANALYSIS_PATH = os.path.join(SRC_PATH, 'analysis')
IMAGE_PATH = os.path.join(PROJECT_ROOT, 'image')


######################################################################
# data preparation
######################################################################
# load data
data = function.read_csv_file('kc_house_data.csv')
print(data.shape)

# split data to X and y
X_train, X_test, y_train, y_test = function.split_data(data, data['price'])
print(X_train.shape)
print(X_test.shape)


######################################################################
# Not use OOP
######################################################################
# data processing
processed_X_train = function.data_processing_pipeline(X_train)
processed_X_test = function.data_processing_pipeline(X_test)
print(processed_X_train)

# train model
summary_table = function.train_regression_sklearn(processed_X_train, y_train)
print(summary_table)

# predict new object
new_obj = dict({'price': [221900.0],
                'bedrooms': ['3'],
                'bathrooms': ['1'],
                'sqft_living': ['1180'],
                'sqft_lot': ['5650'],
                'floors': ['1'],
                'waterfront': ['0'],
                'view': ['0'],
                'condition': ['3'],
                'grade': ['7'],
                'sqft_above': ['1180'],
                'sqft_basement': ['0'],
                'yr_built': ['1955'],
                'yr_renovated': ['0'],
                'lat': ['47.5112'],
                'long': ['-122.2570'],
                'sqft_living15': ['1340'],
                'sqft_lot15': ['5650'],
                'zipcode': ['98178'],
                'date': ['20141013T000000']
    })
new_obj = pd.DataFrame.from_dict(new_obj)
new_obj = function.data_processing_pipeline(new_obj)
modelfile = TRAINED_MODEL_PATH + '/house_price_gbt.pkl'
model = joblib.load(modelfile)
prediction = model.predict(new_obj)
print(prediction)



######################################################################
# use OOP
######################################################################
houseprice = HousePrice()
houseprice.processed_X_train = houseprice.data_processing_pipeline(X_train)
houseprice.processed_X_test = houseprice.data_processing_pipeline(X_test)
houseprice.y_train = y_train
houseprice.y_test = y_test 
print(houseprice.processed_X_train)

# train model
summary_table = houseprice.train_regression_sklearn()
print(summary_table)

# predict new object
new_obj = dict({'price': [221900.0],
                'bedrooms': ['3'],
                'bathrooms': ['1'],
                'sqft_living': ['1180'],
                'sqft_lot': ['5650'],
                'floors': ['1'],
                'waterfront': ['0'],
                'view': ['0'],
                'condition': ['3'],
                'grade': ['7'],
                'sqft_above': ['1180'],
                'sqft_basement': ['0'],
                'yr_built': ['1955'],
                'yr_renovated': ['0'],
                'lat': ['47.5112'],
                'long': ['-122.2570'],
                'sqft_living15': ['1340'],
                'sqft_lot15': ['5650'],
                'zipcode': ['98178'],
                'date': ['20141013T000000']
    })
new_obj = pd.DataFrame.from_dict(new_obj)
new_obj = houseprice.data_processing_pipeline(new_obj)
modelfile = TRAINED_MODEL_PATH + '/house_price_gbt.pkl'
model = joblib.load(modelfile)
prediction = model.predict(new_obj)
print(prediction)
