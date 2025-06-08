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

# user-defined functions
from src import config as cf
from src.analysis.house_price import HousePrice
from src.util import data_manager as dm
from src.util import regression_util as reg_util

# Evaluation metrics for Regression 
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score



######################################################################
# data preparation
######################################################################
# load data
data = dm.read_csv_file('', 'kc_house_data.csv', 'local')
print(data.shape)

# split data to X and y
X_train, X_test, y_train, y_test = reg_util.split_data(data, data['price'])
print(X_train.shape)
print(X_test.shape)

# Initialize
houseprice = HousePrice()

# data processing - using type 1
processsed_X_train = houseprice.data_processing_pipeline(X_train)
processsed_X_test = houseprice.data_processing_pipeline(X_test)
print(processsed_X_train.shape)
print(processsed_X_test.shape)
print(processsed_X_train.head())


######################################################################
# train model
######################################################################
# train model - type
houseprice.processed_X_train = processsed_X_train
houseprice.processsed_X_test = processsed_X_test
houseprice.y_train = y_train
houseprice.y_test = y_test 
summary_table = houseprice.train_regression_sklearn()
print(summary_table)


######################################################################
# Model Evaluation
######################################################################
# prediction train set and test set
y_train_pred = houseprice.model.predict(houseprice.processed_X_train)
y_test_pred = houseprice.model.predict(houseprice.processsed_X_test)

# performance evaluation
print(np.round(mean_squared_error(y_train, y_train_pred)),2)
print(np.round(mean_squared_error(y_test, y_test_pred)),2)


######################################################################
# predict new object
######################################################################
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
new_obj = new_obj[houseprice.TRAIN_VARS]
model_file = 'model/house_price_gbt.pkl'
model = joblib.load(model_file)
prediction = model.predict(new_obj)
print(prediction)