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

# source code
from src import config as cf
from src.analysis.house_price import HousePrice

# Evaluation metrics for Regression 
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score



######################################################################
# Initialize
######################################################################
houseprice = HousePrice()


######################################################################
# data preparation
######################################################################
houseprice.load_dataset()
print(houseprice.data.shape)

houseprice.processed_X_train = houseprice.data_processing_pipeline(houseprice.X_train)
houseprice.processed_X_test = houseprice.data_processing_pipeline(houseprice.X_test)
print(houseprice.processed_X_train)


######################################################################
# train model
######################################################################
summary_table = houseprice.train_regression_sklearn()
print(summary_table)


######################################################################
# Model Evaluation
######################################################################
# prediction train set and test set
houseprice.y_train_pred = houseprice.model.predict(houseprice.processed_X_train)
houseprice.y_test_pred = houseprice.model.predict(houseprice.processed_X_test)
# performance evaluation
print(np.round(mean_squared_error(houseprice.y_train, houseprice.y_train_pred)),2)
print(np.round(mean_squared_error(houseprice.y_test, houseprice.y_test_pred)),2)


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
houseprice = HousePrice()
new_obj = houseprice.data_processing_pipeline(new_obj)
new_obj = new_obj[houseprice.TRAIN_VARS]

modelfile = 'model/house_price_gbt.pkl'

model = joblib.load(modelfile)
prediction = model.predict(new_obj)
print(prediction)

