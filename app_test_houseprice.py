#import test
#from test import house_price_test2
import os
import requests
import ast
import joblib
import pickle as p
import pandas as pd
import numpy as np
from src.util import data_manager as dm
from src.analysis import house_price_module as hp
from src import config as cf


project_name = 'house_price'
version = 'v2'

# read original data and split into train test split
input_df = dm.read_csv_file(os.path.join(cf.DATA_PATH, 'input'), "kc_house_data.csv", "local")
hp.train_test_data("kc_house_data", "local",'v2')
train_df = dm.read_csv_file(os.path.join(cf.DATA_PATH, 'input'), "kc_house_data_train_" + cf.house_price_version + ".csv", "local")
test_df = dm.read_csv_file(os.path.join(cf.DATA_PATH, 'input'), "kc_house_data_test_" + cf.house_price_version + ".csv", "local")
print(input_df.shape, train_df.shape, test_df.shape)


cleaned_train_df = hp.clean_data(train_df)
cleaned_test_df = hp.clean_data(test_df)
X_train = hp.data_processing_pipeline(train_df,1)
X_test = hp.data_processing_pipeline(test_df,2)
y_train = cleaned_train_df['price']
y_test = cleaned_test_df['price']
print(X_train.shape,X_test.shape)
print(X_train.columns)
print(X_test.columns)

result = hp.train_xgboost(X_train, y_train, version)
print(result.score(X_train, y_train))
print(result.score(X_test, y_test))

# predict new data
#input_data = input_df.iloc[20]
input_data = input_df.iloc[30]
input_data_dict = dict(zip(input_df.columns, input_data.astype(str)))
new_obj = pd.DataFrame(columns=input_data_dict.keys()) 
for key in input_data_dict.keys():
	new_obj.loc[0,key] = input_data_dict[key]
processed_obj = hp.data_processing_pipeline(new_obj,3)  
print(processed_obj)

project_name = 'house_price'

model_output = os.path.join(cf.TRAINED_MODEL_PATH, project_name)
modelfile = 'house_price_model_' + version + '.pkl'
model = joblib.load(model_output + '/' + modelfile)
prediction = np.array2string(model.predict(processed_obj))
print(prediction)


print("\n\n")
# Predict data
post_url = "http://127.0.0.1:5000/house_price/predict" 
response = requests.post(post_url, json=input_data_dict)
#response_dict = ast.literal_eval(response.text)
print(response.text)



'''









'''