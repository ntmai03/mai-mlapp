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
from src.analysis.house_price_model import HousePrice
from src import config as cf



# read original data and split into train test split
input_df = dm.read_csv_file(cf.DATA_PATH + '/' + 'input', "kc_house_data.csv", "local")
houseprice = HousePrice()
houseprice.train_test_data('input', "kc_house_data", "local")
file_path = cf.DATA_PATH + '/' + 'input'
train_df = dm.read_csv_file(file_path, "kc_house_data_train_" + cf.house_price_version + ".csv", "local")
test_df = dm.read_csv_file(file_path, "kc_house_data_test_" + cf.house_price_version + ".csv", "local")
print(input_df.shape, train_df.shape, test_df.shape)


cleaned_train_df = houseprice.clean_data(train_df)
cleaned_test_df = houseprice.clean_data(test_df)
X_train, X_test = houseprice.data_processing_pipeline()
y_train = cleaned_train_df['price']
print(X_train.shape, X_test.shape)

result = houseprice.train_xgboost(X_train, y_train)
print(result.score(X_train, y_train))

# predict new data
#input_data = input_df.iloc[20]
input_data = input_df.iloc[30]
input_data_dict = dict(zip(input_df.columns, input_data.astype(str)))
new_obj = pd.DataFrame(columns=input_data_dict.keys()) 
for key in input_data_dict.keys():
	new_obj.loc[0,key] = input_data_dict[key]

processed_obj = houseprice.data_processing(new_obj)  
print(processed_obj)

project_name = 'house_price'
model_output = os.path.join(cf.TRAINED_MODEL_PATH, project_name)
modelfile = 'house_price_model.pkl'
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
test_url = "http://127.0.0.1:5000/house_price/test_data"  # Replace with your API URL
test_response = requests.get(test_url)
test_data = test_response.json()
print(test_data)

# Input data of new object
post_url = "http://127.0.0.1:5000/house_price/test_input" 
#input_df = dm.read_csv_file('', "kc_house_data.csv", "local")

input_data = input_df.iloc[20]
input_data_dict = dict(zip(input_df.columns, input_data.astype(str)))
#print(input_data_dict)
response = requests.post(post_url, json=input_data_dict)
response_dict = ast.literal_eval(response.text)
new_obj = response_dict['input_data']
#new_obj = pd.DataFrame.from_dict(new_obj)
print(new_obj)

print("\n\n")
# Predict data
post_url = "http://127.0.0.1:5000/house_price/predict" 
response = requests.post(post_url, json=input_data_dict)
#response_dict = ast.literal_eval(response.text)
print(response.text)
'''


