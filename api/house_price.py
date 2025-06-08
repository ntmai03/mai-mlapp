# Important imports
import json
from api import app
#app.config.from_object("config.DevelopmentConfig")
from flask import escape, request, redirect, render_template, url_for, flash, jsonify
import pandas as pd
import numpy as np
import pickle as p
import json
import os
import sys
import joblib
from src import config as cf
from src.analysis import house_price_module as hp



# Route to house_price page
@app.route("/house_price", methods=["GET", "POST"])
def house_price():
    return 'Hello World'

@app.route('/house_price/test_data')
def test_data():
    return {
        'test_data': 'test_data',

    }    

@app.route('/house_price/test_input',methods=['POST'])
def test_input():

    #data = request.args.to_dict()
    #input_data = [x for x in request.form.values()]
    json_data = json.loads(request.data)
    return {        
        #'id': json_data['id'],
        #'bathrooms': json_data['bathrooms']
        'input_data': json_data
    }




@app.route('/house_price/predict',methods=['POST'])
def predict():
    json_data = json.loads(request.data)
    
    '''
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
    '''

    input_obj = dict({'price': [1.0],
                    'bedrooms': [json_data['bedrooms']],
                    'bathrooms': [json_data['bathrooms']],
                    'sqft_living': [json_data['sqft_living']],
                    'sqft_lot': [json_data['sqft_lot']],
                    'floors': [json_data['floors']],
                    'waterfront': [json_data['waterfront']],
                    'view': [json_data['view']],
                    'condition': [json_data['condition']],
                    'grade': [json_data['grade']],
                    'sqft_above': [json_data['sqft_above']],
                    'sqft_basement': [json_data['sqft_basement']],
                    'yr_built': [json_data['yr_built']],
                    'yr_renovated': [json_data['yr_renovated']],
                    'lat': [json_data['lat']],
                    'long': [json_data['long']],
                    'sqft_living15': [json_data['sqft_living15']],
                    'sqft_lot15': [json_data['sqft_lot15']],
                    'zipcode': [json_data['zipcode']],
                    'date':  [json_data['date']]
        })           
    


    new_obj = pd.DataFrame.from_dict(input_obj)
    processed_obj = hp.data_processing_pipeline(new_obj,3)    

    
    modelfile = 'model/house_price/house_price_model_v2.pkl'
    #model = p.load(open(modelfile, 'rb'))
    model = joblib.load(modelfile)
    prediction = np.array2string(model.predict(processed_obj))
        
    return {
        'input_data': new_obj.to_dict(),
        'prediction': prediction
    }

    

