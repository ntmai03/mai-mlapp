import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import joblib
import pickle
import sys
from pathlib import Path
import os
# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import ast

from src import config as cf
from src.util import data_manager as dm
from src.model import house_price_module as hp 
#from src.util import regression_util as regu
project_name = 'house_price'
version = 'v2'


############################################### Main flows #######################################################
def app():
    st.sidebar.subheader('Select function')
    task_type = ['Introduction',
                 'Data Understanding',
                 'Exploratory Data Analysis',
                 'Data Processing',
                 'Predictive Model',
                 'Prediction']
    task_option = st.sidebar.selectbox('', task_type)
    st.sidebar.header('')


    #============================================= Introduction ========================================
    if task_option == 'Introduction':
        st.markdown('<p style="color:Green; font-size: 25px;"> Business Objective</p>', unsafe_allow_html=True)
        st.write("This is a demo tool to predict house price based on the characteristics of the house such as how many rooms the home has or how much crime there is in the area plus a whole bunch of factors")
        st.markdown('<p style="color:Green; font-size: 25px;"> Machine Learning Model</p>', unsafe_allow_html=True)
        st.write("It is a regression problem, the project employ xgboost technique to build the predictive model that allows to predict price of a given house")
    
    if task_option == 'Data Processing':
        hp.train_test_data("kc_house_data", "local",version)
        train_df = dm.read_csv_file(os.path.join(cf.DATA_PATH, 'input'), "kc_house_data_train_" + version + ".csv", "local")
        test_df = dm.read_csv_file(os.path.join(cf.DATA_PATH, 'input'), "kc_house_data_test_" + version + ".csv", "local")
        cleaned_train_df = hp.clean_data(train_df)
        cleaned_test_df = hp.clean_data(test_df)
        X_train = hp.data_processing_pipeline(train_df,1)
        X_test = hp.data_processing_pipeline(test_df,2)
        y_train = cleaned_train_df['price']
        y_test = cleaned_test_df['price']
        st.write(X_train.shape,X_test.shape)
        st.write(X_train.columns)
        st.write(X_test.columns)

        result = hp.train_xgboost(X_train, y_train, version)
        st.write(result.score(X_train, y_train))
        st.write(result.score(X_test, y_test))        

    if task_option == 'Prediction':
        st.write("#### Input your data for prediction")
        bedrooms = st.text_input("Num of bedrooms", '3')
        bathrooms = st.text_input("Num of bathrooms", '1')
        sqft_living = st.text_input("sqft_living", '1180')
        sqft_lot = st.text_input("sqft_lot", '5650')
        floors = st.text_input("floors", '1')
        waterfront = st.text_input("waterfront", '0')
        view = st.text_input("view", '0')
        condition = st.text_input("condition", '3')
        grade = st.text_input("grade", '7')
        sqft_above = st.text_input("sqft_above", '1180')
        sqft_basement = st.text_input("sqft_basement", '0')
        yr_built = st.text_input("yr_built", '1955')
        yr_renovated = st.text_input("yr_renovated", '0')
        lat_corr = st.text_input("lat", '47.5112')
        long_corr = st.text_input("long", '-122.2570')
        sqft_living15 = st.text_input("sqft_living15", '1340')
        sqft_lot15 = st.text_input("sqft_lot15", '5650')
        date = st.text_input("date", '20141013T000000')
        zipcode = st.text_input("zipcode", '98178')

        if st.button("Predict"):
            new_obj = dict({'price':  1,   #221900.0
                            'bedrooms':[bedrooms],
                            'bathrooms':[bathrooms],
                            'sqft_living':[sqft_living],
                            'sqft_lot':[sqft_lot],
                            'floors':[floors],
                            'waterfront':[waterfront],
                            'view':[view],
                            'condition':[condition],
                            'grade':[grade],
                            'sqft_above':[sqft_above],
                            'sqft_basement':[sqft_basement],
                            'yr_built':[yr_built],
                            'yr_renovated':[yr_renovated],
                            'lat':[lat_corr],
                            'long':[long_corr],
                            'sqft_living15':[sqft_living15],
                            'sqft_lot15':[sqft_lot15],
                            'zipcode':[zipcode],
                            'date': [date]
                })
            new_obj = pd.DataFrame.from_dict(new_obj)
            processed_obj = hp.data_processing_pipeline(new_obj,3)  
            model_output = os.path.join(cf.TRAINED_MODEL_PATH, project_name)
            modelfile = 'house_price_model_' + version + '.pkl'
            model = joblib.load(os.path.join(model_output, 'house_price_model_v2.pkl'))
            # model = pickle.load(open(os.path.join(model_output, 'house_price_model_v2.pkl'), "rb"))
            prediction = np.array2string(model.predict(processed_obj))
            st.write("**Predicted Price**: ", model.predict(processed_obj)[0])            