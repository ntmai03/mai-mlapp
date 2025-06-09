# utitlity libraries
import sys
import os
from io import BytesIO
# to persist the model and the scaler
import joblib


# Scikit-Learn ≥0.20 is required
import sklearn

# Dataframe manipulation
import numpy as np
import pandas as pd

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

# Modelling Helpers:
from sklearn.preprocessing import Normalizer, scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, ShuffleSplit, cross_validate
from sklearn import model_selection
from sklearn.model_selection import train_test_split

# Regression
#import statsmodels.api as sm
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import xgboost as xgb

# Evaluation metrics for Regression 
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# user-defined functions
from src.util import data_manager as dm
#from src.util import regression_util as reu
from src import config as cf


########################### Define paths to store calculated data #################################  
version = 'v2'

filename = 'kc_house_data'
project_name = 'house_price'
input_path = os.path.join(os.path.join(cf.DATA_PATH, 'input'))
output_path = os.path.join(os.path.join(cf.DATA_PATH, 'output'))
model_path = os.path.join(cf.TRAINED_MODEL_PATH, project_name)
training_path =  os.path.join(cf.TRAINING_MODEL_PATH, project_name)

median_treatment_file = 'median_treatment_' + version + ".csv"
mode_treatment_file = 'mode_treatment_' + version + ".csv"
zipcode_treatment_file = 'zipcode_treatment_' + version + ".csv" 
dummy_vars_filename = 'dummy_vars_' + version + ".csv" 
zipcode_transform_filename = 'zipcode_transform_' + version + ".csv" 
final_train_vars_filename = 'final_train_vars_' + version + ".csv" 



######################################### Define variables used in class ####################################
# rename columns
FEATURE_MAP = {'date': 'date',
                'price': 'price'}

SELECTED_VARS = ['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                     'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                     'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 
                     'sqft_living15', 'sqft_lot15']

# data type conversion
DATA_TYPE = {'zipcode': 'str',
                 'date': 'object',
                 'price': 'float64',
                 'bedrooms': 'float64',
                 'bathrooms': 'float64',
                 'sqft_living': 'float64',
                 'sqft_lot': 'float64',
                 'floors': 'float64',
                 'waterfront': 'object',
                 'view': 'float64',
                 'condition': 'float64',
                 'grade': 'float64',
                 'sqft_above': 'float64',
                 'sqft_basement': 'float64',
                 'yr_built': 'float64',
                 'yr_renovated': 'float64',
                 'lat': 'float64',
                 'long': 'float64',
                 'sqft_living15': 'float64',
                 'sqft_lot15': 'float64'}

# Define variables
TARGET = 'price'
TRAIN_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
                  'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                  'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_ratio', 
                  'zipcode', 'season_spring', 'season_summer', 'season_winter']              
removed_vars = ['id']
target_var = 'price'
cat_vars = ['zipcode', 'waterfront']
time_vars = ['date']
text_vars = []
num_vars = [e for e in SELECTED_VARS if e not in removed_vars + [target_var] + cat_vars + time_vars + text_vars]


def train_test_data(file_name, type, version='v1', test_size=0.2, random_state=99):
    df = dm.read_csv_file(input_path, file_name + '.csv', type)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_filename = file_name + '_train_' + version + ".csv"
    test_filename = file_name + '_test_' + version + ".csv"
    dm.write_csv_file(input_path, train_filename, train_df, type)
    dm.write_csv_file(input_path, test_filename, test_df, type)


def clean_data(df):

    data = df.copy()

    # Rename columns
    data.rename(columns=FEATURE_MAP, inplace=True)
    data.columns = data.columns.str.lower()

    # select columns of interest
    data = data[SELECTED_VARS]

    # data type conversion
    for key in DATA_TYPE:
        data[key] = data[key].astype(DATA_TYPE[key])
    for key in time_vars:
        data[key] = pd.to_datetime(data[key])            

    # remove invalid rows
    data = data[data[TARGET] > 0]            

    # Remove duplicated data
    data = data.drop_duplicates(keep = 'last')

    # Reset index
    data = data.reset_index(drop=True)

    return data


def preprocessing_training_data(df, num_vars, cat_vars, scale_flag=0, version='v1'):
        
    # 1. Numerical Data - fixing missing data
    med = df[num_vars].median()
    df = df.fillna(med)
    med = pd.DataFrame(med).reset_index()
    med.columns = ['feature','median_val']        
    dm.write_csv_file(training_path, median_treatment_file, med, 'local')
              

    # 2. Categorical data - fixing missing data
    if cat_vars is not None:
        mode = df[cat_vars].mode().transpose()
        df = df.fillna(mode)
        mode = pd.DataFrame(mode).reset_index()
        mode.columns = ['feature', 'mode_val']
        dm.write_csv_file(training_path, mode_treatment_file, mode, 'local')        
    
    # 3. transform zipcode from categorical var to numerical var   
    ordered_label = df.groupby(['zipcode'])[TARGET].median().sort_values().index
    ordinal_label = {k:i for i, k in enumerate(ordered_label, 0)}
    df['zipcode'] = df['zipcode'].map(ordinal_label)
    zipcode_df = pd.DataFrame(index=ordinal_label.keys(), data=ordinal_label.values()).reset_index()
    zipcode_df.columns = ['feature', 'order_val']
    dm.write_csv_file(training_path, zipcode_treatment_file, zipcode_df, 'local') 
    num_vars = num_vars + ['zipcode']
    cat_vars = [e for e in cat_vars if e not in ['zipcode']]
    
    # 3. Temporal data
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['season'] = 'N/A'
    df['season'] = np.where(df['month'].isin([12,1,2]), 'winter', df['season'])
    df['season'] = np.where(df['month'].isin([3,4,5]), 'spring', df['season'])
    df['season'] = np.where(df['month'].isin([6,7,8]), 'summer', df['season'])
    df['season'] = np.where(df['month'].isin([9,10,11]), 'autumn', df['season'])
    
    # 4. Create dummy vars
    dummy_vars = ['season']
    dummy_df = pd.DataFrame()
    for var in dummy_vars:
        temp_df = pd.get_dummies(df[var], prefix=var, prefix_sep='_')
        dummy_df = pd.concat([dummy_df, temp_df], axis=1)
    dummy_vars = list(dummy_df.columns)
    dummy_vars_df = pd.DataFrame(dummy_vars).reset_index()
    dummy_vars_df.columns = ['index','feature']      
    dm.write_csv_file(training_path, dummy_vars_filename, dummy_vars_df, 'local') 
    
    # 5. Create new/derived features
    df['sqft_total'] = df['sqft_lot'] + df['sqft_living']
    ratio_vars = ['sqft_living','sqft_lot','sqft_total']
    for var in ratio_vars:
        ordered_label = df.groupby(['zipcode'])[var].median()
        ordinal_label = dict(zip(ordered_label.index, ordered_label.values))
        df['avg_' + var] = df['zipcode'].map(ordinal_label)
        df[var + '_ratio'] = df[var]/df['avg_' + var]
        num_vars = num_vars + ['avg_'+var, var+'_ratio']
        med_df = pd.DataFrame(index=ordinal_label.keys(), data=ordinal_label.values()).reset_index()
        med_df.columns = ['zipcode', 'median_val']
        dm.write_csv_file(training_path, zipcode_transform_filename, med_df, 'local')  
     
    final_train_vars = num_vars + cat_vars + dummy_vars
    final_train_vars = pd.DataFrame(final_train_vars).reset_index()
    final_train_vars.columns = ['index','feature']
    dm.write_csv_file(training_path, final_train_vars_filename, final_train_vars, 'local') 
    
    #6. Scale numeric data
    if scale_flag == 1:
        scaler = MinMaxScaler()
        scaler.fit(df[num_vars])
        joblib.dump(scaler, scaler_path)  
        df[num_vars] = pd.DataFrame(scaler.transform(df[num_vars]), columns=num_vars)
    
    return_df = pd.concat([df[num_vars], df[cat_vars], dummy_df], axis=1)
    return return_df



def preprocessing_test_data(df, num_vars, cat_vars, scale_flag=0, version='v1'):

    # 1. Numerical data
    train_filename = 'median_treatment_' + version + ".csv"   
    med_df = dm.read_csv_file(training_path, median_treatment_file, 'local')        
    med = pd.Series(med_df['median_val'])
    med.index = med_df['feature']
    df = df.fillna(med)

    # 2. Categorical data
    train_filename = 'mode_treatment_' + version + ".csv"   
    if cat_vars is not None:
        mode_df = dm.read_csv_file(training_path, mode_treatment_file, 'local')      
        mode = pd.Series(mode_df['mode_val'])
        mode.index = mode_df['feature']
        df = df.fillna(mode)
    # transform zipcode from categorical var to numerical var 
    zipcode_df = dm.read_csv_file(training_path, zipcode_treatment_file, 'local')
    zipcode_df['feature'] = zipcode_df['feature'].astype(str) 
    df['zipcode'] = df['zipcode'].astype(str)
    ordinal_label = dict(zip(zipcode_df['feature'], zipcode_df['order_val']))
    df['zipcode'] = df['zipcode'].map(ordinal_label)

    num_vars = num_vars + ['zipcode']
    cat_vars = [e for e in cat_vars if e not in ['zipcode']]

    # 3. Temporal data
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['season'] = 'N/A'
    df['season'] = np.where(df['month'].isin([12,1,2]), 'winter', df['season'])
    df['season'] = np.where(df['month'].isin([3,4,5]), 'spring', df['season'])
    df['season'] = np.where(df['month'].isin([6,7,8]), 'summer', df['season'])
    df['season'] = np.where(df['month'].isin([9,10,11]), 'autumn', df['season'])   
    
    # 4. Create dummy vars
    dummy_vars = ['season']
    dummy_df = pd.DataFrame()
    for var in dummy_vars:
        temp_df = pd.get_dummies(df[var], prefix=var, prefix_sep='_')
        dummy_df = pd.concat([dummy_df, temp_df], axis=1)
    dummy_vars = dm.read_csv_file(training_path, dummy_vars_filename, 'local')       
    dummy_vars = dummy_vars['feature']
    for var in dummy_vars:
        if var not in dummy_df.columns:
            dummy_df[var] = 0
          

    # 5. Create new/derived features
    df['sqft_total'] = df['sqft_lot'] + df['sqft_living']
    ratio_vars = ['sqft_living','sqft_lot','sqft_total']
    for var in ratio_vars:        
        med_df = dm.read_csv_file(training_path, zipcode_transform_filename, 'local')    
        ordinal_label = dict(zip(med_df['zipcode'], med_df['median_val']))
        df['avg_' + var] = df['zipcode'].map(ordinal_label)
        df[var + '_ratio'] = df[var]/df['avg_' + var]
        num_vars = num_vars + ['avg_'+var, var+'_ratio']   
    
    #6. Scale numeric data   
    if scale_flag == 1:
        scaler = joblib.load(scaler_path)
        df[num_vars] = pd.DataFrame(scaler.transform(df[num_vars]), columns=num_vars)

    return_df = pd.concat([df[num_vars], df[cat_vars], dummy_df], axis=1)            

    return return_df


def data_processing_pipeline(df,mode=1):

    '''
    removed_vars = []
    target_var = 'price'
    id_vars = ['id']
    cat_vars = ['zipcode', 'waterfront']
    time_vars = ['date']
    text_vars = []
    num_vars = [e for e in train_df.columns if e not in removed_vars + [target_var] + id_vars + cat_vars + time_vars + text_vars]
    '''

    if(mode == 1):
        cleaned_train_df = clean_data(df)
        return_df = preprocessing_training_data(cleaned_train_df, num_vars, cat_vars)
        train_filename = filename + '_train_' + version + ".csv"
        dm.write_csv_file(output_path, train_filename, return_df, type)
    elif(mode == 2):
        cleaned_test_df = clean_data(df)
        return_df = preprocessing_test_data(cleaned_test_df, num_vars, cat_vars)
        test_filename = filename + '_test_' + version + ".csv"
        dm.write_csv_file(output_path, test_filename, return_df, type)
    else:
        new_df = clean_data(df)
        return_df = preprocessing_test_data(new_df, num_vars, cat_vars) 

    return return_df




def data_processing(df):

    file_path = cf.DATA_PATH + '/' + 'input'
    train_df = dm.read_csv_file(file_path, "kc_house_data_train_" + cf.house_price_version + ".csv", "local")

    removed_vars = []
    target_var = 'price'
    id_vars = ['id']
    cat_vars = ['zipcode', 'waterfront']
    time_vars = ['date']
    text_vars = []
    num_vars = [e for e in train_df.columns if e not in removed_vars + [target_var] + id_vars + cat_vars + time_vars + text_vars]

    cleaned_data = clean_data(df)
    final_data = preprocessing_test_data(cleaned_data, num_vars, cat_vars)

    return final_data     


def train_xgboost(X_train, y_train, version='v1'):
    #xgb_model = xgb.XGBRegressor(max_depth=5,n_estimators=50)
    gbt_model = GradientBoostingRegressor()
    gbt_model.fit(X_train, y_train)

    # Tuning hyperparameters

    # Save model
    joblib.dump(gbt_model, model_path + '/' + 'house_price_model_' + version + '.pkl')  

    return gbt_model
    

