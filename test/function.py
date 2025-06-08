# config
from pathlib import Path
import os
import sys

# Dataframe manipulation
import pandas as pd
import numpy as np
import pickle as p
import json
import joblib

# Preprocessing
from sklearn.model_selection import train_test_split


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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import xgboost as xgb


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


# Evaluation metrics for Regression 
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


########################### Define paths to store calculated data #################################  
house_price_encode_ordinal_label = os.path.join(ANALYSIS_PATH, 'house_price_encode_ordinal_label.npy')
house_price_median_imputer = os.path.join(ANALYSIS_PATH, 'house_price_median_imputer.npy')
house_price_knn_imputer = os.path.join(ANALYSIS_PATH, 'house_price_knn_imputer.npy')
house_price_scaler = os.path.join(ANALYSIS_PATH, 'house_price_scaler.pkl')
house_price_dummy_vars = os.path.join(ANALYSIS_PATH, 'house_price_dummy_vars.npy')

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
             'bedrooms': 'int64',
             'bathrooms': 'int64',
             'sqft_living': 'int64',
             'sqft_lot': 'int64',
             'floors': 'int64',
             'waterfront': 'int64',
             'view': 'int64',
             'condition': 'int64',
             'grade': 'int64',
             'sqft_above': 'int64',
             'sqft_basement': 'int64',
             'yr_built': 'int64',
             'yr_renovated': 'int64',
             'lat': 'float64',
             'long': 'float64',
             'sqft_living15': 'int64',
             'sqft_lot15': 'int64'}

# Define variables
TARGET = 'price'
TEXT_VARS = []
CATEGORICAL_VARS = ['zipcode']   
NUMERICAL_VARS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                  'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                  'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                  'sqft_living15', 'sqft_lot15']
TEMPORAL_VARS = 'date'
DISCRETE_VARS = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition']
CONTINUOUS_VARS = ['sqft_living', 'sqft_lot', 'grade', 'sqft_above', 'sqft_basement', 
                    'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
GEOGRAPHICAL_VARS = ['long', 'lat']


TRAIN_NUMERICAL_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                  'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                  'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                  'sqft_living15', 'sqft_lot15', 'sqft_ratio', 'zipcode']
TRAIN_CATEGORICAL_VARS = ['season']
DUMMY_VARS = []
# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = []
# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = []
# variables to log transform
NUMERICALS_LOG_VARS = []
TRAIN_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
              'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
              'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_ratio', 
              'zipcode', 'season_spring', 'season_summer', 'season_winter']

OUTLIER_VARS = ['sqft_lot', 'sqft_above', 'sqft_lot15','sqft_basement', 'bedrooms']
OUTLIER_DICT = {'sqft_lot':43560, 'sqft_lot15':19647, 'sqft_above':4070.0, 'sqft_basement':1580, 'bedrooms':10}
NO_MULTICOLINEARTITY_VARS = ['sqft_living',  'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_basement',
                             'yr_built', 'yr_renovated', 'lat', 'long', 'zipcode', 'season_spring', 'season_summer', 'season_winter']



########################### Define functions #################################  
def read_csv_file(file_name=None):
    data = pd.read_csv(DATA_PATH + '/' + file_name)
    
    return data


def split_data(X, y, test_size=0.2, random_state=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test  



def clean_data(df):

    data = df.copy()

    # Rename columns
    data.rename(columns=FEATURE_MAP, inplace=True)

    # select columns of interest
    data = data[SELECTED_VARS]

    # remove invalid rows
    data = data[data[TARGET] > 0]

     # data type conversion
    for key in DATA_TYPE:
        data[key] = data[key].astype(DATA_TYPE[key])

    # Remove duplicated data
    data = data.drop_duplicates(keep = 'last')

    # Reset index
    data = data.reset_index(drop=True)

    return data


def create_season(df, var):

    data = df.copy()

    data[var] = pd.to_datetime(data[var])
    data['month'] = data[var].apply(lambda var:var.month)
    data['year'] = data[var].apply(lambda var:var.year)
    data['season'] = 'NA'
    data.loc[data.month.isin([12,1,2]), 'season'] = 'winter'
    data.loc[data.month.isin([3,4,5]), 'season'] = 'spring'
    data.loc[data.month.isin([6,7,8]), 'season'] = 'summer'
    data.loc[data.month.isin([9,10,11]), 'season'] = 'autum'

    return data


def create_sqft_ratio(df, var1, var2):

    data = df.copy()

    data['sqft_ratio'] = data[var1]/data[var2]

    return data

def replace_categories(df, var, target):

    data = df.copy()

    ordered_labels = data.groupby([var])[target].mean().sort_values().index
    ordinal_label = {k:i for i,k in enumerate(ordered_labels, 0)}

    return ordinal_label


def encode_categorical_ordinal(df, var_list, target, train_flag=0):

    data = df.copy()

    if(train_flag == 1):
        ordinal_label_dict = {}
        for var in var_list:
            ordinal_label = replace_categories(data, var, target)
            ordinal_label_dict[var]= ordinal_label
        # save the dictionary
        np.save(house_price_encode_ordinal_label, ordinal_label_dict)
    else:
        ordinal_label_dict = np.load(house_price_encode_ordinal_label, allow_pickle=True).item()

    for var in var_list:
        ordinal_label = ordinal_label_dict[var]
        data[var] = data[var].map(ordinal_label)

    return data



def impute_na_median(df, var_list, train_flag=0):

    data = df.copy()

    if(train_flag == 1):
        median_var_dict = {}
        for var in var_list:
            median_val = data[var].median()
            median_var_dict[var] = median_val
        # save result
        np.save(house_price_median_imputer, median_var_dict)
    else:
        median_var_dict = np.load(house_price_median_imputer, allow_pickle=True).item()

    for var in var_list:
        median_var = median_var_dict[var]
        data[var].fillna(median_val, inplace=True)

    return data


def scaling_data(df, var_list, train_flag=0):

    data = df.copy()

    # fit scaler
    scaler = MinMaxScaler()
    scaler.fit(data[var_list])

    # persist the model for future use
    if(train_flag == 1):
        joblib.dump(scaler, house_price_scaler)
    scaler = joblib.load(house_price_scaler)

    data = pd.DataFrame(scaler.transform(data[var_list]), columns=var_list)

    return data


def create_dummy_vars(df, var_list, train_flag=0):  
    
    data = df.copy()
    data_categorical = pd.DataFrame()
    for var in var_list:
        data_dummies = pd.get_dummies(data[var], prefix=var, prefix_sep='_',drop_first=True)  
        data_categorical = pd.concat([data_categorical, data_dummies], axis=1)    
    
    if(train_flag == 1):
        train_dummy = list(data_categorical.columns)
        pd.Series(train_dummy).to_csv(house_price_dummy_vars, index=False)
    else:
        test_dummy = list(data_categorical.columns)
        train_dummy = pd.read_csv(house_price_dummy_vars)
        train_dummy.columns = ['Name']
        train_dummy = list(train_dummy.Name.values)   
        
    for col in train_dummy:
        if col not in data_categorical:
            data_categorical[col] = 0
    if(len(DUMMY_VARS) > 0):
        data_categorical = data_categorical[DUMMY_VARS] 
    
    return data_categorical


def data_processing_pipeline(df, train_flag=0):

    df = clean_data(df)
    df = create_season(df, TEMPORAL_VARS)
    df = create_sqft_ratio(df, 'sqft_living', 'sqft_living15')
    df = encode_categorical_ordinal(df, CATEGORICAL_VARS, TARGET, train_flag)
    df = impute_na_median(df, NUMERICAL_VARS_WITH_NA, train_flag)

    data_scaled = scaling_data(df, TRAIN_NUMERICAL_VARS, train_flag)
    data_categorical = create_dummy_vars(df, TRAIN_CATEGORICAL_VARS, train_flag)
    df = pd.concat([data_scaled,data_categorical], axis=1)
    df = df[TRAIN_VARS]

    return df


def train_regression_sklearn(X_train, y_train):
    # Train model
    model = LinearRegression(fit_intercept = True)
    model.fit(X_train[TRAIN_VARS], y_train)
    model = model

    # Result Summary Table
    summary_table = pd.DataFrame(columns=['FeatureName'], data=TRAIN_VARS)
    summary_table['Coefficient'] = np.transpose(model.coef_)
    summary_table.index = summary_table.index + 1
    summary_table = summary_table.sort_index()

    return summary_table    





######################################### Define class ####################################
class HousePrice:

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
                 'bedrooms': 'int64',
                 'bathrooms': 'int64',
                 'sqft_living': 'int64',
                 'sqft_lot': 'int64',
                 'floors': 'int64',
                 'waterfront': 'int64',
                 'view': 'int64',
                 'condition': 'int64',
                 'grade': 'int64',
                 'sqft_above': 'int64',
                 'sqft_basement': 'int64',
                 'yr_built': 'int64',
                 'yr_renovated': 'int64',
                 'lat': 'float64',
                 'long': 'float64',
                 'sqft_living15': 'int64',
                 'sqft_lot15': 'int64'}

    # Define variables
    TARGET = 'price'
    TEXT_VARS = []
    CATEGORICAL_VARS = ['zipcode']   
    NUMERICAL_VARS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                      'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                      'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                      'sqft_living15', 'sqft_lot15']
    TEMPORAL_VARS = 'date'
    DISCRETE_VARS = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition']
    CONTINUOUS_VARS = ['sqft_living', 'sqft_lot', 'grade', 'sqft_above', 'sqft_basement', 
                        'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    GEOGRAPHICAL_VARS = ['long', 'lat']


    TRAIN_NUMERICAL_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                      'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                      'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                      'sqft_living15', 'sqft_lot15', 'sqft_ratio', 'zipcode']
    TRAIN_CATEGORICAL_VARS = ['season']
    DUMMY_VARS = []
    # numerical variables with NA in train set
    NUMERICAL_VARS_WITH_NA = []
    # categorical variables with NA in train set
    CATEGORICAL_VARS_WITH_NA = []
    # variables to log transform
    NUMERICALS_LOG_VARS = []
    TRAIN_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
                  'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                  'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_ratio', 
                  'zipcode', 'season_spring', 'season_summer', 'season_winter']

    OUTLIER_VARS = ['sqft_lot', 'sqft_above', 'sqft_lot15','sqft_basement', 'bedrooms']
    OUTLIER_DICT = {'sqft_lot':43560, 'sqft_lot15':19647, 'sqft_above':4070.0, 'sqft_basement':1580, 'bedrooms':10}
    NO_MULTICOLINEARTITY_VARS = ['sqft_living',  'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_basement',
                                 'yr_built', 'yr_renovated', 'lat', 'long', 'zipcode', 'season_spring', 'season_summer', 'season_winter']

    def __init__(self):
 
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.target = None
        self.processed_X_train = None
        self.processed_X_test = None    


    def clean_data(self, df):

        data = df.copy()

        # Rename columns
        data.rename(columns=self.FEATURE_MAP, inplace=True)

        # select columns of interest
        data = data[self.SELECTED_VARS]

        # remove invalid rows
        data = data[data[self.TARGET] > 0]

         # data type conversion
        for key in self.DATA_TYPE:
            data[key] = data[key].astype(self.DATA_TYPE[key])

        # Remove duplicated data
        data = data.drop_duplicates(keep = 'last')

        # Reset index
        data = data.reset_index(drop=True)

        return data


    def create_season(self, df, var):

        data = df.copy()

        data[var] = pd.to_datetime(data[var])
        data['month'] = data[var].apply(lambda var:var.month)
        data['year'] = data[var].apply(lambda var:var.year)
        data['season'] = 'NA'
        data.loc[data.month.isin([12,1,2]), 'season'] = 'winter'
        data.loc[data.month.isin([3,4,5]), 'season'] = 'spring'
        data.loc[data.month.isin([6,7,8]), 'season'] = 'summer'
        data.loc[data.month.isin([9,10,11]), 'season'] = 'autum'

        return data


    def create_sqft_ratio(self, df, var1, var2):

        data = df.copy()

        data['sqft_ratio'] = data[var1]/data[var2]

        return data

    def replace_categories(self, df, var, target):

        data = df.copy()

        ordered_labels = data.groupby([var])[target].mean().sort_values().index
        ordinal_label = {k:i for i,k in enumerate(ordered_labels, 0)}

        return ordinal_label


    def encode_categorical_ordinal(self, df, var_list, target, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            ordinal_label_dict = {}
            for var in var_list:
                ordinal_label = self.replace_categories(data, var, target)
                ordinal_label_dict[var]= ordinal_label
            # save the dictionary
            np.save(house_price_encode_ordinal_label, ordinal_label_dict)
        else:
            ordinal_label_dict = np.load(house_price_encode_ordinal_label, allow_pickle=True).item()

        for var in var_list:
            ordinal_label = ordinal_label_dict[var]
            data[var] = data[var].map(ordinal_label)

        return data



    def impute_na_median(self, df, var_list, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            median_var_dict = {}
            for var in var_list:
                median_val = data[var].median()
                median_var_dict[var] = median_val
            # save result
            np.save(house_price_median_imputer, median_var_dict)
        else:
            median_var_dict = np.load(house_price_median_imputer, allow_pickle=True).item()

        for var in var_list:
            median_var = median_var_dict[var]
            data[var].fillna(median_val, inplace=True)

        return data


    def scaling_data(self, df, var_list, train_flag=0):

        data = df.copy()

        # fit scaler
        scaler = MinMaxScaler()
        scaler.fit(data[var_list])

        # persist the model for future use
        if(train_flag == 1):
            joblib.dump(scaler, house_price_scaler)
        scaler = joblib.load(house_price_scaler)

        data = pd.DataFrame(scaler.transform(data[var_list]), columns=var_list)

        return data


    def create_dummy_vars(self, df, var_list, train_flag=0):  
        
        data = df.copy()
        data_categorical = pd.DataFrame()
        for var in var_list:
            data_dummies = pd.get_dummies(data[var], prefix=var, prefix_sep='_',drop_first=True)  
            data_categorical = pd.concat([data_categorical, data_dummies], axis=1)    
        
        if(train_flag == 1):
            train_dummy = list(data_categorical.columns)
            pd.Series(train_dummy).to_csv(house_price_dummy_vars, index=False)
        else:
            test_dummy = list(data_categorical.columns)
            train_dummy = pd.read_csv(house_price_dummy_vars)
            train_dummy.columns = ['Name']
            train_dummy = list(train_dummy.Name.values)   
            
        for col in train_dummy:
            if col not in data_categorical:
                data_categorical[col] = 0
        if(len(self.DUMMY_VARS) > 0):
            data_categorical = data_categorical[self.DUMMY_VARS] 
        
        return data_categorical


    def data_processing_pipeline(self, df, train_flag=0):

        df = self.clean_data(df)
        df = self.create_season(df, self.TEMPORAL_VARS)
        df = self.create_sqft_ratio(df, 'sqft_living', 'sqft_living15')
        df = self.encode_categorical_ordinal(df, self.CATEGORICAL_VARS, self.TARGET, train_flag)
        df = self.impute_na_median(df, self.NUMERICAL_VARS_WITH_NA, train_flag)

        data_scaled = self.scaling_data(df, self.TRAIN_NUMERICAL_VARS, train_flag)
        data_categorical = self.create_dummy_vars(df, self.TRAIN_CATEGORICAL_VARS, train_flag)
        df = pd.concat([data_scaled,data_categorical], axis=1)
        df = df[self.TRAIN_VARS]

        return df

    def train_regression_sklearn(self):
        # Train model
        model = LinearRegression(fit_intercept = True)
        model.fit(self.processed_X_train[self.TRAIN_VARS], self.y_train)
        self.model = model

        # Result Summary Table
        summary_table = pd.DataFrame(columns=['FeatureName'], data=self.TRAIN_VARS)
        summary_table['Coefficient'] = np.transpose(model.coef_)
        summary_table.index = summary_table.index + 1
        summary_table = summary_table.sort_index()

        return summary_table        