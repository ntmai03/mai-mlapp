import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from src import config as cf
from src.analysis.coin_trading import Coin_Trading 

import datetime
from datetime import timedelta

############################################### Main flows #######################################################
def app():
    st.sidebar.subheader('Select function')
    task_type = ['Introduction',
                 'Add new symbol',
                 'Trading']
    task_option = st.sidebar.selectbox('', task_type)
    st.sidebar.header('')


    #============================================= Introduction ========================================
    if task_option == 'Introduction':
        st.markdown('<p style="color:Green; font-size: 25px;"> Business Objective</p>', unsafe_allow_html=True)


    #===================================== PART II:  ADD AND DISPLAY COIN LIST ===================================
    if (task_option == 'Add new symbol'):
        st.write("Input a new symbol that exists on Binance and click Add button to add this symbol to the symbol list")
        new_symbol = st.sidebar.text_input('Input new symbol').upper()
        if st.sidebar.button('Add'):
            cf.data['BITCOIN_SYMBOL'].append(new_symbol)
            cf.update_yaml_config_file(cf.data)

    # display list of coins, list of coins is stored in file config.yml
    SYMBOL = cf.data['BITCOIN_SYMBOL']
    selected_symbol = st.sidebar.selectbox("Select symbol", SYMBOL)



    #============================================= Trading ========================================
    if task_option == 'Trading':
        #if (selected_symbol != 'Select symbol'):
            TODAY = datetime.date.today() 
            start_time = TODAY - datetime.timedelta(cf.data['default_start_trading'])
            rsi_period = st.sidebar.number_input('rsi_period',1, 20, 6)
            sma_period = st.sidebar.number_input('sma_period',1, 20, 6)
            lower_threshold = st.sidebar.number_input('lower_threshold',1, 100, 10)
            upper_threshold = st.sidebar.number_input('upper_threshold',1, 100, 90)
            cutloss_flag = st.sidebar.number_input('cutloss_flag',0, 1, 1)
            increase_flag = st.sidebar.number_input('increase_flag',0, 1, 1)
            position = st.sidebar.number_input('position',0, 0, 1)
            units = st.sidebar.number_input('units',0, 2, 2)

            bar_length = "15m"
            rsi_limit1 = 1
            rsi_limit2 = 1
            cutloss_th=1.2

            if(st.sidebar.button('Start Trading')):
                ct = Coin_Trading(symbol=selected_symbol)
                #ct.rsi_trading(start_time, rsi_period, sma_period, lower_th, upper_th, bar_length, units, position)
                ct.rsi_trading(start_time, bar_length, rsi_period, sma_period, lower_threshold, upper_threshold, 
                    rsi_limit1=rsi_limit1,rsi_limit2=rsi_limit2,position = position, cutloss_flag=1, cutloss_th=cutloss_th, increase_flag=1)