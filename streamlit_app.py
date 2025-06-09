import streamlit as st
import numpy as np

from PIL import Image
from multipage import MultiPage
from page import introduction
from page import house_price_streamlit

# Config layout
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
display = Image.open('image/common/DataScienceBanner.jpg')
display = np.array(display)

col1, col2, col3 = st.columns([2,5,2])

with col1:
   st.write("")

with col2:
   st.image(display)

with col3:
   st.markdown("written by [Mai Nguyen](https://www.linkedin.com/in/ntmai03/)")

# Create an instance of the app
app = MultiPage(col1)
# Add all applications here
app.add_page("Select Application", introduction.app)


'''
pages = ['Introduction',
         'House Price',
         'Coin Trading Bot']

page_option = st.sidebar.selectbox('', pages)
st.sidebar.header('')

if(page_option == 'Introduction'):
   st.write('Introduction')
if(page_option == 'House Price'):
   house_price_streamlit.app()
if(page_option == 'Coin Trading Bot'):
   st.write('Coin Trading Bot')
'''

