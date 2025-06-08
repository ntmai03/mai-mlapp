import streamlit as st



pages = ['Introduction',
         'House Price',
         'Coin Trading Bot']

page_option = st.sidebar.selectbox('', pages)
st.sidebar.header('')

if(page_option == 'Introduction'):
   st.write('Introduction')
if(page_option == 'House Price'):
   st.write('House Price')
if(page_option == 'Coin Trading Bot'):
   st.write('Coin Trading Bot')

