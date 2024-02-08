import streamlit as st
import predict
import eda_fin


navigation =st.sidebar.selectbox('Pilih Halaman: ',('EDA','Predict Sales'))

if navigation == "EDA": 
    eda_fin.run()
else:
    predict.run()