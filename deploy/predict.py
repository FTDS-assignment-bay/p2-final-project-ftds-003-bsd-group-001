import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st

with open('xgb.pkl', 'rb') as file_1: 
  xgb = pickle.load(file_1)
with open('scaler.pkl', 'rb') as file_2: 
 scaler = pickle.load(file_2)
with open('ohe.pkl', 'rb') as file_3: 
  ohe = pickle.load(file_3)
with open('n_col.txt','r') as file_4:
 n_col = json.load(file_4)
with open('c_col.txt', 'r') as file_5:
 c_col = json.load(file_5)


def run():
  
    with st.form(key = 'final'): 
        shipmode = st.text_input('shipmode= ')
        segment = st.text_input('segment = ')
        city = st.text_input('city = ')
        state = st.text_input('state = ')
        region = st.text_input('region = ')
        category = st.text_input('category = ')
        sub_category = st.text_input('sub category = ')
        product_name = st.text_input('product name = ')
        sales = st.number_input('sales', min_value = 0.00, value = None, help ='Sales')
        quantity = st.number_input('qty', min_value = 0.00, value = None)
        discount = st.number_input('discount', min_value = 0.00, value = None)
        ordy = st.number_input('order year', min_value = 0.00, value = None)
        ordm = st.number_input('order month', min_value = 0.00, value = None)
        ordd = st.number_input('order day', min_value = 0.00, value = None)
        orddw = st.number_input('order day of week', min_value = 0.00, value = None)
        isweekend = st.slider('Is weekend = ',min_value = 0,max_value = 1, help = '0 = No, 1 = Yes')
        sd = st.number_input('shipping duration', min_value = 0.00, value = None)
        unit_price = st.number_input('unit price', min_value = 0.00, value = None)
        pcluster = st.number_input('price cluster', min_value = 0.00, value = None)
        distinct_cluster_label = st.text_input('distinct cluster label = ')

        submitted = st.form_submit_button('predict')

        df_inf = {
        'shipmode': shipmode,
        'segment':segment,
        'city':city,
        'state':state,
        'region':region,
        'category':category,
        'sub_category':sub_category,
        'product name':product_name,
        'sales':sales,
        'quantity':quantity,
        'discount':discount,
        'order year':ordy,
        'order month':ordm,
        'order day':ordd,
        'order day week':orddw,
        'is weekend':isweekend,
        'shipping duration':sd,
        'unit_price':unit_price,
        'pcluster':pcluster,
        'distinct_cluster_label':distinct_cluster_label,

        }
        
        df_inf1 = pd.DataFrame([df_inf])
        df_inf1

        if submitted:
            df_inf_n = df_inf1[n_col]
            df_inf_c = df_inf1[c_col]
            df_inf_n_scaled = scaler.transform(df_inf_n)
            df_inf_c_encoded = ohe.transform(df_inf_c).toarray()
            df_inf_final = np.concatenate([df_inf_n_scaled,df_inf_c_encoded], axis = 1)
            y_pred_inf = xgb.predict(df_inf_final) #from the data, the profit will be 35.6
            a = round(y_pred_inf[0],2)
            
    st.write('Profit:', a )

if __name__ == '__main__':
    run()