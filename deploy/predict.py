import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st

# Load models and data structures
def load_model_data():
    with open('xgb.pkl', 'rb') as file:
        xgb = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('ohe.pkl', 'rb') as file:
        ohe = pickle.load(file)
    with open('n_col.txt', 'r') as file:
        n_col = json.load(file)
    with open('c_col.txt', 'r') as file:
        c_col = json.load(file)
    return xgb, scaler, ohe, n_col, c_col

xgb, scaler, ohe, n_col, c_col = load_model_data()

def run():
    st.title("Sales Prediction")
    with st.form(key='final'):
        # Example options for select boxes. Replace these with your actual options.
        shipmode_options = ['First Class', 'Second Class', 'Standard Class', 'Same Day']
        segment_options = ['Consumer', 'Corporate', 'Home Office']
        region_options = ['South', 'East', 'Central', 'West']
        category_options = ['Furniture', 'Office Supplies', 'Technology']
        
        shipmode = st.selectbox('Ship Mode', options=shipmode_options, index=2)
        segment = st.selectbox('Segment', options=segment_options)
        city = st.text_input('City')
        state = st.text_input('State')
        region = st.selectbox('Region', options=region_options)
        category = st.selectbox('Category', options=category_options)
        sub_category = st.text_input('Sub Category')  # Consider selectbox if you have a fixed list of subcategories.
        product_name = st.text_input('Product Name')
        
        # For numerical inputs
        sales = st.number_input('Sales', min_value=0.0, value=100.0)
        quantity = st.number_input('Quantity', min_value=1, value=1)
        discount = st.number_input('Discount', min_value=0.0, value=0.0, step=0.01)
        
        # For date input
        order_date = st.date_input('Order Date')
        ordy, ordm, ordd = order_date.year, order_date.month, order_date.day
        
        # Additional numerical inputs
        isweekend = st.select_slider('Is Weekend', options=[0, 1], value=0)
        sd = st.number_input('Shipping Duration', min_value=0, value=2)
        unit_price = st.number_input('Unit Price', min_value=0.0, value=0.0)
        pcluster = st.number_input('Price Cluster', min_value=0, value=0)
        distinct_cluster_label = st.text_input('Distinct Cluster Label')

        submitted = st.form_submit_button('Predict')

        if submitted:
            df_inf = pd.DataFrame([{
                'shipmode': shipmode, 'segment': segment, 'city': city, 'state': state,
                'region': region, 'category': category, 'sub_category': sub_category,
                'product name': product_name, 'sales': sales, 'quantity': quantity,
                'discount': discount, 'order year': ordy, 'order month': ordm,
                'order day': ordd, 'is weekend': isweekend, 'shipping duration': sd,
                'unit_price': unit_price, 'pcluster': pcluster, 'distinct_cluster_label': distinct_cluster_label,
            }])
            # Data preprocessing and prediction steps remain the same
            df_inf_n = df_inf[n_col]
            df_inf_c = df_inf[c_col]
            df_inf_n_scaled = scaler.transform(df_inf_n)
            df_inf_c_encoded = ohe.transform(df_inf_c).toarray()
            df_inf_final = np.concatenate([df_inf_n_scaled,df_inf_c_encoded], axis = 1)
            y_pred_inf = xgb.predict(df_inf_final) #from the data, the profit will be 35.6
            a = round(y_pred_inf[0],2)
            
    st.write('Profit:', a )

if __name__ == '__main__':
    run()
