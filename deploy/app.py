# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Assuming the necessary EDA functions are defined in eda.py and imported here
from eda import (average_sales_by_region, average_sales_and_profit_over_time,
                 segment_vs_region_distribution, sales_vs_profit_across_segments,
                 category_composition_for_profit_and_sales)

from prediction import make_prediction

# In your model training script and your Streamlit app script (app.py)
from transformers import UnitPriceTransformer, KMeansAndLabelTransformer, DynamicOneHotEncoder

# Load the dataset for EDA
@st.cache_data
def load_data():
    return pd.read_csv('superstore_clean.csv')

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "EDA", "Make a Prediction"])

if selection == "Home":
    st.title("Welcome to the Superstore Sales Dashboard")

elif selection == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    # Display EDA plots directly here or call a function that does
    average_sales_by_region(df)
    average_sales_and_profit_over_time(df)
    segment_vs_region_distribution(df)
    sales_vs_profit_across_segments(df)
    category_composition_for_profit_and_sales(df)

elif selection == "Make a Prediction":
    st.title("Make a Sales Prediction")
    with st.form("input_form"):
        # Simplify input fields based on what's actually used
        sales = st.number_input('Sales', value=100.0, format="%.2f")
        quantity = st.number_input('Quantity', value=2, format="%d")
        discount = st.number_input('Discount', value=0.0, format="%.2f")
        sub_category = st.selectbox('Sub-Category', ['Bookcases', 'Chairs', 'Labels', 'Tables', 'Storage', 'Furnishings', 'Art', 'Phones', 'Binders', 'Appliances', 'Paper', 'Accessories', 'Envelopes', 'Fasteners', 'Supplies', 'Machines', 'Copiers'])

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_features = pd.DataFrame([[sales, quantity, discount, sub_category]], columns=['sales', 'quantity', 'discount', 'sub_category'])
            predicted_profit = make_prediction(input_features)
            st.write(f'Predicted Profit: {predicted_profit:.2f}')

