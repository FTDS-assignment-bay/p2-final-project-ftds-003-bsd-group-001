import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
page_title = 'Sales Dataset',
layout = 'wide',
initial_sidebar_state = 'expanded'
)

def run():

    st.title('Exploratory Data Analysis of Sales')

if __name__ == '__main__':
    run()