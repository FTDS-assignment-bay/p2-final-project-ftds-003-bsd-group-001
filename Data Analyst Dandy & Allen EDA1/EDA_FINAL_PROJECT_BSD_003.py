import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# set page configuration
st.set_page_config(
    page_title= 'Superstore_EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)


# create function for EDA
def run():

    # create title
    st.title('Superstore EDA')

    # create sub header
    st.subheader('Exploration Data Analysis for Superstore Dataset')

    # add image
    st.image('https://upload.wikimedia.org/wikipedia/en/thumb/7/7f/Headuarters.JPG/1200px-Headuarters.JPG', use_column_width=True,
             caption='Superstore')

    # create a description
    st.write('by FTDS-BSD-003-Group 1')
    st.write('# Introduction to EDA')

    # Magic Syntax
    st.write('''
    On this page, we will do Exploratory Data Analysis,
    Using the dataset containing Profit and Sales from Superstore.
    This dataset obtained from Kaggle
    ''')
    # create straight line
    st.markdown('---')

    # show dataframe
    df = pd.read_csv('superstore_clean.csv')
    st.dataframe(df)
    # Menampilkan jumlah baris dan kolom
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
 
    # create straight line
    st.markdown('---')
    st.subheader('Average Sales Across Different Regions')
    # Grouping by region and calculating mean sales
    df_bar = df[['region', 'sales']]
    df_bar = df_bar.groupby('region').mean().sort_values(by='sales', ascending=False)

    # Streamlit app
    # Plotting the BarChart with Seaborn using Streamlit
    fig, ax = plt.subplots(figsize=[14, 5])
    sns.barplot(x=df_bar.index, y='sales', data=df_bar, palette='viridis', ax=ax)

    # Adding labels to the bars
    for index, value in enumerate(df_bar['sales']):
        ax.text(index, value, str(int(value)), ha='center', va='bottom')

    # Display the plot using Streamlit
    st.pyplot(fig)
    plt.close(fig)  # Close the Matplotlib figure to avoid duplication
    st.write('''
    The bar chart visualizes the average sales across different regions, revealing how sales performance varies geographically. The chart indicates that the Central region has the highest average sales, followed by the South, East, and West regions. 
    This insight suggests regional preferences or operational strengths that could influence strategic decisions such as inventory distribution, marketing focus, and resource allocation to optimize sales performance in underperforming regions.
    ''')

    st.markdown('---')
    st.subheader('Average Sales and Profit over Time Period(2014-2018)')
    # First of all, we are going to take only the subset of data for our purpose. (To keep things simple)
    df_line = df[['order_date', 'sales', 'profit']].sort_values('order_date')  # Chronological Ordering
    df_line['order_date'] = pd.to_datetime(df_line['order_date'])  # Converting into DateTime
    df_line = df_line.groupby('order_date').mean()  # Groupby to get the average sales and profit on each day
    # Plotting the Line Chart using Streamlit
    fig, ax = plt.subplots(figsize=[13, 4])
    ax.plot(df_line.index, 'sales', data=df_line, color='green', label='Avg Sales')  # Avg sales over Time
    ax.plot(df_line.index, 'profit', data=df_line, color='#F05454', label='Avg Profit')  # Avg profit over Time
    # ax.set_title("Average Sales and Profit over Time Period (2014-2018)", size=20, pad=20)
    ax.legend()

    # Display the plot using Streamlit
    st.pyplot(fig)
    plt.close(fig)  # Close the Matplotlib figure to avoid duplication
    st.write('''
    The line chart presents a temporal view of average sales and profits from 2014 to 2018, offering insights into the 
    store's financial health and sales trends over time. Notably, while sales exhibit seasonal peaks and troughs, 
    profit trends might indicate underlying operational efficiencies or cost management strategies. 
    The visualization underscores the importance of aligning sales strategies with cost control to enhance profitability.
    ''')
       
    # create straight line
    st.markdown('---')
    # Display Countries Credit Score Comparison as a subheader
    st.subheader('Segment Vs Region Distribution')
    # Define target colors using Viridis colormap
    target_colors1 = plt.cm.viridis([0.2, 0.5, 0.8, 1.0])

    # Create bar chart
    fig = plt.figure(figsize=(25, 4))
    grid = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)

    # Bar Chart
    ax1 = fig.add_subplot(grid[0, :1])
    sns.countplot(x='segment', data=df, ax=ax1, hue='region', palette=target_colors1)
    st.pyplot(fig)
    plt.close(fig)
    st.write('''
    The count plot and pie chart illustrate the distribution of customer segments across different regions and the overall 
    composition of these segments. The visual analysis reveals a balanced distribution of segments across regions, with a 
    particular emphasis on the consumer segment's predominance. These insights can guide targeted marketing strategies and 
    product offerings to cater to the dominant segments in each region.''')

    # create straight line
    st.markdown('---')
        # Create a Streamlit app
    st.subheader('Sales vs Profit Across Different Customer Segments')

    # Taking a subset of data
    df_scatter = df[['sales', 'profit', 'segment']]

    # Visualizing the Scatter Plot
    fig, ax = plt.subplots(figsize=[20, 7])

    # Profit in the Y-axis, and Sales in the X. Hue will classify the dots according to Segment.
    # The size of the dots is according to the volume of "Sales".
    sns.scatterplot(x=df_scatter['sales'], y=df_scatter['profit'], hue=df_scatter['segment'], palette='viridis', size=df_scatter['sales'], sizes=(100, 1000), legend='auto', ax=ax)

    # Customize the plot
    plt.title("Sales vs Profit Across Different Customer Segments", size=20, pad=20)

    # Display the Scatter Plot using Streamlit
    st.pyplot(fig)
    plt.close(fig)
    st.write('''
    The scatter plot explores the relationship between sales and profit across different customer segments, with the size of 
    each point representing the volume of sales. This visualization highlights the variability in profitability across different 
    sales volumes and segments, suggesting that while some segments may generate higher sales, the associated profits vary significantly. 
    This insight is crucial for refining segmentation strategies, product offerings, and pricing models to enhance profitability.''')
    # create straight line
    st.markdown('---')

if __name__ == '__main__':
    run()
    
