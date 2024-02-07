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
    from the bar chart above the region with the highest average sales is South Region with 241 values followed by East Region with 238 values then West Region with 226 values, and the lowest ones is Central Region with 215 values.
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
    - From chart above, it can be seen that the average peak sales in 2014-2015 reached around 2800 and the average peak of profit was around 400 and the losses around 400.
    - In 2015-2016 the peak of average sales reached more than 4000 and in the same month the average profit peaked at 600 and the loss peaked at 1800. 
    - In 2016-2017 the average sales reaches 2600 but the average profit reached around 1200 and the loss is around 500 in the end of the year. 
    - In the last year, average sales fluctuated between 2000-2500, but profit peaked at 800 then the average losses were only around 300.
    - Most of the time when sales increase the profit also increase. 
    - Overall, the average sales and profit loss reached its peak in 2015. This likely happen with increase in overhead expenses includes costs that occur due to operating a business, like rent and administrative costs.
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
    - From the bar chart that visualize segment against region, we can see that the highest segment is consumer, followed by corporate and home office. 
    - The region distributions against the three segments have almost the same pattern, West Region always occupied the highest records, 
    the second place is East Region followed by Central Region and the lowest record is South Region.''')
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
    - Scatter plot above shows the bigger circle is the higher sales of Segment that distinct by the different color. The light green show home office segment, the dark green show the corporate segment, and the blue color show the consumer segment.
    - The distribution of the segment commonly range around 0-5000 sales product and profited to 2000 while the loss reach around -4000.
    - For sales in range 5000-15.000 dominated by the consumer segment where the profit bigger than the loss
    - For sales in range 15.000-20.000 only generated by corporate segment, and also produced the highest profit of the sales exceeded 8000
    - There are extreme case where values of sales exceeded 20.000 but ends up losing profit from home office segment''')


    # create straight line
    st.markdown('---')
    st.subheader('New dataset after clusterings')
    # show dataframe
    df2 = pd.read_csv('clustered.csv')
    st.dataframe(df2)
    # Menampilkan jumlah baris dan kolom
    st.write(f"Rows: {df2.shape[0]}")
    st.write(f"Columns: {df2.shape[1]}")
    # Menampilkan value counts cluster 
    # Display value counts for 'price_cluster'
    value_counts = df2['price_cluster'].value_counts()
    st.write("Cluster Value Counts:")
    st.write(value_counts)

    st.subheader('Sales vs Profit with Price Clustering')

    # Display Scatter Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.scatterplot(x=df2['profit'], y=df2['sales'], hue=df2['price_cluster'], palette='viridis', ax=ax)
    plt.title("Clustering Results")
    plt.xlabel("Profit")
    plt.ylabel("Sales")

    # Display the Scatter Plot using Streamlit
    st.pyplot(fig)
    plt.close(fig)
    st.write('''
    - Scatter plot above shows ''')

    st.markdown('---')
    # Create a Streamlit app
    st.subheader('Heatmap of Numerical Columns Correlation')
    # Melihat distribusi data pada numerical column
    numerical_columns = df2.select_dtypes(include=np.number).columns
    # Taking a subset of data
    df_num = df2[numerical_columns].copy()

    # Calculate correlation
    corr = df_num.corr()

    # Visualize the Heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr, square=True, annot=True, cmap='viridis', ax=ax)

    # Display the heatmap using Streamlit
    st.pyplot(fig)
    plt.close(fig)
    st.write('''
    - Heatmap above shows ''')

    st.markdown('---')
    # Create a Streamlit app
    st.subheader('Boxplots of Important Features by Price Cluster with Correlation > 0.05')

    # Define important features with correlation > 0.05
    important_feature = list(corr[corr['price_cluster'] > 0.05].index)

    # Display boxplots for selected features
    fig, ax = plt.subplots(figsize=(25, 25))
    for i, col in enumerate(important_feature[:-1]):
        ax = plt.subplot(4, 2, i + 1)
        sns.boxplot(x=df2['price_cluster'], y=df2[col])
        plt.xlabel("Cluster", fontsize=15)
        plt.ylabel(col, fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

    plt.tight_layout()

    # Display the boxplots using Streamlit
    st.pyplot(fig)
    plt.close(fig)
    st.write('''
    - Boxplot above shows ''')
if __name__ == '__main__':
    run()
    
