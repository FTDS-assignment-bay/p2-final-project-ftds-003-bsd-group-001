import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from matplotlib.gridspec import GridSpec

def average_sales_by_region(df):
    """
    Generate a bar plot for average sales by region.
    """
    df_bar = df[['region', 'sales']]
    df_bar = df_bar.groupby('region').mean().sort_values(by='sales', ascending=False)
    fig, ax = plt.subplots(figsize=[10, 6])
    sns.barplot(x=df_bar.index, y='sales', data=df_bar, palette='viridis', ax=ax)
    ax.set_title('Average Sales Across Different Regions')
    ax.set_xlabel('Region')
    ax.set_ylabel('Average Sales')
    for index, value in enumerate(df_bar['sales']):
        ax.text(index, value, f"{value:.2f}", ha='center', va='bottom')
    return fig

def average_sales_and_profit_over_time(df):
    """
    Generate a line plot for average sales and profit over time.
    """
    df_line = df[['order_date', 'sales', 'profit']].sort_values('order_date')
    df_line['order_date'] = pd.to_datetime(df_line['order_date'])
    df_line = df_line.groupby(df_line['order_date'].dt.to_period("M")).mean()
    df_line.index = df_line.index.to_timestamp()
    fig, ax = plt.subplots(figsize=[10, 6])
    ax.plot(df_line.index, 'sales', data=df_line, color='green', label='Avg Sales')
    ax.plot(df_line.index, 'profit', data=df_line, color='red', label='Avg Profit')
    ax.legend()
    ax.set_title('Average Sales and Profit Over Time (Monthly)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    return fig

def segment_vs_region_distribution(df):
    """
    Generate a count plot for segments across different regions.
    """
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x='segment', data=df, hue='region', palette='viridis')
    plt.title('Segment vs. Region Distribution')
    plt.xlabel('Segment')
    plt.ylabel('Count')
    plt.legend(title='Region')
    return fig

def sales_vs_profit_across_segments(df):
    """
    Generate a scatter plot comparing sales and profit across different customer segments.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='sales', y='profit', hue='segment', data=df, palette='viridis', size='sales', sizes=(20, 200), ax=ax)
    ax.set_title('Sales vs. Profit Across Different Customer Segments')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Profit')
    return fig

def category_composition_for_profit_and_sales(df):
    """
    Generate pie charts for the composition of category for profit and sales.
    """
    df_pie = df.groupby('category').agg({'sales': 'sum', 'profit': 'sum'}).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].pie(df_pie['sales'], labels=df_pie['category'], autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    axs[0].set_title('Sales Composition by Category')
    axs[1].pie(df_pie['profit'], labels=df_pie['category'], autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    axs[1].set_title('Profit Composition by Category')
    return fig

# Additional EDA functions can be added following the same pattern
