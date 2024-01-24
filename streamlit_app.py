################################################################
# Packages
###############################################################

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
#import requests
#from io import StringIO
import warnings
warnings.filterwarnings("ignore")

###############################################################
# Data import
###############################################################

path = "./data/raw/autoscout24.csv"

df_raw = pd.read_csv(path)

###############################################################
# Home Page
###############################################################

def home():
    st.title("WiseRidePricer - Explore the Automotive Market")
    st.write("""
        Explore the full potential of the cutting-edge Streamlit app WiseRidePricer designed to revolutionize your car buying and selling experience! 
        Dive into a world of comprehensive data analysis and captivating visualizations derived from a rich dataset sourced from Autoscout24. 
        Uncover total sales figures, discern dominant brands, unravel feature correlations, and delve into temporal shifts within the automotive market. 
        Embark on a data-driven journey with this app, where you have a plethora of regression methods at your fingertips. Predicting sales prices becomes a breeze 
        as you select and analyze specific car characteristics. WiseRidePricer goes beyond predictions - it empowers you with model evaluation metrics, aiding in 
        performance assessment and assisting you in choosing the optimal model for your needs. Experience the ultimate in user-friendly interfaces with a 
        streamlined dashboard. An intuitive platform that not only effectively communicates intricate results but also forecasts prices based on your chosen models 
        is available. Elevate your car-related decisions with this app - your go-to companion for informed choices in the dynamic world of automotive transactions.
    """)  

###############################################################
# EDA
###############################################################
 
#==============================================================
# Plotting functions
#==============================================================
    
#--------------------------------------------------------------
# Best Selling Makes
#--------------------------------------------------------------

def plot_best_selling_makes(df, limit_selection):
    """
    Plots a bar chart of the top best-selling car makes based on the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing automotive market data.
    - limit_selection (int): Maximum number of makes to display on the chart.

    Returns:
    - None
    """

    # List to store individual DataFrames for each non-numeric column
    dfs = []

    # Iterate over non-numeric columns in the DataFrame
    for column in df.columns:
        # Count the occurrences of each level for the current column
        counts = df[column].value_counts().reset_index()
        # Rename columns for consistency
        counts.columns = ['Level', 'Count']
        # Add a new column for the variable name
        counts['Variable'] = column
        # Append the counts DataFrame to the list
        dfs.append(counts)

    # Concatenate all DataFrames in the list
    counts_df = pd.concat(dfs, ignore_index=True)

    # Create a bar chart for the 'make' variable
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Level', y='Count', data=counts_df[counts_df["Variable"] == "make"].head(limit_selection), ax=ax)
    ax.set_title(f'Top {limit_selection} Best-Selling Car Makes')
    ax.set_xlabel('Car Make')
    ax.set_ylabel('Number of Sales')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right') # Rotate x-axis labels for better readability

    # Display the plot
    st.pyplot(fig)

#--------------------------------------------------------------
# Best Selling Models
#--------------------------------------------------------------

def plot_best_selling_models(df, limit_selection):
    """
    Plots a bar chart of the top best-selling car models based on the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing automotive market data.
    - limit_selection (int): Maximum number of models to display on the chart.

    Returns:
    - None
    """

    # List to store individual DataFrames for each non-numeric column
    dfs = []

    # Iterate over non-numeric columns in the DataFrame
    for column in df.columns:
        # Count the occurrences of each level for the current column
        counts = df[column].value_counts().reset_index()
        # Rename columns for consistency
        counts.columns = ['Level', 'Count']
        # Add a new column for the variable name
        counts['Variable'] = column
        # Append the counts DataFrame to the list
        dfs.append(counts)

    # Concatenate all DataFrames in the list
    counts_df = pd.concat(dfs, ignore_index=True)

    # Create a bar chart for the 'model' variable
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Level', y='Count', data=counts_df[counts_df["Variable"] == "model"].head(limit_selection), ax=ax)
    ax.set_title(f'Top {limit_selection} Best-Selling Car Models')
    ax.set_xlabel('Car Model')
    ax.set_ylabel('Number of Sales')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right') # Rotate x-axis labels for better readability

    # Display the plot
    st.pyplot(fig)

#--------------------------------------------------------------
# Most valuable car makes
#--------------------------------------------------------------

def plot_most_valuable_makes(df, limit_selection):
    """
    Plots a bar chart of the top most valuable car makes based on the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing automotive market data.
    - limit_selection (int): Maximum number of makes to display on the chart.

    Returns:
    - None
    """

    # Group by 'make' and calculate the mean and standard deviation of prices
    top_price_makes = df.groupby('make')['price'].mean().sort_values(ascending=False).head(limit_selection)
    top_price_models_std = df.groupby('make')['price'].std().sort_values(ascending=False).head(limit_selection)

    # Create a bar chart for the most valuable car makes
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_price_makes.index, y=top_price_makes.values, ax=ax, ci=None)
    ax.set_title(f'Top {limit_selection} Most Valuable Car Makes')
    ax.set_ylabel('Mean Price (€)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees

    # Add error bars to the bar chart
    ax.errorbar(x=top_price_makes.index, y=top_price_makes.values, yerr=top_price_models_std.values, fmt='none', color='black', capsize=5)

    # Display the plot
    st.pyplot(fig)

#--------------------------------------------------------------
# Most valuable car models
#--------------------------------------------------------------

def plot_most_valuable_models(df, limit_selection):
    """
    Plots a bar chart of the top most valuable car models based on the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing automotive market data.
    - limit_selection (int): Maximum number of models to display on the chart.

    Returns:
    - None
    """

    # Group by 'model' and calculate the mean and standard deviation of prices
    top_price_models = df.groupby('model')['price'].mean().sort_values(ascending=False).head(limit_selection)
    top_price_models_std = df.groupby('model')['price'].std().sort_values(ascending=False).head(limit_selection)

    # Create a bar chart for the most valuable car models
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_price_models.index, y=top_price_models.values, ax=ax, ci=None)
    ax.set_title(f'Top {limit_selection} Most Valuable Car Models')
    ax.set_ylabel('Mean Price (€)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees

    # Add error bars to the bar chart
    ax.errorbar(x=top_price_models.index, y=top_price_models.values, yerr=top_price_models_std.values, fmt='none', color='black', capsize=5)

    # Display the plot
    st.pyplot(fig)

#--------------------------------------------------------------
# Correlation heatmap
#--------------------------------------------------------------
    
def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for selected numerical columns in the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing automotive market data.

    Returns:
    - None
    """

    # Select numerical columns for correlation analysis
    numerical_columns = ['price', 'mileage', 'hp', 'year']

    # Calculate the correlation matrix
    correlation_matrix = df[numerical_columns].corr()

    # Create a heatmap for the correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=.5, ax=ax)

    # Set the title for the heatmap
    ax.set_title(f"Correlation Heatmap")

    # Display the plot
    st.pyplot(fig)

#--------------------------------------------------------------
# Bivariate relationships
#--------------------------------------------------------------
        
def bivariate_relation_plot(df, x_col, y_col, num_bins=10):
    """
    Plots a bivariate relation using a regression plot and a binned barplot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing automotive market data.
    - x_col (str): Name of the x-axis variable.
    - y_col (str): Name of the y-axis variable.
    - num_bins (int, optional): Number of bins for the binned barplot. Defaults to 10.

    Returns:
    - None
    """

    # Create a subplot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Regression plot 
    sns.regplot(x=x_col, y=y_col, data=df,
                line_kws={'color': 'red', 'linestyle': '--'},
                scatter_kws={'edgecolor': 'black', 'facecolors': 'blue', 'alpha': 0.2}, ax=axs[0])

    axs[0].set_xlabel(x_col.title())
    axs[0].set_ylabel(y_col.title())
    axs[0].set_title(f'Regression Plot for {y_col.title()} vs {x_col.title()}')

    # Plot 2: Binned barplot
    # Divide the x-variable into equally sized bins
    df['bins'] = pd.cut(df[x_col], bins=num_bins)

    # Calculate the mean and standard deviation for each bin
    bin_stats = df.groupby('bins')[y_col].agg(['mean', 'std']).reset_index()

    # Extract the midpoint of each interval as x-values
    bin_mids = bin_stats['bins'].apply(lambda x: x.mid).values

    # Custom bin widths
    bin_widths = [interval.length for interval in bin_stats['bins']]

    # Create the barplot with spacing between bars and black edges
    axs[1].bar(bin_mids, bin_stats['mean'], width=bin_widths, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1, label='Mean')

    # Add error bars separately
    axs[1].errorbar(bin_mids, bin_stats['mean'], yerr=bin_stats['std'], fmt='none', capsize=5, color='black', label='Std')

    axs[1].set_xlabel(x_col.title())
    axs[1].set_ylabel(y_col.title())
    axs[1].set_title(f'Binned Barplot for Car Price VS {x_col.title()}')
    axs[1].tick_params(axis='x', rotation=90)

    # Display the plot
    st.pyplot(fig)
 
#--------------------------------------------------------------
# Number of Sales per variable
#--------------------------------------------------------------

def number_sales_per_variable(df, v):
    """
    Plots the mean price and the number of sales for each level of a specified variable.

    Parameters:
    - df (pd.DataFrame): DataFrame containing automotive market data.
    - v (str): Name of the variable for analysis.

    Returns:
    - None
    """

    dfs = []

    # Iterate over non-numeric columns
    for column in df.columns:
        # Count the occurrences of each level
        counts = df[column].value_counts().reset_index()
        # Rename columns for consistency
        counts.columns = ['Level', 'Count']
        # Add a new column for the variable name
        counts['Variable'] = column
        # Append the counts DataFrame to the list
        dfs.append(counts)

    # Concatenate all DataFrames in the list
    counts_df = pd.concat(dfs, ignore_index=True)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Mean +- Standard Deviation of the variable "price"
    price_stats = df.groupby(v)['price'].agg(['mean', 'std']).reset_index()

    # Manually create x-axis
    x_values = list(range(len(price_stats)))

    sns.barplot(x=x_values, y=price_stats['mean'], ci='sd', ax=axs[0])
    axs[0].set_title(f'Mean Price for each {v.title()}')
    axs[0].set_xlabel(f'{v.title()}')
    axs[0].set_ylabel('Price')
    axs[0].set_xticks(x_values)
    axs[0].set_xticklabels(price_stats[v])

    # Add error bars separately
    axs[0].errorbar(x=x_values, y=price_stats['mean'], yerr=price_stats['std'], fmt='none', color='black', capsize=5)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90, ha='right')

    # Plot 2: Number of Sales per Variable
    sns.barplot(x='Level', y='Count', data=counts_df[counts_df["Variable"] == v], ax=axs[1])
    axs[1].set_title(f'Number of Sales per {v.title()} Type')
    axs[1].set_xlabel(f'{v.title()}')
    axs[1].set_ylabel('Number of Sales')
    # Rotate x-axis labels for better readability
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, ha='right')

    # Display the plot
    st.pyplot(fig)

#==============================================================
# EDA page 
#==============================================================
    
def eda_visualization():
    """
    Function for exploratory data analysis (EDA) visualization.

    Parameters:
    - None

    Returns:
    - None
    """
    st.title("Explorative Data Analysis")
    st.write("""
        Welcome to the heart of WiseRidePricer's data exploration journey!
        Unleash the power of data visualization to uncover hidden insights and trends in the German automotive market.

        Let's embark on this insightful journey together, where every data point tells a story! Choose your visualization:
    """)

    # Select Box for "What do you want to plot?"
    chosen_option = st.selectbox("What do you want to plot?", ["Best-Selling Makes", "Best-Selling Models", "Most valuable car makes",
                                                             "Most valuable car models", "Relationship between car prices and car features"])

    # Explanation for selecting specific visualizations
    st.write("""
        Choose the type of visualization you want to explore. Options include best-selling makes, best-selling models, most valuable car makes, 
        most valuable car models, or explore the relationship between car prices and car features.
    """)

    if chosen_option in ["Best-Selling Makes", "Best-Selling Models", "Most valuable car makes", "Most valuable car models"]:
        # Slider for "Limit selection to (Max 50)"
        limit_selection = st.slider("Limit selection to (Max 50)", min_value=1, max_value=50, value=10)

        # Explanation for limiting the selection
        st.write("""
            Adjust the slider to limit the selection to a maximum number of entries for better visualization.
        """)

    if chosen_option == "Relationship between car prices and car features":
        # Multiselect for "Chose makes to be included"
        chosen_makes = st.multiselect("Chose makes to be included", ["All"] + list(df_raw["make"].unique()), default=["All"])

        # Multiselect for "Chose models to be included"
        chosen_models = st.multiselect("Chose models to be included", ["All"] + list(df_raw["model"].unique()), default=["All"])

        # Multiselect for "Chose fuel types to be included"
        chosen_fuels = st.multiselect("Chose fuel types to be included", ["All"] + list(df_raw["fuel"].unique()), default=["All"])

        # Multiselect for "Chose gear types to be included"
        chosen_gears = st.multiselect("Chose gear types to be included", ["All"] + list(df_raw["gear"].unique()), default=["All"])

        # Multiselect for "Chose offer types to be included"
        chosen_offers = st.multiselect("Chose offer types to be included", ["All"] + list(df_raw["offerType"].unique()), default=["All"])

        # Slider for "Select mileage range"
        mileage_range = st.slider("Select mileage range", min_value=df_raw["mileage"].min(), max_value=df_raw["mileage"].max(), value=(df_raw["mileage"].min(), df_raw["mileage"].max()))

        # Slider for "Select price range"
        price_range = st.slider("Select price range", min_value=df_raw["price"].min(), max_value=df_raw["price"].max(), value=(df_raw["price"].min(), df_raw["price"].max()))

        # Slider for "Select hp range"
        hp_range = st.slider("Select hp range", min_value=df_raw["hp"].min(), max_value=df_raw["hp"].max(), value=(df_raw["hp"].min(), df_raw["hp"].max()))

        # Slider for "Select year range"
        year_range = st.slider("Select year range", min_value=df_raw["year"].min(), max_value=df_raw["year"].max(), value=(df_raw["year"].min(), df_raw["year"].max()))

        # Selectbox for "Chose plot type" (simple selectbox instead of multiselect)
        chosen_plot = st.selectbox("Chose plot type", ["Correlation heatmap", "Price VS Milage", "Price VS Fuel Type", "Price VS Gear", "Price VS Offer Type", "Price VS HP", "Price VS Year"], index=0)

        # Explanation for filtering options
        st.write("""
            Customize your analysis by selecting specific makes, models, fuel types, gear types, and offer types. Adjust sliders to define the range for mileage, 
            price, horsepower (HP), and manufacturing year. Choose the desired plot type to visualize the relationship between car prices and selected features.
            Click the button to generate and display the selected plot based on the chosen options and filters.
                 """)
    
        # Generate a subset of the DataFrame based on the selected filters
        filtered_df = df_raw[
            (df_raw["make"].isin(chosen_makes) if "All" not in chosen_makes else True) &
            (df_raw["model"].isin(chosen_models) if "All" not in chosen_models else True) &
            (df_raw["fuel"].isin(chosen_fuels) if "All" not in chosen_fuels else True) &
            (df_raw["gear"].isin(chosen_gears) if "All" not in chosen_gears else True) &
            (df_raw["offerType"].isin(chosen_offers) if "All" not in chosen_offers else True) &
            (df_raw["mileage"].between(mileage_range[0], mileage_range[1])) &
            (df_raw["price"].between(price_range[0], price_range[1])) &
            (df_raw["hp"].between(hp_range[0], hp_range[1])) &
            (df_raw["year"].between(year_range[0], year_range[1]))
        ]

    # Button to trigger the plot
    if st.button("Plot it"):         
        # Check if the filtered DataFrame is empty
        if filtered_df.empty:
            # Display a warning message if no data is available for the selected filters
            st.warning("No data available for the selected filters. Please adjust your selection.")
        else:
            if chosen_option == "Best-Selling Makes":
                # Call function to plot the best-selling car makes
                plot_best_selling_makes(df_raw, limit_selection)
            elif chosen_option == "Best-Selling Models":
                # Call function to plot the best-selling car models
                plot_best_selling_models(df_raw, limit_selection)
            elif chosen_option == "Most valuable car makes":
                # Call function to plot the most valuable car makes
                plot_most_valuable_makes(df_raw, limit_selection)
            elif chosen_option == "Most valuable car models":
                # Call function to plot the most valuable car models
                plot_most_valuable_models(df_raw, limit_selection)
            elif chosen_option == "Relationship between car prices and car features":
                # Check the selected plot type
                if chosen_plot == "Correlation heatmap":
                    # Call function to plot the correlation heatmap
                    plot_correlation_heatmap(filtered_df)
                elif chosen_plot == "Price VS Milage":
                    # Call function to plot the bivariate relation between price and mileage
                    bivariate_relation_plot(filtered_df, 'mileage', 'price', num_bins=10)
                elif chosen_plot == "Price VS HP":
                    # Call function to plot the bivariate relation between price and horsepower
                    bivariate_relation_plot(filtered_df, 'hp', 'price', num_bins=10)
                elif chosen_plot == "Price VS Year":
                    # Call function to plot the bivariate relation between price and year
                    bivariate_relation_plot(filtered_df, 'year', 'price', num_bins=10)
                elif chosen_plot == "Price VS Fuel Type":
                    # Call function to plot the number of sales per fuel type
                    number_sales_per_variable(filtered_df, 'fuel')
                elif chosen_plot == "Price VS Gear":
                    # Call function to plot the number of sales per gear type
                    number_sales_per_variable(filtered_df, 'gear')
                elif chosen_plot == "Price VS Offer Type":
                    # Call function to plot the number of sales per offer type
                    number_sales_per_variable(filtered_df, 'offerType')
            
###############################################################
# Modelling
###############################################################    

# Modelling Page
def modelling():
    st.title("Modelling")
    # Hier fügen Sie den Code für Seite 3 ein

###############################################################
# Prediction
###############################################################    

# Prediction Page
def prediction():
    st.title("Prediction")
    # Hier fügen Sie den Code für Seite 4 ein

###############################################################
# Navigation
###############################################################

selected = option_menu(
    menu_title=None,
    options=["Home", "Visualize", "Predict"],
    orientation="horizontal"
)

if selected == "Home":
    home()
if selected == "Visualize":
    eda_visualization()
if selected == "Predict":
    prediction()
