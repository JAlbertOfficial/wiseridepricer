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
        ...
    """)
    # Hier können Sie ein Bild oder ein Flowchart hinzufügen
    # st.image("path/to/image.png", use_column_width=True)
    #st.button("Get Started")

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

    fig, ax = plt.subplots(figsize=(12, 6))
    # Create a barplot for the current variable
    sns.barplot(x='Level', y='Count', data=counts_df[counts_df["Variable"] == "make"].head(limit_selection), ax=ax)
    ax.set_title(f'Top {limit_selection} Best-Selling Car Makes')
    ax.set_xlabel('Car Make')
    ax.set_ylabel('Number of Sales')
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    # Anzeigen der Abbildung
    st.pyplot(fig)

#--------------------------------------------------------------
# Best Selling Models
#--------------------------------------------------------------

def plot_best_selling_models(df, limit_selection):
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

    fig, ax = plt.subplots(figsize=(12, 6))
    # Create a barplot for the current variable
    sns.barplot(x='Level', y='Count', data=counts_df[counts_df["Variable"] == "model"].head(limit_selection), ax=ax)
    ax.set_title(f'Top {limit_selection} Best-Selling Car Models')
    ax.set_xlabel('Car Model')
    ax.set_ylabel('Number of Sales')
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    # Anzeigen der Abbildung
    st.pyplot(fig)

#--------------------------------------------------------------
# Most valuable car makes
#--------------------------------------------------------------

def plot_most_valuable_makes(df, limit_selection):
    # Group by 'make' and calculate mean and standard deviation of prices
    top_price_makes = df.groupby('make')['price'].mean().sort_values(ascending=False).head(limit_selection)
    top_price_models_std = df.groupby('make')['price'].std().sort_values(ascending=False).head(limit_selection)

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_price_makes.index, y=top_price_makes.values, ax=ax, ci=None)
    ax.set_title(f'Top {limit_selection} Most Valuable Car Makes')
    ax.set_ylabel('Mean Price (€)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees

    # Add error bars to the bar chart
    ax.errorbar(x=top_price_makes.index, y=top_price_makes.values, yerr=top_price_models_std.values, fmt='none', color='black', capsize=5)

    # Anzeigen der Abbildung
    st.pyplot(fig)

#--------------------------------------------------------------
# Most valuable car models
#--------------------------------------------------------------

def plot_most_valuable_models(df, limit_selection):
    # Group by 'make' and calculate mean and standard deviation of prices
    top_price_models = df.groupby('model')['price'].mean().sort_values(ascending=False).head(limit_selection)
    top_price_models_std = df.groupby('model')['price'].std().sort_values(ascending=False).head(limit_selection)

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_price_models.index, y=top_price_models.values, ax=ax, ci=None)
    ax.set_title(f'Top {limit_selection} Most Valuable Car Models')
    ax.set_ylabel('Mean Price (€)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees

    # Add error bars to the bar chart
    ax.errorbar(x=top_price_models.index, y=top_price_models.values, yerr=top_price_models_std.values, fmt='none', color='black', capsize=5)

    # Anzeigen der Abbildung
    st.pyplot(fig)

#--------------------------------------------------------------
# Correlation heatmap
#--------------------------------------------------------------
    
def plot_correlation_heatmap(df):
    numerical_columns = ['price', 'mileage', 'hp', 'year']

    correlation_matrix = df[numerical_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=.5, ax=ax)

    ax.set_title(f"Correlation heatmap")

    st.pyplot(fig) 

#--------------------------------------------------------------
# Bivariate relationships
#--------------------------------------------------------------
        
def bivariate_relation_plot(df, x_col, y_col, num_bins=10):
    # Erstelle den Subplot mit 1 Zeile und 2 Spalten
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: regplot (ähnlich wie lmplot)
    sns.regplot(x=x_col, y=y_col, data=df,
                line_kws={'color': 'red', 'linestyle': '--'},
                scatter_kws={'edgecolor': 'black', 'facecolors': 'blue', 'alpha': 0.2}, ax=axs[0])

    axs[0].set_xlabel(x_col.title())
    axs[0].set_ylabel(y_col.title())
    axs[0].set_title(f'Regplot for {y_col.title()} vs {x_col.title()}')

    # Plot 2: binned barplot
    # Teile die x-Variable in gleich große Bins auf
    df['bins'] = pd.cut(df[x_col], bins=num_bins)

    # Berechne den Mittelwert und die Standardabweichung für jeden Bin
    bin_stats = df.groupby('bins')[y_col].agg(['mean', 'std']).reset_index()

    # Extrahiere die Mitte jedes Intervalls als x-Werte
    bin_mids = bin_stats['bins'].apply(lambda x: x.mid).values

    # Benutzerdefinierte Bin-Breiten
    bin_widths = [interval.length for interval in bin_stats['bins']]

    # Erzeuge den Barplot mit Abstand zwischen den Balken und schwarzer Umrandung
    axs[1].bar(bin_mids, bin_stats['mean'], width=bin_widths, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1, label='Mittelwert')

    # Füge die Fehlerbalken separat hinzu
    axs[1].errorbar(bin_mids, bin_stats['mean'], yerr=bin_stats['std'], fmt='none', capsize=5, color='black', label='Std')

    axs[1].set_xlabel(x_col.title())
    axs[1].set_ylabel(y_col.title())
    axs[1].set_title(f'Binned Barplot for Car Price VS {x_col.title()}')    
    axs[1].tick_params(axis='x', rotation=90)

    # Zeige den Plot an
    st.pyplot(fig)  # Verwenden Sie st.pyplot statt plt.show()

#--------------------------------------------------------------
# Number of Sales per variable
#--------------------------------------------------------------

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def number_sales_per_variable(df, v):
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

    # Erstelle Subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Mean +- Standardabweichung der Variable "price"
    # Hier nehmen wir an, dass "price" eine numerische Variable ist
    price_stats = df.groupby(v)['price'].agg(['mean', 'std']).reset_index()

    # Manuell erstellte x-Achse
    x_values = list(range(len(price_stats)))

    sns.barplot(x=x_values, y=price_stats['mean'], ci='sd', ax=axs[0])
    axs[0].set_title(f'Mean Price for each {v.title()}')
    axs[0].set_xlabel(f'{v.title()}')
    axs[0].set_ylabel('Price')
    axs[0].set_xticks(x_values)
    axs[0].set_xticklabels(price_stats[v])

    # Füge die Fehlerbalken separat hinzu
    axs[0].errorbar(x=x_values, y=price_stats['mean'], yerr=price_stats['std'], fmt='none', color='black', capsize=5)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90, ha='right')

    # Plot 2: Number of Sales per Variable
    sns.barplot(x='Level', y='Count', data=counts_df[counts_df["Variable"] == v], ax=axs[1])
    axs[1].set_title(f'Number of Sales per {v.title()} Type')
    axs[1].set_xlabel(f'{v.title()}')
    axs[1].set_ylabel('Number of Sales')
    # Rotate x-axis labels for better readability
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, ha='right')

    # Anzeigen der Abbildung
    st.pyplot(fig)

#==============================================================
# EDA page 
#==============================================================
    
def eda_visualization():
    st.title("Explorative Data Analysis")
    st.write("""
        Welcome to the heart of WiseRidePricer's data exploration journey!
        Unleash the power of data visualization to uncover hidden insights and trends in the German automotive market.

        Let's embark on this insightful journey together, where every data point tells a story! Choose your visualization:
    """)

    # Select Box für "What do you want to plot?"
    chosen_option = st.selectbox("What do you want to plot?", ["Best-Selling Makes", "Best-Selling Models", "Most valuable car makes",
                                                             "Most valuable car models", "Relationship between car prices and car features"])

    if chosen_option in ["Best-Selling Makes", "Best-Selling Models", "Most valuable car makes", "Most valuable car models"]:
        # Schieberegler für "Limit selection to (Max 50)"
        limit_selection = st.slider("Limit selection to (Max 50)", min_value=1, max_value=50, value=10)

    if chosen_option == "Relationship between car prices and car features":
        # Multiselect für "Chose makes to be included"
        chosen_makes = st.multiselect("Chose makes to be included", ["All"] + list(df_raw["make"].unique()), default=["All"])

        # Multiselect für "Chose models to be included"
        chosen_models = st.multiselect("Chose models to be included", ["All"] + list(df_raw["model"].unique()), default=["All"])

        # Multiselect für "Chose fuel types to be included"
        chosen_fuels = st.multiselect("Chose fuel types to be included", ["All"] + list(df_raw["fuel"].unique()), default=["All"])

        # Multiselect für "Chose gear types to be included"
        chosen_gears = st.multiselect("Chose gear types to be included", ["All"] + list(df_raw["gear"].unique()), default=["All"])

        # Multiselect für "Chose offer types to be included"
        chosen_offers = st.multiselect("Chose offer types to be included", ["All"] + list(df_raw["offerType"].unique()), default=["All"])

        # Slider für "Select mileage range"
        mileage_range = st.slider("Select mileage range", min_value=df_raw["mileage"].min(), max_value=df_raw["mileage"].max(), value=(df_raw["mileage"].min(), df_raw["mileage"].max()))

        # Slider für "Select price range"
        price_range = st.slider("Select price range", min_value=df_raw["price"].min(), max_value=df_raw["price"].max(), value=(df_raw["price"].min(), df_raw["price"].max()))

        # Slider für "Select hp range"
        hp_range = st.slider("Select hp range", min_value=df_raw["hp"].min(), max_value=df_raw["hp"].max(), value=(df_raw["hp"].min(), df_raw["hp"].max()))

        # Slider für "Select year range"
        year_range = st.slider("Select year range", min_value=df_raw["year"].min(), max_value=df_raw["year"].max(), value=(df_raw["year"].min(), df_raw["year"].max()))

        # Multiselect für "Chose plot type"
        chosen_plot = st.multiselect("Chose plot type", ["Correlation heatmap", "Price VS Milage", "Price VS Fuel Type", "Price VS Gear", "Price VS Offer Type", "Price VS HP", "Price VS Year"], default=["Correlation heatmap"])
    
        # Erzeugen Sie ein Subset des DataFrame basierend auf den ausgewählten Filtern
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

    if st.button("Plot it"):
        if chosen_option == "Best-Selling Makes":
            plot_best_selling_makes(df_raw, limit_selection)
        elif chosen_option == "Best-Selling Models":
            plot_best_selling_models(df_raw, limit_selection)
        elif chosen_option == "Most valuable car makes":
            plot_most_valuable_makes(df_raw, limit_selection)
        elif chosen_option == "Most valuable car models":
            plot_most_valuable_models(df_raw, limit_selection)
        elif chosen_option == "Relationship between car prices and car features":
            if "Correlation heatmap" in chosen_plot:
                if filtered_df.empty:
                    st.warning("No data available for the selected filters. Please adjust your selection.")
                else:
                    plot_correlation_heatmap(filtered_df)
            if "Price VS Milage" in chosen_plot:
                bivariate_relation_plot(filtered_df, 'mileage', 'price', num_bins=10)
            if "Price VS HP" in chosen_plot:
                bivariate_relation_plot(filtered_df, 'hp', 'price', num_bins=10)
            if "Price VS Year" in chosen_plot:
                bivariate_relation_plot(filtered_df, 'year', 'price', num_bins=10)
            if "Price VS Fuel Type" in chosen_plot:
                number_sales_per_variable(filtered_df, 'fuel')
            if "Price VS Gear" in chosen_plot:
                number_sales_per_variable(filtered_df, 'gear')
            if "Price VS Offer Type" in chosen_plot:
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
