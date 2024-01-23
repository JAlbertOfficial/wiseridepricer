################################################################
# Packages
###############################################################

import streamlit as st
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
#df_top5 = df_full['make'].value_counts().head(5).index
#df_top10 = df_full['make'].value_counts().head(10).index


#DATA_URL = ('https://github.com/JAlbertOfficial/wiseridepricer/blob/main/data/raw/autoscout24.csv')

#data = pd.read_csv(DATA_URL)


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
    st.button("Get Started")

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
# Most expensive car makes
#--------------------------------------------------------------

def plot_most_expensive_makes(df, limit_selection):
    # Group by 'make' and calculate mean and standard deviation of prices
    top_price_makes = df.groupby('make')['price'].mean().sort_values(ascending=False).head(limit_selection)
    top_price_std = df.groupby('make')['price'].std().sort_values(ascending=False).head(limit_selection)

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_price_makes.index, y=top_price_makes.values, ax=ax, ci=None)
    ax.set_title(f'Top {limit_selection} Most Valuable Car Makes')
    ax.set_ylabel('Mean Price (€)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees

    # Add error bars to the bar chart
    ax.errorbar(x=top_price_makes.index, y=top_price_makes.values, yerr=top_price_std.values, fmt='none', color='black', capsize=5)

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
        # Select Box für "Chose makes to be included"
        chosen_make = st.selectbox("Chose makes to be included", ["All makes"] + list(df_raw["make"].unique()))

        if chosen_make == "All makes":
            # Select Box für "Chose plot" (Option: "All makes")
            chosen_plot_all_makes = st.selectbox("Chose plot", ["Correlation between all variables", "Price VS Milage",
                                                                "Price VS Fuel Type", "Price VS Gear", "Price VS Offer Type",
                                                                "Price VS HP", "Price VS Year"])
        else:
            # Select Box für "Chose models to be included" (Option: specific make)
            chosen_model = st.selectbox("Chose models to be included", ["All models"] + list(df_raw[df_raw["make"] == chosen_make]["model"].unique()))

            if chosen_model == "All models":
                # Select Box für "Chose plot" (Option: "All models")
                chosen_plot_all_models = st.selectbox("Chose plot", ["Correlation between all variables", "Price VS Milage",
                                                                    "Price VS Fuel Type", "Price VS Gear", "Price VS Offer Type",
                                                                    "Price VS HP", "Price VS Year"])
            else:
                # Select Box für "Chose plot" (Option: specific model)
                chosen_plot_specific_model = st.selectbox("Chose plot", ["Price VS Milage", "Price VS Fuel Type", "Price VS Gear",
                                                                         "Price VS Offer Type", "Price VS HP", "Price VS Year"])

    if st.button("Plot it"):
        if chosen_option == "Best-Selling Makes":
            plot_best_selling_makes(df_raw, limit_selection)
        elif chosen_option == "Best-Selling Models":
            plot_best_selling_models(df_raw, limit_selection)
        elif chosen_option == "Most valuable car makes":
            plot_most_expensive_makes(df_raw, limit_selection)

# Annahme: df_raw und limit_selection wurden zuvor definiert
eda_visualization() 

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

# Hauptnavigation in Form von klickbaren Kacheln
selected_page = st.sidebar.button("Home")
if selected_page:
    home()

selected_page = st.sidebar.button("EDA and Visualization")
if selected_page:
    eda_visualization()

selected_page = st.sidebar.button("Modelling")
if selected_page:
    modelling()

selected_page = st.sidebar.button("Prediction")
if selected_page:
    prediction()

# Linke Sidebar für spezifische Optionen beim Plotten und Modellieren
if "EDA and Visualization" in st.session_state:
    st.sidebar.title("Visualization Options")
    # Hier können Sie spezifische Optionen für die Visualisierung hinzufügen, z.B. Datenauswahl, Diagrammtypen usw.
elif "Modelling" in st.session_state:
    st.sidebar.title("Modeling Options")
    # Hier können Sie spezifische Optionen für das Modellieren hinzufügen, z.B. Datensubset-Auswahl, Preprocessing-Optionen, Modelltypen usw." 