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

def eda_visualization():
    st.title("Explorative Data Analysis")
    st.write("""
        Welcome to the heart of WiseRidePricer's data exploration journey!
        Unleash the power of data visualization to uncover hidden insights and trends in the German automotive market.

        Let's embark on this insightful journey together, where every data point tells a story! Choose your visualization:
    """)

    # Select Box für "What do you want to plot?"
    chosen_option = st.selectbox("What do you want to plot?", ["Best-Selling Makes", "Best-Selling Models", "Most expensive car makes",
                                                             "Most expensive car models", "Relationship between car prices and car features"])

    if chosen_option in ["Best-Selling Makes", "Best-Selling Models", "Most expensive car makes", "Most expensive car models"]:
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