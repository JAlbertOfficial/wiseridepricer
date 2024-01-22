################################################################
# Packages
###############################################################

import streamlit as st
import pandas as pd
import seaborn as sns
#import requests
#from io import StringIO
import warnings
warnings.filterwarnings("ignore")

###############################################################
# Data import
###############################################################

path = "./data/raw/autoscout24.csv"

df_full = pd.read_csv(path)
df_top5 = df_full['make'].value_counts().head(5).index
df_top10 = df_full['make'].value_counts().head(10).index


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

        Choose from the following exciting possibilities:
        - Gain a holistic view of the entire dataset, revealing intriguing patterns and trends (Select Box 1).
        - Dive into specific car brands to discover unique insights (Select Box 2).
        - Explore detailed information about specific car models (Select Box 3).

        Let's embark on this insightful journey together, where every data point tells a story! Chose your visualization:
    """)
    # Horizontal angeordnete Select Boxes
    col1, col2, col3 = st.columns(3)

    with col1:
        select_box_1 = st.selectbox("1.) Analyze complete car market", ["Option 1", "Option 2", "Option 3"])

    with col2:
        select_box_2 = st.selectbox("2.) Analyze specific car make", ["Option A", "Option B", "Option C"])

    with col3:
        select_box_3 = st.selectbox("3.) Analyze specific car model", ["Model 1", "Model 2", "Model 3"])

    

# Modelling Page
def modelling():
    st.title("Modelling")
    # Hier fügen Sie den Code für Seite 3 ein

# Prediction Page
def prediction():
    st.title("Prediction")
    # Hier fügen Sie den Code für Seite 4 ein

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