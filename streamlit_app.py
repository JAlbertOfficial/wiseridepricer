import streamlit as st

# Home Page
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

# EDA and Visualization Page
def eda_visualization():
    st.title("EDA and Visualization")
    # Hier fügen Sie den Code für Seite 2 ein

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
    # Hier können Sie spezifische Optionen für das Modellieren hinzufügen, z.B. Datensubset-Auswahl, Preprocessing-Optionen, Modelltypen usw.