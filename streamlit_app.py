from collections import defaultdict
import streamlit as st
from pathlib import Path
from src.models.predict_model import make_prediction
from src.features.build_features import add_answer
import pandas as pd

home = "Project Home"
data = "Data Sources"
features = "Feature Engineering"
training = "Model Training"


def render_home():
    st.subheader("Project Home Page")
    st.write("Explore the full potential of the cutting-edge Streamlit app WiseRidePricer designed to revolutionize your car buying and selling experience! Dive into a world of comprehensive data analysis and captivating visualizations derived from a rich dataset sourced from Autoscout24. Uncover total sales figures, discern dominant brands, unravel feature correlations, and delve into temporal shifts within the automotive market. Embark on a data-driven journey with this app, where you have a plethora of regression methods at your fingertips. Predicting sales prices becomes a breeze as you select and analyze specific car characteristics. WiseRidePricer goes beyond predictions – it empowers you with model evaluation metrics, aiding in performance assessment and assisting you in choosing the optimal model for your needs. Experience the ultimate in user-friendly interfaces with a streamlined dashboard. An intuitive platform that not only effectively communicates intricate results but also forecasts prices based on your chosen models is available. Elevate your car-related decisions with this app – your go-to companion for informed choices in the dynamic world of automotive transactions!")

    st.subheader("Make a prediction")
    with st.form("model_prediciton"):
        input_data = st.text_input(
            "Model Input", "Enter some input features to make a prediction"
        )
        is_submitted = st.form_submit_button()

    if not is_submitted:
        st.info("Press Submit to Make a Prediction")
        st.stop()
    st.subheader("Input:")
    st.write(input_data)
    prediction = make_prediction(input_data)
    st.subheader("Prediction:")
    st.write(prediction)


def render_data_directory(dir: Path):
    st.subheader(dir.name)
    file_types = defaultdict(int)
    all_files = []
    for sub_path in (
        x for x in dir.iterdir() if x.is_file() and not x.name.startswith(".")
    ):
        file_types[sub_path.suffix] += 1
        all_files.append(sub_path.name)
    st.write("Total Files in Directory: ", len(all_files))
    if len(all_files):
        st.write("Total Files per File Type")
        st.json(file_types)
        with st.expander("All Files"):
            st.json(all_files)
    for sub_dir in (
        x for x in dir.iterdir() if x.is_dir() and not x.name.startswith(".")
    ):
        render_data_directory(sub_dir)


def render_data():
    st.subheader("Data Source Information")
    st.write("The following data were gathered from the following sources:")
    render_data_directory(Path("data"))


def render_features():
    st.subheader("Feature Engineering Process")
    st.write("The following transformations were applied to the following datasets:")
    st.subheader("Adding Answer to Universe Feature example:")
    df = pd.DataFrame(
        [
            {"name": "alice", "favorite_animal": "dog"},
            {"name": "bob", "favorite_animal": "cat"},
        ]
    )
    st.write("Initial data")
    st.write(df)
    df = add_answer(df)
    st.write("Transformed data")
    st.write(df)


def render_training():
    st.subheader("Model Training Overview")
    st.write("The following models and hyperparameters were tested:")
    for sub_path in (
        x
        for x in Path("models").iterdir()
        if x.is_file() and not x.name.startswith(".")
    ):
        st.subheader(sub_path.name)
        st.write("Size in bytes: ", len(sub_path.read_bytes()))

display_page = st.sidebar.radio("View Page:", (home, data, features, training))
st.header("WiseRidePricer")

if display_page == home:
    render_home()
elif display_page == data:
    render_data()
elif display_page == features:
    render_features()
elif display_page == training:
    render_training()
