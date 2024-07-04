#my script
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Python Assignment Group 3")

path_EDA = "../data/bike_data_cleaned_features.csv"

data = pd.read_csv(path_EDA)

selected_feature = st.sidebar.selectbox('Select Feature', ['EDA', 'MODEL', 'FINAL PREDICTIONS'])

if selected_feature == 'EDA':
    st.header("EDA Process")

    st.text("Histogram total_bike_ct")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.histplot(data['total_bike_ct'])
    st.pyplot(fig)

    st.text("Trend of total_bike_ct")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(data.groupby('year_month')['total_bike_ct'].sum().reset_index())
    st.pyplot(fig)

    
elif selected_feature == 'MODEL':
    cat = ["bored", "happy", "bored", "bored", "happy", "bored"]
    dog = ["happy", "happy", "happy", "happy", "bored", "bored"]
    activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

    width = st.sidebar.slider("plot width", 1, 25, 3)
    height = st.sidebar.slider("plot height", 1, 25, 1)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(activity, dog, label="dog")
    ax.plot(activity, cat, label="cat")
    ax.legend()

    st.pyplot(fig)