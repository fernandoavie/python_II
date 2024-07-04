import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import os

script_dir = os.path.dirname(__file__)
# Set plotly template
px.defaults.template = "plotly_dark"


def show_model(bike_data):
    st.markdown("""<h1 style='text-align: center;'>Modelings</h1>""",
                unsafe_allow_html=True)

    # show correlations
    numerical_features = bike_data.select_dtypes(
        include=['int64', 'float64']).columns
    correlations = bike_data[numerical_features].corr()
    # Create a pandas heatmap dataframe from the correlation matrix only using pandas
    mask = np.tril(np.ones_like(correlations, dtype=bool))

    # Apply the mask to the correlation matrix
    masked_correlations = correlations.mask(mask)

    # Create a pandas heatmap dataframe from the masked correlation matrix
    heatmap_df = masked_correlations.style.background_gradient(
        cmap='coolwarm', axis=None)

    st.dataframe(heatmap_df)

    correlations_w_target = bike_data[numerical_features].corrwith(
        bike_data['total_bike_ct'])

    # Create a pandas heatmap dataframe from the correlation matrix only using pandas
    corr_wtarget = correlations_w_target.sort_values(ascending=False).to_frame(
    ).style.background_gradient(cmap='coolwarm', axis=None)

    st.dataframe(corr_wtarget)


if __name__ == "__main__":
    st.set_page_config(page_title='GRP3 - Bike Rental Dashboard',
                       page_icon='🚲')
    path_prep = os.path.join(
        script_dir, "../python_grp3/data/bike_data_cleaned_features.csv")
    data = pd.read_csv(path_prep)
    show_model(data)
