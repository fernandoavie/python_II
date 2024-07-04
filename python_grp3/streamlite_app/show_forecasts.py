import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
import pandas as pd
from datetime import timedelta

import os

script_dir = os.path.dirname(__file__)

# Set plotly template
px.defaults.template = "plotly_dark"


def build_forecasts(start_date_str, end_date_str):
    min_date = pd.to_datetime('2011-01-01')
    max_date = pd.to_datetime('2012-12-31')
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    if start_date >= min_date and end_date <= max_date:
        # Load the data
        bike_data = pd.read_csv(os.path.join(
            script_dir, "../data/bike_data_cleaned_features.csv"), index_col='date', parse_dates=['date'])
        X_prep = pickle.load(open(os.path.join(script_dir,
                                               "../streamlite_app/X_prep.pkl"), 'rb'))
        model = joblib.load(os.path.join(
            script_dir, "../streamlite_app/xgb_best_model.pkl"))

        bike_data.index = pd.to_datetime(bike_data.index)
        X_prep_df = pd.DataFrame(X_prep, index=bike_data.index)
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        yt = bike_data.loc[start_date:end_date, 'total_bike_ct']
        date_range_hours = (end_date - start_date).total_seconds() / 3600

        next_start_date = end_date + timedelta(days=1)
        next_end_date = next_start_date + (end_date - start_date)
        X_test = X_prep_df.loc[next_start_date:next_end_date]

        # Predict using our XGboost model
        yt_preds = model.predict(X_test)

    else:
        st.error(
            f"Invalid time period: {start_date} to {end_date}. Please select a date within the range from {min_date} to {max_date}.")
        st.stop()

    return yt, yt_preds, date_range_hours


def build_df_forecasts_and_plot(yt, yt_preds):
    # index
    start_date = yt.index[0]
    end_date = yt.index[-1]
    new_end_date = end_date + (end_date - start_date)
    new_index = pd.date_range(start=start_date, end=new_end_date, freq='H')

    # Aggregation
    yt_1_df = pd.DataFrame(
        yt_preds, index=new_index[-yt_preds.shape[0]:], columns=['yt_1'])
    # Concatenate yt and yt_1_df along the column axis
    yt_combined = pd.concat([yt, yt_1_df], axis=1)

    # plot

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=yt_combined.index, y=yt_combined['total_bike_ct'], mode='lines', name='Bike Count'))
    fig_forecast.add_trace(go.Scatter(
        x=yt_combined.index, y=yt_combined['yt_1'], mode='lines', name='Forecast', line=dict(dash='dot')))
    # fig_forecast.show()

    return yt_combined, fig_forecast


if __name__ == "__main__":
    st.set_page_config(
        page_title='GRP3 - Bike Rental Dashboard', page_icon='🚲')

    col1, col2 = st.columns([1, 1])

    with col1:
        start_date = st.date_input('Start Date', pd.to_datetime('2011-01-01'))

    with col2:
        end_date = st.date_input('End Date', pd.to_datetime('2011-01-02'))

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    yt, yt_preds, date_range_hours = build_forecasts(
        start_date, end_date)

    yt_combined, fig_forecast = build_df_forecasts_and_plot(yt, yt_preds)

    st.plotly_chart(fig_forecast, use_container_width=True)
