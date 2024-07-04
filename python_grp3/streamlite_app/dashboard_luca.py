import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go

# Modeling libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
import joblib
import random

################################################### - Functions - ###################################################
# load data


def load_data():
    path = '../data/bike_data_cleaned_features.csv'
    df = pd.read_csv(path, index_col='date', parse_dates=True)
    return df

# Function to compare the previous period same day of the week


def compare_previous_period(df, date):
    # Ensure 'date' is in the right format for pandas operations
    date = pd.Timestamp(date)
    previous_period = date - timedelta(weeks=1)
    # Assuming 'df.index' is a DateTimeIndex; ensure we're comparing compatible types
    previous_period_same_day_data = df[df.index.dayofweek ==
                                       previous_period.weekday()]
    previous_period_same_day_data = previous_period_same_day_data.groupby(
        df.index.dayofweek).sum()
    return previous_period_same_day_data

# Filter data based on the time period selected


def filter_data(df, time_period, end_date):
    # Ensure end_date is compatible with pandas datetime
    now = pd.Timestamp(end_date)
    periods = {'Last hour': pd.Timedelta(hours=1), 'Last 24 hours': pd.Timedelta(days=1),
               'Last week': pd.Timedelta(weeks=1), 'Last month': pd.Timedelta(days=30),
               'Last quarter': pd.Timedelta(days=90), 'Last year': pd.Timedelta(days=365)}

    if time_period not in periods:
        raise ValueError(f"Invalid time period: {time_period}")

    start_date = now - periods[time_period]
    mask = (df.index >= start_date) & (df.index <= now)
    return df.loc[mask]


def calculate_delta(df, filtered_data, column, weekday, aggregation='sum'):
    # Filter rows where the weekday of the index matches the specified weekday
    df_weekday = df[df.index.weekday == weekday]
    filtered_weekday = filtered_data[filtered_data.index.weekday == weekday]

    # Calculate the aggregation for each week
    df_weekly_agg = df_weekday.resample('W')[column].agg(aggregation)
    filtered_weekly_agg = filtered_weekday.resample(
        'W')[column].agg(aggregation)

    # Calculate the delta: (this_week - last_week) / last_week
    delta = (filtered_weekly_agg.iloc[-1] - df_weekly_agg.shift(
        1).iloc[-1]) / df_weekly_agg.shift(1).iloc[-1]

    return delta

################################################### - Modeling - ###################################################


def forecast_model(df, horizon):
    model = joblib.load(
        '../streamlite_app/xgb_best_model.pkl')
    df_forecast = df[-horizon:]
    # split target
    X = df_forecast.drop(
        columns=['total_bike_ct', 'casual_user_ct', 'registered_user_ct'], axis=1)
    # Preprocessing
    labelenc_val_cat = ['year', 'is_holiday', 'is_workingday',
                        'hourly_rental_deviation_label', 'weather_condition']
    print("Label Encoding: ", labelenc_val_cat)
    ohe_val_cat = ['season', 'day_part']
    print("One Hot Encoding: ", ohe_val_cat)
    # take everything apart from target variable + casual and registered user count
    num_vars = df_forecast.select_dtypes(include=['float', 'int']).drop(
        ['total_bike_ct', 'casual_user_ct', 'registered_user_ct'], axis=1).columns
    print("Numerical Variables: ", num_vars)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_labelenc", OrdinalEncoder(), labelenc_val_cat),
            ("cat_ohe", OneHotEncoder(), ohe_val_cat),
            ("num", RobustScaler(), num_vars)
        ],
    )
    X_prep = preprocessor.fit_transform(X)

    # Forecast
    forecast = model.predict(X_prep)
    return forecast

# Main Plot (after score cards)

################################################### - Main - ###################################################


if __name__ == '__main__':
    df = load_data()

    # Title + Image
    st.set_page_config(page_title='GRP3 - Bike Rental Dashboard',
                       page_icon='🚲')
    st.markdown("<h1 style='text-align: center;'>Bike Rental Dashboard</h1>",
                unsafe_allow_html=True)
    st.image('../data/bike.jpg', use_column_width=True)

    # Set layout

    # Side bar
    st.sidebar.title('Dash Filters')
    # Filters
    start_date, end_date = pd.Timestamp(2011, 1, 1), pd.Timestamp(2012, 12, 31)

    selected_date_range = st.sidebar.date_input(
        "Select the time period and date range for the analysis",
        value=(start_date, end_date),
        min_value=start_date,
        max_value=end_date,
        format="YYYY/MM/DD"
    )

    selected_start_date, selected_end_date = pd.Timestamp(
        selected_date_range[0]), pd.Timestamp(selected_date_range[1])
    filtered_data = df[(df.index >= selected_start_date)
                       & (df.index <= selected_end_date)]

    # Score Cards - handling delta calculation explicitly for Streamlit compatibility
    score1, score2, score3, score4 = st.columns(4)
    with score1:
        total_bikes = filtered_data['total_bike_ct'].sum()
        delta = calculate_delta(
            df, filtered_data, 'total_bike_ct', selected_end_date.weekday())
        st.metric(label="Total Bikes",
                  value=f"{int(total_bikes):,}",
                  delta=f"{delta * 100:.2f}%",
                  help="Total bikes rented compared to last period same weekday")

    with score2:
        avg_bikes = filtered_data['total_bike_ct'].mean()
        delta_avg = calculate_delta(
            df, filtered_data, 'total_bike_ct', selected_end_date.weekday(), 'mean')
        st.metric(label="Daily Avg. Bikes Rental",
                  value=f"{int(avg_bikes):,}",
                  delta=f"{delta_avg * 100:.2f}%",
                  help="Daily Avg bikes rented compared to last period same weekday")
    with score3:
        total_registered = filtered_data['registered_user_ct'].sum()
        delta_tot_registered = calculate_delta(
            df, filtered_data, 'registered_user_ct', selected_end_date.weekday())
        st.metric(label="Total Registered Users",
                  value=f"{int(total_registered):,}",
                  delta=f"{delta_tot_registered * 100:.2f}%",
                  help="Total Registered users rented compared to last period same weekday")

    with score4:
        ratio_casual_registered = filtered_data['casual_to_registered_ratio'].mean(
        )
        delta_avg_casual_registered_ratio = calculate_delta(
            df, filtered_data, 'casual_to_registered_ratio', selected_end_date.weekday(), 'mean')
        st.metric(label="Avg. Casual vs Registered User Ratio",
                  value=f"{float(ratio_casual_registered)*100:.2f}%",
                  delta=f"{delta_avg_casual_registered_ratio * 100:.2f}%",
                  help="Avg. Casual to Registered users compared to last period same weekday")

    # Using forcast model function, plot 1 graph that shows the hourly forecast for the next 24 hours and another line that shows the actual values for the last 24hours
    forcast_horizon = 24

    # Calculate the forecast for the next 24 hours
    forecast = forecast_model(df, 24)

    # Get the actual data for the last 24 hours
    actual = df['total_bike_ct'][-24:]

    # Get the dates for the forecast horizon
    forecast_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')

    # Create a new DataFrame to store the actual and forecast data
    df_plot = pd.concat([
        pd.DataFrame({'Value': actual, 'Type': 'Actual'}, index=actual.index),
        pd.DataFrame({'Value': forecast, 'Type': 'Forecast'},
                     index=forecast_dates)
    ])

    # Create a line plot with Plotly
    fig = go.Figure()
    for type, group in df_plot.groupby('Type'):
        fig.add_trace(go.Scatter(x=group.index, y=group['Value'], mode='lines', name=type,
                                 line=dict(color='red' if type == 'Forecast' else 'blue')))
    fig.update_layout(title='Bike Rental Forecast',
                      xaxis_title='Date', yaxis_title='Total Bike Count')

    # Display the plot in Streamlit
    st.plotly_chart(fig)
