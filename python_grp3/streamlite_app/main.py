# my script
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import plotly.express as px

import os

# local script
from show_eda import show_eda
from show_forecasts import build_forecasts, build_df_forecasts_and_plot
from show_weather import show_weather

script_dir = os.path.dirname(__file__)

st.set_option('deprecation.showPyplotGlobalUse', False)

def weather(data):

    filtered_data = data[(data['date'] >= start_date)
                         & (data['date'] <= end_date)]

    # Preprocessing data
    weather_condition_mapping = {
        1: 'Clear Conditions, No Precipitation',
        2: 'Misty Conditions with Cloud Cover',
        3: 'Light Precipitation and Thunderstorms',
        4: 'Severe Weather Conditions'
    }

    # Apply the mapping to the 'weather_condition' column
    filtered_data['weather_condition'] = filtered_data['weather_condition'].map(
        weather_condition_mapping)

    # create weather_condition with emojis
    weather_condition_emoji = {
        'Clear Conditions, No Precipitation': '☀️',
        'Misty Conditions with Cloud Cover': '🌥️',
        'Light Precipitation and Thunderstorms': '🌦️',
        'Severe Weather Conditions': '⛈️'
    }

    filtered_data['Weather'] = filtered_data['weather_condition'].map(
        weather_condition_emoji)


    # According to the UCI Machine Learning Repository,the temperatures in the bike sharing dataset are normalized based on a -8 to +39 degrees Celsius scale.
    filtered_data['temp_celsius_real'] = filtered_data['temp_celsius'] * (39 - (-8)) + (-8)
    filtered_data['atem_celsius_real'] = filtered_data['atem_celsius'] * (39 - (-8)) + (-8)
    # We convert the humidity to a percentage:
    filtered_data['humidity_real'] = filtered_data['humidity'] * 100
    # km/h, you can multiply by the maximum speed (67 km/h):
    filtered_data['windspeed_real'] = filtered_data['windspeed'] * 67

    # Weather Condistion for the last hour
    center_spacer1, center_column, center_spacer2 = st.columns([1, 6, 1])
    with center_column:
        st.metric(" ",
            f"{filtered_data['weather_condition'].mode().values[0]}")

    # display a histogram with the distribution of the bike count per hour

    fig = px.bar(filtered_data, x='date', y='total_bike_ct', labels={
        'date': 'Period', 'total_bike_ct': 'Bike Count'}, title='Bike count')
    fig.add_hline(y=data['total_bike_ct'].mean(
    ), line_dash="dash", annotation_text="Mean", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True)

    # Display pandas dataframe with weather variable for the last 24 hours with date + hour as index
    df_display = filtered_data.set_index('date')
    df_display = df_display.rename(columns={
        'total_bike_ct': 'Bike Count',
        'temp_celsius_real': 'Temperature (°C)',
        'atem_celsius_real': 'Apparent Temperature (°C)',
        'humidity_real': 'Humidity (%)',
        'windspeed_real': 'Wind Speed (km/h)'
    })
    
    st.write(df_display[['Weather', 'Bike Count', 'Temperature (°C)', 'Apparent Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']])

def load_data():
    path_EDA = os.path.join(script_dir, '../data/bike_data_cleaned_features.csv')
    data = pd.read_csv(path_EDA)
    return data


def main(data):
    data['date'] = pd.to_datetime(data['date'])
    filtered_data = data[(data['date'] >= start_date)
                         & (data['date'] <= end_date)]

    # Calculate the total number of bikes
    total_bikes = filtered_data['total_bike_ct'].sum()

    score1, score2, score3, score4 = st.columns(4)

    with score1:
        data['date'] = pd.to_datetime(data['date'])
        filtered_data = data[(data['date'] >= start_date)
                             & (data['date'] <= end_date)]

        # Calculate the total number of bikes
        total_bikes = filtered_data['total_bike_ct'].sum()

        # Display the scorecard
        st.header('Total Bikes')
        st.write(f'{total_bikes}')

    return 0


def calculate_total_bikes(data, start_date, end_date):

    # Ensure date column is datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Filter data based on date range
    filtered_data = data[(data['date'] >= start_date)
                         & (data['date'] <= end_date)]

    # Calculate total bikes
    total_bikes = filtered_data['total_bike_ct'].sum()
    return total_bikes


def calculate_avg_rented(data, start_date, end_date):

    # Ensure date column is datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Filter data based on date range
    filtered_data = data[(data['date'] >= start_date)
                         & (data['date'] <= end_date)]

    # Calculate total bikes
    avg_bikes = filtered_data['total_bike_ct'].mean()

    return math.ceil(avg_bikes)


def calculate_registered_users(data, start_date, end_date):

    # Ensure date column is datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Filter data based on date range
    filtered_data = data[(data['date'] >= start_date)
                         & (data['date'] <= end_date)]

    # Calculate total bikes
    registered_users = filtered_data['registered_user_ct'].sum()
    return registered_users


def ratio_registered_casual(data, start_date, end_date):

    # Ensure date column is datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Filter data based on date range
    filtered_data = data[(data['date'] >= start_date)
                         & (data['date'] <= end_date)]

    registered_users = filtered_data['registered_user_ct'].sum()
    casual_users = filtered_data['casual_user_ct'].sum()

    # Calculate total bikes
    ratio = registered_users/casual_users
    return round(ratio, 2)


def eda_process(data):
    st.header("EDA Process")

    st.text("Histogram total_bike_ct")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.histplot(data['total_bike_ct'])
    st.pyplot(fig)

    st.text("Trend of total_bike_ct")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(data.groupby('year_month')[
                 'total_bike_ct'].sum().reset_index())
    st.pyplot(fig)


def select_features(data):
    selected_feature = st.sidebar.selectbox(
        'Select Process', ['HOME', 'CURRENT', 'EDA', 'MODEL', ])
    if selected_feature == 'HOME':

        # Calculate the duration of the current period in days
        period_duration_days = (end_date - start_date).days
        start_date_prev = start_date - \
            pd.Timedelta(days=period_duration_days) - pd.Timedelta(days=1)

        end_date_prev = start_date - pd.Timedelta(days=1)

        if start_date_prev < pd.to_datetime('2011-01-01'):
            start_date_prev = pd.to_datetime('2011-01-01')

        if end_date_prev <= pd.to_datetime('2011-01-01'):
            end_date_prev = pd.to_datetime('2011-01-02')

        column1, column2, column3, column4 = st.columns(4)
        with column1:
            total_bikes_value = calculate_total_bikes(
                data, start_date, end_date)
            previous_total_bikes_value = calculate_total_bikes(
                data, start_date_prev, end_date_prev)
            total_bikes_delta = (
                total_bikes_value - previous_total_bikes_value) / previous_total_bikes_value * 100
            st.metric(label="Total Bikes Rented",
                      value=f"{calculate_total_bikes(data, start_date, end_date):,}",
                      delta=f"{total_bikes_delta:.2f}%",
                      help="Total bikes rented in this period")
        with column2:
            avg_bikes_value = calculate_avg_rented(data, start_date, end_date)
            previous_avg_bikes_value = calculate_avg_rented(
                data, start_date_prev, end_date_prev)
            avg_bikes_delta = (
                avg_bikes_value - previous_avg_bikes_value) / previous_avg_bikes_value * 100
            st.metric(label="Average Bikes Rented",
                      value=f"{avg_bikes_value:,}",
                      delta=f"{avg_bikes_delta:.2f}%",
                      help="Average bikes rented in this period")

        with column3:
            registered_users_value = calculate_registered_users(
                data, start_date, end_date)
            previous_registered_users_value = calculate_registered_users(
                data, start_date_prev, end_date_prev)
            registered_users_delta = (
                registered_users_value - previous_registered_users_value) / previous_registered_users_value * 100
            st.metric(label="Total Users Registered",
                      value=f"{registered_users_value:,}",
                      delta=f"{registered_users_delta:.2f}%",
                      help="Users registered in this period")

        with column4:
            ratio_value = ratio_registered_casual(data, start_date, end_date)
            previous_ratio_value = ratio_registered_casual(
                data, start_date_prev, end_date_prev)
            ratio_delta = (ratio_value - previous_ratio_value) / \
                previous_ratio_value * 100
            st.metric(label="Ratio Registered/Casual",
                      value=f"{ratio_value:,}",
                      delta=f"{ratio_delta:.2f}%",
                      help="Ratio between registered users and casual users")

        st.write(
            f"Previous period: {start_date_prev.date()} - {end_date_prev.date()}")

        column2_1, column2_2, column2_3, column2_4 = st.columns(4)

        with column2_1:
            st.metric(label="Total Bikes Rented",
                      value=f"{calculate_total_bikes(data, start_date_prev, end_date_prev):,}",
                      help="Total bikes rented in this period")
        with column2_2:
            st.metric(label="Average Bikes Rented",
                      value=f"{calculate_avg_rented(data, start_date_prev, end_date_prev):,}",
                      help="Average bikes rented in this period")
        with column2_3:
            st.metric(label="Total Users Registered",
                      value=f"{calculate_registered_users(data, start_date_prev, end_date_prev):,}",
                      help="Users registered in this period")
        with column2_4:
            st.metric(label="Ratio Registered/Casual",
                      value=f"{ratio_registered_casual(data, start_date_prev, end_date_prev):,}",
                      help="Ratio between users that have been registered and the casuals ones")

        # Forecast Plot
        st.markdown(
            "<h2 style='text-align: center;'>Bike Trend Forecast</h2>", unsafe_allow_html=True)
        yt, yt_preds, date_range_hours = build_forecasts(
            start_date, end_date)

        yt_combined, fig_forecast = build_df_forecasts_and_plot(yt, yt_preds)

        delta_bikes = int(yt_preds.sum() - yt.sum())/yt.sum()

        st.metric(label="Next Day Forecasted Bikes",
                  value=f"{int(yt_preds.sum())}",
                  delta=f"{delta_bikes:.2%}",
                  help="Forecasted number of bikes for the next day")

        st.plotly_chart(fig_forecast, use_container_width=True)

        # ADD MORE CHART HERE!
        weather(data)

############################################# SWITCHING FROM HOME #############################################
    elif selected_feature == 'EDA':
        path_EDA = os.path.join(script_dir, "../data/bike_data_cleaned.csv")
        data = pd.read_csv(path_EDA)
        show_eda(data)

    elif selected_feature == 'MODEL':
        # Load image
        st.image('../python_grp3/data/table_model_perf.png')
        st.image('../python_grp3/data/residuals.png')

    elif selected_feature == 'CURRENT':
        show_weather()


if __name__ == '__main__':
    st.set_page_config(page_title='GRP3 - Bike Rental Dashboard',
                       page_icon='🚲')
    st.markdown("<h1 style='text-align: center;'>Bike Rental Dashboard</h1>",
                unsafe_allow_html=True)

    df = load_data()

    col1, col2 = st.columns([1, 1])

    min_date = pd.to_datetime('2011-01-01')
    max_date = pd.to_datetime('2012-12-31')

    with col1:
        start_date = st.date_input('Start Date', pd.to_datetime('2011-01-01'))

    with col2:
        end_date = st.date_input('End Date', pd.to_datetime('2011-01-07'))

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    select_features(df)
