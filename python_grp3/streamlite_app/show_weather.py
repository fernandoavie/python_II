from show_forecasts import build_forecasts, build_df_forecasts_and_plot
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import random

import os

script_dir = os.path.dirname(__file__)

# local functions

# Set plotly template
px.defaults.template = "plotly_dark"


def show_weather():
    st.markdown("<h1 style='text-align: center;'>±24H Business Overview</h1>",
                unsafe_allow_html=True)
    path_EDA = os.path.join(
        script_dir, "../data/bike_data_cleaned_features.csv")
    data = pd.read_csv(path_EDA)
    start_year = 2011
    end_year = 2012

    # Generate a random year, month, and day
    random_year = random.randint(start_year, end_year)
    random_month = random.randint(1, 12)
    random_day = random.randint(1, 28)  # Use 28 to avoid issues with February

    # Create the start date
    start_dt = pd.Timestamp(
        year=random_year, month=random_month, day=random_day)

    # Create the end date (1 day1 after the start date)
    end_dt = start_dt + pd.Timedelta(days=1)

    # start_dt = pd.Timestamp('2012-12-19')
    # end_dt = pd.Timestamp('2012-12-20')

    # Preprocessing data
    weather_condition_mapping = {
        1: 'Clear Conditions, No Precipitation',
        2: 'Misty Conditions with Cloud Cover',
        3: 'Light Precipitation and Thunderstorms',
        4: 'Severe Weather Conditions'
    }

    # Apply the mapping to the 'weather_condition' column
    data['weather_condition'] = data['weather_condition'].map(
        weather_condition_mapping)

    # create weather_condition with emojis
    weather_condition_emoji = {
        'Clear Conditions, No Precipitation': '☀️',
        'Misty Conditions with Cloud Cover': '🌥️',
        'Light Precipitation and Thunderstorms': '🌦️',
        'Severe Weather Conditions': '⛈️'
    }

    data['Weather'] = data['weather_condition'].map(
        weather_condition_emoji)

    # According to the UCI Machine Learning Repository,the temperatures in the bike sharing dataset are normalized based on a -8 to +39 degrees Celsius scale.
    data['temp_celsius_real'] = data['temp_celsius'] * (39 - (-8)) + (-8)
    data['atem_celsius_real'] = data['atem_celsius'] * (39 - (-8)) + (-8)
    # We convert the humidity to a percentage:
    data['humidity_real'] = data['humidity'] * 100
    # km/h, you can multiply by the maximum speed (67 km/h):
    data['windspeed_real'] = data['windspeed'] * 67

    # Set the data to the last 48hours
    data['date'] = pd.to_datetime(data['date'])
    last_24_data = data[(data['date'] >= pd.Timestamp('2012-12-19'))
                        & (data['date'] <= pd.Timestamp('2012-12-20'))]

    # Retrieve Forecasts
    yt, yt_preds, date_range_hours = build_forecasts(
        start_date_str=str(start_dt), end_date_str=str(end_dt))

    # Block one Main KPIs
    # Weather Condistion for the last hour
    center_spacer1, center_column, center_spacer2 = st.columns([1, 6, 1])
    with center_column:
        st.metric(" ",
                  f"{last_24_data['weather_condition'].mode().values[0]}")

    left_spacer, score1, score2, score3, score4, score5, right_spacer = st.columns([
        1, 5, 5, 5, 5, 5, 1])

    previous_day_data = data[(data['date'] >= (
        start_dt - pd.Timedelta(days=1))) & (data['date'] < start_dt)]

    with score1:
        delta_bikes = int(last_24_data['total_bike_ct'].sum(
        ) - previous_day_data['total_bike_ct'].sum())
        st.metric("Total Bikes",
                  f"{last_24_data['total_bike_ct'].sum()}", delta=delta_bikes)
    with score2:
        d = (yt_preds.sum(
        ) - last_24_data['total_bike_ct'].sum()) / last_24_data['total_bike_ct'].sum()
        st.metric("Next Day Forecasted Bikes",
                  f"{int(yt_preds.sum())}", delta=f"{round(d * 100, 2)}%")
    with score3:
        delta_tot_user = int(last_24_data['casual_user_ct'].sum(
        ) - previous_day_data['casual_user_ct'].sum())
        st.metric(
            "Total Casual", f"{last_24_data['casual_user_ct'].sum()}", delta=delta_tot_user)
    with score4:
        delta_tot_user = int(last_24_data['registered_user_ct'].sum(
        ) - previous_day_data['registered_user_ct'].sum())
        st.metric(
            "Total Registered", f"{last_24_data['registered_user_ct'].sum()}", delta=delta_tot_user)
    with score5:
        delta_cas_reg_rate = float(last_24_data['casual_to_registered_ratio'].mean(
        ) - previous_day_data['casual_to_registered_ratio'].mean())
        st.metric("User Type Ratio",
                  f"{last_24_data['casual_to_registered_ratio'].mean():.2f}%", delta=f"{delta_cas_reg_rate:.2f}%")

    # Weather block

    left_spacer, w1, w2, w3, w4, right_spacer = st.columns([
        1, 5, 5, 5, 5, 1])
    with w1:
        delta_w1 = last_24_data['temp_celsius_real'].mean(
        ) - previous_day_data['temp_celsius_real'].mean()
        st.metric("Average Temperature",
                  f"{last_24_data['temp_celsius_real'].mean():.2f}°C", delta=f"{round(delta_w1, 2)}°C")
    with w2:
        delta_w2 = last_24_data['atem_celsius_real'].mean(
        ) - previous_day_data['atem_celsius_real'].mean()
        st.metric("Felt Temperature",
                  f"{last_24_data['atem_celsius_real'].mean():.2f}°C", delta=f"{round(delta_w2, 2)}°C")
    with w3:
        delta_w3 = last_24_data['humidity_real'].mean(
        ) - previous_day_data['humidity_real'].mean()
        st.metric("Avg Humidity",
                  f"{last_24_data['humidity_real'].mean():.2f}%", delta=f"{round(delta_w3, 2)}%")
    with w4:
        delta_w4 = last_24_data['windspeed_real'].mean(
        ) - previous_day_data['windspeed_real'].mean()
        st.metric("Avg Windspeed km/h",
                  f"{round(last_24_data['windspeed_real'].mean(),2):.2f}", delta=f"{round(delta_w4, 2)}km/h")

    # Plot forecast + Line_Polar

    # use build_forcast_and_plot function to get the forecast and plot
    yt, yt_preds, date_range_hours = build_forecasts(
        start_dt, end_dt)

    yt_combined, fig_forecast = build_df_forecasts_and_plot(yt, yt_preds)

    st.plotly_chart(fig_forecast, use_container_width=True)

    # display a histogram with the distribution of the bike count per hour
    fig = px.bar(last_24_data, x='date', y='total_bike_ct', labels={
        'date': 'Hours', 'total_bike_ct': 'Bike Count'}, title='Last 24hours bike count')
    fig.add_hline(y=last_24_data['total_bike_ct'].mean(
    ), line_dash="dash", annotation_text="Mean", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True)

    # Display pandas dataframe with weather variable for the last 24 hours with date + hour as index
    df_display = last_24_data.set_index('date')
    df_display = df_display.rename(columns={
        'total_bike_ct': 'Bike Count',
        'temp_celsius_real': 'Temperature (°C)',
        'atem_celsius_real': 'Apparent Temperature (°C)',
        'humidity_real': 'Humidity (%)',
        'windspeed_real': 'Wind Speed (km/h)'
    })
    st.write(df_display[['Weather', 'Bike Count', 'Temperature (°C)',
                         'Apparent Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']])

    # Disclaimer
    st.write(f"Data from {start_dt} to {end_dt}")
    st.caption("Random Date just for illustration purposes")


if __name__ == "__main__":
    st.set_page_config(page_title='GRP3 - Bike Rental Dashboard',
                       page_icon='🚲')
    # path_EDA = "/Users/lucazosso/Desktop/IE_Course/Term_2/Python II/Group_Assignement/python_grp3/data/bike_data_cleaned_features.csv"
    # data = pd.read_csv(path_EDA)
    # st.markdown("<h1 style='text-align: center;'>±24H Business Overview</h1>",
    #             unsafe_allow_html=True)

    show_weather()
