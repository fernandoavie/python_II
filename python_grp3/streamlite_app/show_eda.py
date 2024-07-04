import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import os

script_dir = os.path.dirname(__file__)

# Set plotly template
px.defaults.template = "plotly_dark"


def show_eda(data):
    st.markdown("""<h1 style='text-align: center;'>Exploratory Data Analysis</h1>""",
                unsafe_allow_html=True)

    left_spacer, score1, score2, score3, score4, right_spacer = st.columns([
        1, 2, 2, 2, 2, 1])

    # show # null values
    with score1:
        # Sum of all null values in the DataFrame
        total_null_values = data.isnull().sum().sum()
        st.metric("# Null Values", total_null_values, help="No missing values")

    # show duplicated rows
    with score2:
        # Count of all duplicated rows
        total_duplicated_rows = int(data.duplicated().sum())
        st.metric("# Duplicated Rows", total_duplicated_rows)

    with score3:
        # Outlier Ratio
        # Calculate Q1, Q3, and IQR
        Q1 = data['total_bike_ct'].quantile(0.25)
        Q3 = data['total_bike_ct'].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = data[(data['total_bike_ct'] < lower_bound) | (
            data['total_bike_ct'] > upper_bound)]['total_bike_ct']

        # Count of outliers
        outliers_count = outliers.count()
        outliers_ratio = outliers_count / data.shape[0]
        st.metric("Outliers Ratio", f"{outliers_ratio:.2%}")
    with score4:
        # Nbr. of observations
        observations = data.shape[0]
        st.metric("Nbr. of Observations", f"{observations:,}")

    # Second Block

    # Plots
    col1, col2 = st.columns(2)

    with col1:
        # Histogram for Bike Rental Distribution
        fig1 = px.histogram(data, x="total_bike_ct", nbins=100,
                            title="Bike Rental Distribution")
        fig1.update_layout(margin=dict(
            l=20, r=20, t=50, b=20), showlegend=False)
        # Set to use the full width of the column
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Line plot for Year-Month Trend
        year_month_data = data.groupby('year_month')[
            'total_bike_ct'].sum().reset_index()
        fig2 = px.line(year_month_data, x='year_month',
                       y='total_bike_ct', title="Year-Month Trend of Bike Rentals")
        fig2.update_layout(margin=dict(
            l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Third Block
    # Cummulative Chart
    # show cumulative sum of bike counts by year_month
    data['year_month'] = pd.to_datetime(data['year_month'])
    bike_data_y1 = data[data['year'] == 0]
    bike_data_y2 = data[data['year'] == 1]

    # Calculate cumulative sum for each year
    cumulative_y1 = bike_data_y1.groupby(
        'year_month')['total_bike_ct'].sum().cumsum().reset_index()
    cumulative_y2 = bike_data_y2.groupby(
        'year_month')['total_bike_ct'].sum().cumsum().reset_index()

    # Extract the month
    cumulative_y1['month'] = cumulative_y1['year_month'].dt.month
    cumulative_y2['month'] = cumulative_y2['year_month'].dt.month

    # Create a line plot for 2011
    line_plot_y1 = px.line(cumulative_y1, x='month', y='total_bike_ct', labels={
                           'total_bike_ct': 'Cumulative Bike Count 2011'})

    # Create a line plot for 2012
    line_plot_y2 = px.line(cumulative_y2, x='month', y='total_bike_ct', labels={
                           'total_bike_ct': 'Cumulative Bike Count 2012'}, color_discrete_sequence=['red'])

    # Combine the plots
    line_plot = go.Figure()
    line_plot.add_trace(go.Scatter(x=line_plot_y1.data[0]['x'], y=line_plot_y1.data[0]['y'],
                                   mode='lines', name='2011',
                                   hovertemplate="Month: %{x}<br>Total: %{y}<extra></extra>"))
    line_plot.add_trace(go.Scatter(x=line_plot_y2.data[0]['x'], y=line_plot_y2.data[0]['y'],
                                   mode='lines', name='2012',
                                   hovertemplate="Month: %{x}<br>Total: %{y}<extra></extra>"))
    line_plot.update_layout(title="Cumulative Bike Rental Count by Month",
                            xaxis_title="Year - Month", yaxis_title="Cumulative Bike Count")

    st.plotly_chart(line_plot)

    # 4 Block: Other Insights
    df_is_holiday_user_types = data[data['is_holiday'] == 1].groupby(
        'hour')[['casual_user_ct', 'registered_user_ct']].mean().reset_index()
    df_is_not_holiday_user_types = data[data['is_holiday'] == 0].groupby(
        'hour')[['casual_user_ct', 'registered_user_ct']].mean().reset_index()

    fig1 = make_subplots(rows=1, cols=2, subplot_titles=(
        "Distribution of User Types on Holidays", "Distribution of User Types not on Holidays"))

    # Adding bar traces for holidays
    fig1.add_trace(go.Bar(
        x=df_is_holiday_user_types['hour'], y=df_is_holiday_user_types['casual_user_ct'], name="Casual Users - Holiday"), row=1, col=1)
    fig1.add_trace(go.Bar(
        x=df_is_holiday_user_types['hour'], y=df_is_holiday_user_types['registered_user_ct'], name="Registered Users - Holiday"), row=1, col=1)

    # Adding bar traces for non-holidays
    fig1.add_trace(go.Bar(
        x=df_is_not_holiday_user_types['hour'], y=df_is_not_holiday_user_types['casual_user_ct'], name="Casual Users - Non-Holiday"), row=1, col=2)
    fig1.add_trace(go.Bar(x=df_is_not_holiday_user_types['hour'], y=df_is_not_holiday_user_types[
                   'registered_user_ct'], name="Registered Users - Non-Holiday"), row=1, col=2)

    fig1.update_layout(height=600, width=1000,
                       title_text="Bike Temporal Analysis")

    # Rescale y-axis
    fig1.update_yaxes(range=[0, 400], row=1, col=1)
    fig1.update_yaxes(range=[0, 400], row=1, col=2)
    # Show plot
    st.plotly_chart(fig1)

    # 5 Block:
    # 5 Block: Weather Analysis
    # Group data by weather and calculate the mean bike count
    weather_data = data.groupby(
        'weather_condition')['total_bike_ct'].mean().reset_index()

    # Create a bar chart
    weather_chart = px.bar(weather_data, x='weather_condition', y='total_bike_ct', labels={
                           'total_bike_ct': 'Average Bike Count', 'weather': 'Weather Condition'})

    # Update layout
    weather_chart.update_layout(title_text="Bike Usage by Weather Condition")

    weather_chart.update_xaxes(
        tickvals=[1, 2, 3, 4],
        ticktext=['Clear and Variably Cloudy Conditions, No Precipitation',
                  'Misty Conditions with Varied Cloud Cover',
                  'Light Precipitation and Thunderstorms',
                  'Severe Weather Conditions']
    )

    # Show plot
    st.plotly_chart(weather_chart)


if __name__ == "__main__":
    st.set_page_config(page_title='GRP3 - Bike Rental Dashboard',
                       page_icon='🚲')
    path_EDA = os.path.join(
        script_dir, "../python_grp3/data/bike_data_cleaned.csv")
    data = pd.read_csv(path_EDA)
    show_eda(data)
