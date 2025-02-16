"""
RECOMMENDED to run it via docker instead of local.
"""

import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="🚗 Vehicle Entry Stats Dashboard", layout="wide")
st.title("🚗 Vehicle Entry and Forecast Stats")

col1, col2 = st.columns([2, 1])
with col1:
    refresh_rate = st.slider("Refresh Rate (seconds)",
                             min_value=1, max_value=10, value=1)
    st.subheader("Live Stats & Chart")
    live_count_placeholder_enter = st.empty()
    live_count_placeholder_leave = st.empty()
    forecast_chart_placeholder = st.empty()

with col2:
    st.subheader("Summary")
    summary_placeholder = st.empty()

# Retry logic for fetching live stats
def fetch_data_with_retries(url, retries=5, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch data: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                st.warning(f"[WARNING] Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(delay)
            else:
                st.error(f"[ERROR] Max retries reached. Could not fetch data: {e}")
                return None

@st.cache_data(ttl=1)
def get_live_stats():
    return fetch_data_with_retries("http://api:8000/api/stats/live")

@st.cache_data(ttl=60)
def get_forecast():
    return fetch_data_with_retries("http://api:8000/api/forecast")

timestamps, entries, leaves = [], [], []
while True:
    stats = get_live_stats()
    if stats:
        timestamps.append(time.strftime('%H:%M:%S'))
        entries.append(stats['last_visited_one_second'])
        leaves.append(stats['last_leave_one_second'])
        live_count_placeholder_enter.metric(
            "Vehicles Entered (Last Second)", stats['last_visited_one_second'])
        live_count_placeholder_leave.metric(
            "Vehicles Left (Last Second)", stats['last_leave_one_second'])
        summary_placeholder.text(
            f"Total Vehicles Visited: {stats['total_vehicle_visited']}")

    forecast_data = get_forecast()

    if forecast_data and 'forecast' in forecast_data:
        try:
            df = pd.DataFrame(forecast_data['forecast'])
            df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
            now = datetime.now()
            start_time = now - timedelta(hours=2)
            end_time = now + timedelta(hours=2)
            df = df[(df['ds'] >= start_time) & (df['ds'] <= end_time)]
            fig = px.line(df, x='ds', y='yhat', title='Vehicle Count Forecasting (+1 hours)', labels={
                        'ds': 'Time', 'yhat': 'Vehicle Count'})
            fig.add_scatter(x=df['ds'], y=df['yhat_upper'], mode='lines', line=dict(
                dash='dot'), name='Upper Bound')
            fig.add_scatter(x=df['ds'], y=df['yhat_lower'], mode='lines', line=dict(
                dash='dot'), name='Lower Bound')
            forecast_chart_placeholder.plotly_chart(fig)
        except Exception as e:
            st.error(f"[ERROR] Failed to display forecast: {e}")

    time.sleep(refresh_rate)
    st.rerun()
