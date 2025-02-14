import streamlit as st
import requests
import plotly.express as px
import time

st.set_page_config(page_title="🚗 Vehicle Entry Stats Dashboard", layout="wide")
st.title("🚗 Vehicle Entry Stats")

# Fixed Layout Sections
col1, col2 = st.columns([2, 1])
with col1:
    refresh_rate = st.slider(
        "Refresh Rate (seconds)", min_value=1, max_value=10, value=1)

    st.subheader("Live Stats & Chart")
    # chart_placeholder = st.container()

    live_count_placeholder_enter = st.empty()
    live_count_placeholder_leave = st.empty()
with col2:
    st.subheader("Summary")
    summary_placeholder = st.empty()

with st.sidebar:
    with st.echo():
        st.write("This code will be printed to the sidebar.")

    # with st.spinner("Loading..."):
    #     time.sleep(5)
    # st.success("Done!")


@st.cache_data(ttl=1)
def get_live_stats():
    try:
        response = requests.get("http://127.0.0.1:8000/api/stats/live")
        return response.json()
    except Exception as e:
        st.error(f"[ERROR] Failed to fetch stats: {e}")
        return None


# Track Data for Chart
timestamps, entries, leaves = [], [], []

while True:
    stats = get_live_stats()
    if stats:
        timestamps.append(time.strftime('%H:%M:%S'))
        entries.append(stats['last_visited_one_second'])
        leaves.append(stats['last_leave_one_second'])

        # with chart_placeholder:
        #     fig = px.line(
        #         x=timestamps,
        #         y=[entries, leaves],
        #         labels={"x": "Time", "y": "Count"},
        #         title="Vehicle Entries/Exits Over Time",
        #         color_discrete_sequence=["green", "red"]
        #     )
        #     st.plotly_chart(fig, use_container_width=True)

        live_count_placeholder_enter.metric(
            "Vehicles Entered (Last Second)", stats['last_visited_one_second'])
        live_count_placeholder_leave.metric(
            "Vehicles Left (Last Second)", stats['last_leave_one_second'])
        summary_placeholder.text(
            f"Total Vehicles Visited: {stats['total_vehicle_visited']}")

    time.sleep(refresh_rate)
    st.rerun()
