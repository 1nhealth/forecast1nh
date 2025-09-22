# pages/8_PC_Performance.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# Import all calculation functions and the new formatting helper
from pc_calculations import calculate_heatmap_data, calculate_average_time_metrics, calculate_top_status_flows
from helpers import format_days_to_dhm

# --- Page Configuration ---
st.set_page_config(page_title="PC Performance", page_icon="📞", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📞 PC Performance Dashboard")
st.info("This dashboard analyzes the operational efficiency and patterns of the Pre-Screening team's activities.")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data from Session State ---
processed_data = st.session_state.referral_data_processed
ts_col_map = st.session_state.ts_col_map
parsed_status_history_col = "Parsed_Lead_Status_History" 

# --- Calculation ---
with st.spinner("Analyzing status histories for PC activity..."):
    if parsed_status_history_col not in processed_data.columns:
        st.error(f"The required column '{parsed_status_history_col}' was not found in the processed data.")
        st.info("This column is generated during the initial data processing step from the 'Lead Status History' column. Please ensure the source data contains this column.")
        st.stop()

    contact_heatmap, sts_heatmap = calculate_heatmap_data(processed_data, ts_col_map, parsed_status_history_col)
    time_metrics = calculate_average_time_metrics(processed_data, ts_col_map, parsed_status_history_col)
    top_flows = calculate_top_status_flows(processed_data, ts_col_map, parsed_status_history_col)

# --- Display Heatmaps ---
st.header("Activity Heatmaps")
st.markdown("Visualizing when key activities occur during the week.")
col1, col2 = st.columns(2)
with col1, st.container(border=True):
    st.subheader("Pre-StS Contact Attempts by Time of Day")
    if not contact_heatmap.empty:
        fig1 = px.imshow(contact_heatmap, labels=dict(x="Hour of Day", y="Day of Week", color="Contacts"), aspect="auto", color_continuous_scale="Mint")
        fig1.update_layout(xaxis_nticks=12)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No pre-StS contact attempt data found.")
with col2, st.container(border=True):
    st.subheader("Sent To Site Events by Time of Day")
    if not sts_heatmap.empty:
        fig2 = px.imshow(sts_heatmap, labels=dict(x="Hour of Day", y="Day of Week", color="Events"), aspect="auto", color_continuous_scale="Mint")
        fig2.update_layout(xaxis_nticks=12)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No 'Sent To Site' data found.")

st.divider()

# --- Display KPIs ---
st.header("Key Performance Indicators")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1, st.container(border=True):
    value = time_metrics.get('avg_time_to_first_contact')
    st.metric(label="Average Time to First Contact", value=format_days_to_dhm(value))
with kpi_col2, st.container(border=True):
    value = time_metrics.get('avg_time_between_contacts')
    st.metric(label="Average Time Between Contact Attempts", value=format_days_to_dhm(value))
with kpi_col3, st.container(border=True):
    value = time_metrics.get('avg_time_new_to_sts')
    st.metric(label="Average Time from New to Sent To Site", value=format_days_to_dhm(value))

st.divider()

# --- Display Status Flows ---
st.header("Top 5 Common Status Flows to 'Sent to Site'")
if not top_flows:
    st.info("There is not enough data from referrals that have reached 'Sent to Site' to determine common status flows yet.")
else:
    with st.container(border=True):
        for i, (path, count) in enumerate(top_flows):
            st.markdown(f"**{i+1}. Most Common Path** ({count} referrals)")
            st.info(f"`{path}`")
            if i < len(top_flows) - 1:
                st.divider()