# pages/8_PC_Performance.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, time

# Import all calculation functions and the formatting helper
from pc_calculations import (
    calculate_heatmap_data, 
    calculate_average_time_metrics, 
    calculate_top_status_flows,
    calculate_ttfc_effectiveness,
    calculate_contact_attempt_effectiveness
)
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

# --- NEW: Date Filter ---
st.divider()
submission_date_col = "Submitted On_DT" 

if submission_date_col in processed_data.columns:
    # Get the overall date range from the data to set as filter defaults
    min_date = processed_data[submission_date_col].min().date()
    max_date = processed_data[submission_date_col].max().date()

    # Create the date input widgets in a container
    with st.container(border=True):
        st.subheader("Filter Data by Submission Date")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
        st.stop()

    # Convert dates to datetimes to properly filter the timestamp column
    # Combine end_date with max time to ensure the entire day is included
    start_datetime = datetime.combine(start_date, time.min)
    end_datetime = datetime.combine(end_date, time.max)
    
    # Create the filtered DataFrame based on the selected date range
    filtered_df = processed_data[
        (processed_data[submission_date_col] >= start_datetime) &
        (processed_data[submission_date_col] <= end_datetime)
    ].copy()

    st.metric(label="Total Referrals in Selected Range", value=f"{len(filtered_df):,}")

else:
    st.warning(f"Date column '{submission_date_col}' not found. Cannot apply date filter.")
    # If the date column is missing, use the original unfiltered dataframe to prevent the app from crashing
    filtered_df = processed_data

st.divider()
# --- End of Date Filter ---

# --- Calculation ---
# All calculations now use the 'filtered_df' which is controlled by the date filter.
with st.spinner("Analyzing status histories for PC activity in selected date range..."):
    if parsed_status_history_col not in filtered_df.columns:
        st.error(f"The required column '{parsed_status_history_col}' was not found in the processed data.")
        st.stop()

    contact_heatmap, sts_heatmap = calculate_heatmap_data(filtered_df, ts_col_map, parsed_status_history_col)
    time_metrics = calculate_average_time_metrics(filtered_df, ts_col_map, parsed_status_history_col)
    top_flows = calculate_top_status_flows(filtered_df, ts_col_map, parsed_status_history_col)
    ttfc_df = calculate_ttfc_effectiveness(filtered_df, ts_col_map)
    attempt_effectiveness_df = calculate_contact_attempt_effectiveness(filtered_df, ts_col_map, parsed_status_history_col)

# --- Display Heatmaps ---
st.header("Activity Heatmaps")
st.markdown("Visualizing when key activities occur during the week.")
col1, col2 = st.columns(2)
with col1, st.container(border=True):
    st.subheader("Pre-StS Contact Attempts by Time of Day")
    if not contact_heatmap.empty and contact_heatmap.values.sum() > 0:
        fig1 = px.imshow(contact_heatmap, labels=dict(x="Hour of Day", y="Day of Week", color="Contacts"), aspect="auto", color_continuous_scale="Mint")
        fig1.update_layout(xaxis_nticks=12)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No pre-StS contact attempt data found in the selected date range.")

with col2, st.container(border=True):
    st.subheader("Sent To Site Events by Time of Day")
    if not sts_heatmap.empty and sts_heatmap.values.sum() > 0:
        fig2 = px.imshow(sts_heatmap, labels=dict(x="Hour of Day", y="Day of Week", color="Events"), aspect="auto", color_continuous_scale="Mint")
        fig2.update_layout(xaxis_nticks=12)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No 'Sent To Site' data found in the selected date range.")

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
    st.info("There is not enough data in the selected date range to determine common status flows.")
else:
    with st.container(border=True):
        for i, (path, count) in enumerate(top_flows):
            st.markdown(f"**{i+1}. Most Common Path** ({count} referrals)")
            st.info(f"`{path}`")
            if i < len(top_flows) - 1:
                st.divider()

st.divider()

# --- TTFC Effectiveness ---
st.header("Time to First Contact Effectiveness")
st.markdown("Analyzes how the speed of the first contact impacts downstream funnel conversions.")

if ttfc_df.empty or ttfc_df['Attempts'].sum() == 0:
    st.info("Not enough data in the selected date range to analyze the effectiveness of first contact timing.")
else:
    display_df = ttfc_df.copy()
    display_df['StS Rate'] = display_df['StS_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df['ICF Rate'] = display_df['ICF_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df['Enrollment Rate'] = display_df['Enrollment_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df.rename(columns={'Total_StS': 'Total Sent to Site', 'Total_ICF': 'Total ICFs', 'Total_Enrolled': 'Total Enrollments'}, inplace=True)
    final_cols = ['Time to First Contact', 'Attempts', 'Total Sent to Site', 'StS Rate', 'Total ICFs', 'ICF Rate', 'Total Enrollments', 'Enrollment Rate']
    with st.container(border=True):
        st.dataframe(display_df[final_cols], hide_index=True, use_container_width=True)

st.divider()

# --- Contact Attempt Effectiveness ---
st.header("Contact Attempt Effectiveness")
st.markdown("Analyzes how the number of pre-site status changes impacts downstream funnel conversions.")

if (attempt_effectiveness_df.empty or 
    'Total Referrals' not in attempt_effectiveness_df.columns or 
    attempt_effectiveness_df['Total Referrals'].sum() == 0):
    st.info("Not enough data in the selected date range to analyze the effectiveness of contact attempts.")
else:
    display_df = attempt_effectiveness_df.copy()
    display_df['StS Rate'] = display_df['StS_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df['ICF Rate'] = display_df['ICF_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df['Enrollment Rate'] = display_df['Enrollment_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df.rename(columns={'Total_StS': 'Total Sent to Site', 'Total_ICF': 'Total ICFs', 'Total_Enrolled': 'Total Enrollments'}, inplace=True)
    final_cols = ['Number of Attempts', 'Total Referrals', 'Total Sent to Site', 'StS Rate', 'Total ICFs', 'ICF Rate', 'Total Enrollments', 'Enrollment Rate']
    with st.container(border=True):
        st.dataframe(display_df[final_cols], hide_index=True, use_container_width=True)