# pages/8_PC_Performance.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

from pc_calculations import (
    calculate_heatmap_data, 
    calculate_average_time_metrics, 
    calculate_top_status_flows,
    calculate_ttfc_effectiveness,
    calculate_contact_attempt_effectiveness,
    calculate_performance_over_time,
    analyze_heatmap_efficiency # Import new function
)
from helpers import format_days_to_dhm, load_css

st.set_page_config(page_title="PC Performance", page_icon="📞", layout="wide")

# Load custom CSS for branded theme
load_css("custom_theme.css")

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📞 PC Performance Dashboard")
st.info("This dashboard analyzes the operational efficiency and patterns of the Pre-Screening team's activities.")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

processed_data = st.session_state.referral_data_processed
ts_col_map = st.session_state.ts_col_map
parsed_status_history_col = "Parsed_Lead_Status_History" 

st.divider()
submission_date_col = "Submitted On_DT" 
if submission_date_col in processed_data.columns:
    min_date = processed_data[submission_date_col].min().date()
    max_date = processed_data[submission_date_col].max().date()
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
    start_datetime = datetime.combine(start_date, time.min)
    end_datetime = datetime.combine(end_date, time.max)
    filtered_df = processed_data[
        (processed_data[submission_date_col] >= start_datetime) &
        (processed_data[submission_date_col] <= end_datetime)
    ].copy()
    st.metric(label="Total Referrals in Selected Range", value=f"{len(filtered_df):,}")
else:
    st.warning(f"Date column '{submission_date_col}' not found. Cannot apply date filter.")
    filtered_df = processed_data
st.divider()

# Business Hours Toggle
with st.container(border=True):
    st.subheader("⏰ Time Metrics View Option")
    business_hours_only = st.radio(
        "Select time calculation method:",
        options=[False, True],
        format_func=lambda x: "All Hours (24/7)" if not x else "Business Hours Only (Mon-Fri, 9am-5pm)",
        horizontal=True,
        help="Choose whether to calculate time metrics using all hours or only business hours (Mon-Fri, 9am-5pm). Business hours mode excludes weekends and after-hours activity."
    )

st.divider()

with st.spinner("Analyzing status histories for PC activity in selected date range..."):
    if parsed_status_history_col not in filtered_df.columns:
        st.error(f"The required column '{parsed_status_history_col}' was not found in the processed data.")
        st.stop()

    contact_heatmap, sts_heatmap = calculate_heatmap_data(filtered_df, ts_col_map, parsed_status_history_col, business_hours_only=business_hours_only)
    time_metrics = calculate_average_time_metrics(filtered_df, ts_col_map, parsed_status_history_col, business_hours_only=business_hours_only)
    top_flows = calculate_top_status_flows(filtered_df, ts_col_map, parsed_status_history_col)
    ttfc_df = calculate_ttfc_effectiveness(filtered_df, ts_col_map, business_hours_only=business_hours_only)
    attempt_effectiveness_df = calculate_contact_attempt_effectiveness(filtered_df, ts_col_map, parsed_status_history_col, business_hours_only=business_hours_only)
    over_time_df = calculate_performance_over_time(filtered_df, ts_col_map)
    # --- NEW: Call the analysis function ---
    heatmap_insights = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap, business_hours_only=business_hours_only)

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

# --- NEW SECTION: Display Call Timing Recommendations ---
st.header("Call Timing Recommendations")
st.markdown("**Optimize your calling strategy** based on historical success rates (Sent to Site conversions).")

if not heatmap_insights:
    st.info("Not enough data to generate call timing recommendations.")
else:
    # Group recommendations by day
    from collections import defaultdict

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    recommendations_by_day = defaultdict(lambda: {'best': [], 'avoid': []})

    # Parse best times
    for time_str in heatmap_insights.get("best_times", []):
        if ', ' in time_str:
            day, time = time_str.split(', ', 1)
            recommendations_by_day[day]['best'].append(time)

    # Parse avoid times
    for time_str in heatmap_insights.get("avoid_times", []):
        if ', ' in time_str:
            day, time = time_str.split(', ', 1)
            recommendations_by_day[day]['avoid'].append(time)

    # Get days that have recommendations
    days_with_recommendations = [day for day in days_order
                                  if recommendations_by_day[day]['best'] or recommendations_by_day[day]['avoid']]

    # Display recommendations grouped by day - 2 cards per row
    for i in range(0, len(days_with_recommendations), 2):
        # Create a row with 2 columns
        row_cols = st.columns(2)

        # First card in the row
        day1 = days_with_recommendations[i]
        with row_cols[0]:
            with st.container(border=True):
                st.markdown(f"### {day1}")

                # Create two columns within the card - Best on left, Avoid on right
                day_cols = st.columns(2)

                # Best times on the left
                with day_cols[0]:
                    st.markdown("**:green[✓ Best Times]**")
                    if recommendations_by_day[day1]['best']:
                        for time in recommendations_by_day[day1]['best']:
                            st.markdown(f":green[●] {time}")
                    else:
                        st.caption("No standout times")

                # Avoid times on the right
                with day_cols[1]:
                    st.markdown("**:red[✗ Avoid]**")
                    if recommendations_by_day[day1]['avoid']:
                        for time in recommendations_by_day[day1]['avoid']:
                            st.markdown(f":red[●] {time}")
                    else:
                        st.caption("No times to avoid")

        # Second card in the row (if it exists)
        if i + 1 < len(days_with_recommendations):
            day2 = days_with_recommendations[i + 1]
            with row_cols[1]:
                with st.container(border=True):
                    st.markdown(f"### {day2}")

                    # Create two columns within the card - Best on left, Avoid on right
                    day_cols = st.columns(2)

                    # Best times on the left
                    with day_cols[0]:
                        st.markdown("**:green[✓ Best Times]**")
                        if recommendations_by_day[day2]['best']:
                            for time in recommendations_by_day[day2]['best']:
                                st.markdown(f":green[●] {time}")
                        else:
                            st.caption("No standout times")

                    # Avoid times on the right
                    with day_cols[1]:
                        st.markdown("**:red[✗ Avoid]**")
                        if recommendations_by_day[day2]['avoid']:
                            for time in recommendations_by_day[day2]['avoid']:
                                st.markdown(f":red[●] {time}")
                        else:
                            st.caption("No times to avoid")

st.divider()

# Show which mode is active
mode_label = ":blue[Business Hours Only (Mon-Fri, 9am-5pm)]" if business_hours_only else ":gray[All Hours (24/7)]"
st.header(f"Key Performance Indicators - {mode_label}")
st.caption("These metrics reflect the selected time calculation method above.")

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

st.divider()

st.header("Performance Over Time (Weekly)")
st.markdown("Track key metrics on a weekly basis to identify trends.")

if over_time_df.empty:
    st.info("Not enough data in the selected date range to generate a performance trend graph.")
else:
    with st.container(border=True):
        secondary_metric = 'Total Qualified per Week'
        
        primary_metric_options = [col for col in over_time_df.columns if col != secondary_metric]
        primary_metric = st.selectbox(
            "Select a primary metric to display on the chart:",
            options=primary_metric_options
        )
        
        compare_with_volume = st.toggle(f"Compare with {secondary_metric}", value=True)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=over_time_df.index, y=over_time_df[primary_metric], name=primary_metric, 
                       line=dict(color='#53CA97')),
            secondary_y=False,
        )

        if compare_with_volume:
            fig.add_trace(
                go.Scatter(x=over_time_df.index, y=over_time_df[secondary_metric], name=secondary_metric, line=dict(dash='dot', color='gray')),
                secondary_y=True,
            )

        fig.update_yaxes(title_text=f"<b>{primary_metric}</b>", secondary_y=False)
        if compare_with_volume:
            fig.update_yaxes(title_text=f"<b>{secondary_metric}</b>", secondary_y=True, showgrid=False)

        fig.update_layout(
            title_text=f"Weekly Trend: {primary_metric}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)