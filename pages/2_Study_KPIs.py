# pages/2_Study_KPIs.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from helpers import load_css
from constants import (
    STAGE_PASSED_ONLINE_FORM,
    STAGE_SENT_TO_SITE,
    STAGE_APPOINTMENT_SCHEDULED,
    STAGE_SIGNED_ICF,
    STAGE_ENROLLED
)

# --- Page Configuration ---
st.set_page_config(page_title="Study KPIs", page_icon="📈", layout="wide")

# Load custom CSS for branded theme
load_css("custom_theme.css")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📈 Study KPIs")

# --- KPI Mappings ---
KPI_OPTIONS = {
    "Qualified Referrals": STAGE_PASSED_ONLINE_FORM,
    "Sent to Sites": STAGE_SENT_TO_SITE,
    "Appointments": STAGE_APPOINTMENT_SCHEDULED,
    "Signed ICFs": STAGE_SIGNED_ICF,
    "Enrollments": STAGE_ENROLLED
}

RESAMPLE_RULES = {
    "Day": "D",
    "Week": "W",
    "Month": "ME"
}


# --- Helper Functions ---
def calculate_kpi_over_time(df, ts_col_map, stage_constant, resample_rule, start_date, end_date):
    """
    Calculate KPI counts aggregated by the specified time period.

    Args:
        df: The processed referral DataFrame
        ts_col_map: Mapping of stage names to timestamp column names
        stage_constant: The stage constant (e.g., STAGE_PASSED_ONLINE_FORM)
        resample_rule: Pandas resample rule ('D', 'W', 'ME')
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        DataFrame with columns ['Period', 'Count']
    """
    ts_col = ts_col_map.get(stage_constant)

    if ts_col is None or ts_col not in df.columns:
        return pd.DataFrame(columns=['Period', 'Count'])

    # Filter to rows that have reached this stage
    stage_df = df.dropna(subset=[ts_col]).copy()

    # Convert dates to datetime for comparison
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Filter by date range
    stage_df = stage_df[(stage_df[ts_col] >= start_dt) & (stage_df[ts_col] <= end_dt)]

    if stage_df.empty:
        return pd.DataFrame(columns=['Period', 'Count'])

    # Set the timestamp as index and resample
    stage_df = stage_df.set_index(ts_col)

    # Count referrals per period
    kpi_series = stage_df.resample(resample_rule).size()

    # Convert to DataFrame
    result_df = kpi_series.reset_index()
    result_df.columns = ['Period', 'Count']

    return result_df


def create_kpi_chart(data_df, kpi_name, chart_type, time_horizon):
    """
    Create a Plotly chart (line or bar) for the KPI data.

    Args:
        data_df: DataFrame with 'Period' and 'Count' columns
        kpi_name: Name of the KPI for chart title
        chart_type: 'Line' or 'Bar'
        time_horizon: 'Day', 'Week', or 'Month' for axis labeling

    Returns:
        Plotly figure
    """
    if data_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected criteria",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=400)
        return fig

    # Brand colors
    brand_green = '#53CA97'

    if chart_type == "Line":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_df['Period'],
            y=data_df['Count'],
            mode='lines+markers',
            name=kpi_name,
            line=dict(color=brand_green, width=2),
            marker=dict(size=8),
            hovertemplate='%{x|%Y-%m-%d}<br>Count: %{y}<extra></extra>'
        ))
    else:  # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data_df['Period'],
            y=data_df['Count'],
            name=kpi_name,
            marker_color=brand_green,
            text=data_df['Count'],
            textposition='auto',
            hovertemplate='%{x|%Y-%m-%d}<br>Count: %{y}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=f"{kpi_name} Over Time (by {time_horizon})",
        xaxis_title=time_horizon,
        yaxis_title="Count",
        height=500,
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_color='black',
            bordercolor='#cccccc'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Format x-axis based on time horizon
    if time_horizon == "Day":
        fig.update_xaxes(tickformat="%Y-%m-%d")
    elif time_horizon == "Week":
        fig.update_xaxes(tickformat="%Y-%m-%d")
    else:  # Month
        fig.update_xaxes(tickformat="%Y-%m")

    return fig


# --- Data Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Get Data ---
df = st.session_state.referral_data_processed
ts_col_map = st.session_state.ts_col_map

# Determine date range from data
all_dates = []
for stage, ts_col in ts_col_map.items():
    if ts_col in df.columns:
        valid_dates = df[ts_col].dropna()
        if not valid_dates.empty:
            all_dates.extend(valid_dates.tolist())

if all_dates:
    min_date = pd.Timestamp(min(all_dates)).date()
    max_date = pd.Timestamp(max(all_dates)).date()
else:
    min_date = df["Submitted On_DT"].min().date()
    max_date = df["Submitted On_DT"].max().date()

# --- Controls Section ---
with st.container(border=True):
    st.subheader("KPI Selection & Filters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_kpi = st.selectbox(
            "Select KPI",
            options=list(KPI_OPTIONS.keys()),
            key="study_kpi_selector"
        )

    with col2:
        time_horizon = st.selectbox(
            "Time Horizon",
            options=["Day", "Week", "Month"],
            index=2,  # Default to Month
            key="study_kpi_time_horizon"
        )

    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="study_kpi_date_range"
        )

    with col4:
        chart_type = st.radio(
            "Chart Type",
            options=["Line", "Bar"],
            horizontal=True,
            key="study_kpi_chart_type"
        )

# --- Main Visualization ---
st.divider()

# Validate date range input
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range

    # Calculate KPI data
    resample_rule = RESAMPLE_RULES[time_horizon]
    stage_constant = KPI_OPTIONS[selected_kpi]

    kpi_data = calculate_kpi_over_time(
        df=df,
        ts_col_map=ts_col_map,
        stage_constant=stage_constant,
        resample_rule=resample_rule,
        start_date=start_date,
        end_date=end_date
    )

    # Display summary metrics
    with st.container(border=True):
        metric_cols = st.columns(3)

        total_count = kpi_data['Count'].sum() if not kpi_data.empty else 0
        avg_count = kpi_data['Count'].mean() if not kpi_data.empty else 0
        period_count = len(kpi_data) if not kpi_data.empty else 0

        metric_cols[0].metric(
            label=f"Total {selected_kpi}",
            value=f"{total_count:,}"
        )
        metric_cols[1].metric(
            label=f"Avg per {time_horizon}",
            value=f"{avg_count:,.1f}"
        )
        metric_cols[2].metric(
            label=f"Number of {time_horizon}s",
            value=f"{period_count}"
        )

    # Create and display chart
    with st.container(border=True):
        fig = create_kpi_chart(kpi_data, selected_kpi, chart_type, time_horizon)
        st.plotly_chart(fig, use_container_width=True)

    # Data table section
    with st.container(border=True):
        st.subheader("Data Breakdown")

        if not kpi_data.empty:
            # Format the display dataframe
            display_df = kpi_data.copy()
            display_df['Period'] = display_df['Period'].dt.strftime('%Y-%m-%d')

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Download buttons
            col1, col2, _ = st.columns([1, 1, 4])

            with col1:
                csv_data = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f'study_kpis_{selected_kpi.replace(" ", "_")}_{time_horizon}.csv',
                    mime='text/csv',
                    key='download_csv_study_kpis'
                )

            with col2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    display_df.to_excel(writer, sheet_name='KPI Data', index=False)
                excel_data = output.getvalue()

                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f'study_kpis_{selected_kpi.replace(" ", "_")}_{time_horizon}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download_excel_study_kpis'
                )
        else:
            st.info("No data to display for the selected criteria.")
else:
    st.warning("Please select a valid date range (start and end dates).")
