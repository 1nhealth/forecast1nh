# helpers.py
import streamlit as st
import pandas as pd
import numpy as np
from constants import BUSINESS_HOURS_START, BUSINESS_HOURS_END, BUSINESS_DAYS

def format_performance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Applies standard formatting to a performance DataFrame for display."""
    if df.empty:
        return df

    formatted_df = df.copy()

    if 'Score' in formatted_df.columns:
        formatted_df['Score'] = formatted_df['Score'].round(1)

    for col in formatted_df.columns:
        if '%' in col and pd.api.types.is_numeric_dtype(formatted_df[col]):
            formatted_df[col] = (formatted_df[col] * 100).map('{:,.1f}%'.format).replace('nan%', '-')
        elif ('Lag' in col or 'TTC' in col or 'Steps' in col) and pd.api.types.is_numeric_dtype(formatted_df[col]):
             formatted_df[col] = formatted_df[col].map('{:,.1f}'.format).replace('nan', '-')
        elif ('Count' in col or 'Qualified' in col) and pd.api.types.is_numeric_dtype(formatted_df[col]):
            formatted_df[col] = formatted_df[col].map('{:,.0f}'.format).replace('nan', '-')

    return formatted_df

def load_css(file_name):
    """A function to load a local CSS file into the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Please make sure it is in the project root directory.")

def format_days_to_dhm(days_float):
    """Converts a float number of days into a 'd h m' string format."""
    if pd.isna(days_float) or days_float < 0:
        return "N/A"
    
    # Convert float days to total seconds
    total_seconds = days_float * 24 * 60 * 60
    
    # Calculate days
    days = int(total_seconds // (24 * 3600))
    total_seconds %= (24 * 3600)
    
    # Calculate hours
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    
    # Calculate minutes
    minutes = int(total_seconds // 60)
    
    return f"{days} d {hours} h {minutes} m"

def calculate_business_hours_between(start_dt, end_dt):
    """
    Calculate elapsed time in business hours only between two timestamps.

    Business hours are defined as Monday-Friday, 9:00 AM to 5:00 PM.

    Args:
        start_dt: Start timestamp (pandas Timestamp or datetime)
        end_dt: End timestamp (pandas Timestamp or datetime)

    Returns:
        float: Elapsed time in days (business hours / 24) for consistency with calendar time calculations
               Returns 0 if no business hours elapsed
               Returns NaN if inputs are invalid

    Examples:
        Friday 4pm → Monday 10am = 2 business hours = 0.0833 days
        Friday 5pm → Monday 9am = 0 business hours = 0 days
        Monday 10am → Monday 2pm = 4 business hours = 0.1667 days
    """
    # Handle invalid inputs
    if pd.isna(start_dt) or pd.isna(end_dt):
        return np.nan

    # Convert to pandas Timestamp if needed
    start_dt = pd.Timestamp(start_dt)
    end_dt = pd.Timestamp(end_dt)

    # Handle negative time span
    if end_dt < start_dt:
        return 0

    # Handle same timestamp
    if start_dt == end_dt:
        return 0

    total_business_hours = 0.0

    # Iterate day by day
    current_date = start_dt.date()
    end_date = end_dt.date()

    while current_date <= end_date:
        # Check if current day is a weekday (Monday=0 to Friday=4)
        weekday = pd.Timestamp(current_date).weekday()

        if weekday in BUSINESS_DAYS:
            # Define business hours for this day
            business_start = pd.Timestamp.combine(current_date, pd.Timestamp(f'{BUSINESS_HOURS_START}:00:00').time())
            business_end = pd.Timestamp.combine(current_date, pd.Timestamp(f'{BUSINESS_HOURS_END}:00:00').time())

            # Calculate overlap between [start_dt, end_dt] and [business_start, business_end]
            overlap_start = max(start_dt, business_start)
            overlap_end = min(end_dt, business_end)

            # If there's an overlap, add it to total
            if overlap_end > overlap_start:
                hours_overlap = (overlap_end - overlap_start).total_seconds() / 3600
                total_business_hours += hours_overlap

        # Move to next day
        current_date = (pd.Timestamp(current_date) + pd.Timedelta(days=1)).date()

    # Convert hours to days for consistency with existing code
    return total_business_hours / 24


def calculate_avg_lag_generic(df, col_from, col_to, business_hours_only=False):
    """
    Calculate average time lag between two timestamp columns.

    Args:
        df: DataFrame containing the timestamp columns
        col_from: Name of the start timestamp column
        col_to: Name of the end timestamp column
        business_hours_only: If True, calculate elapsed time in business hours only (Mon-Fri 9am-5pm)
                           If False, calculate calendar time (default)

    Returns:
        float: Average time lag in days, or NaN if no valid data
    """
    if col_from is None or col_to is None or col_from not in df.columns or col_to not in df.columns:
        return np.nan

    if not all([pd.api.types.is_datetime64_any_dtype(df[col_from]),
                pd.api.types.is_datetime64_any_dtype(df[col_to])]):
        return np.nan

    valid_df = df.dropna(subset=[col_from, col_to])
    if valid_df.empty:
        return np.nan

    if business_hours_only:
        # Calculate business hours for each row
        business_hours_diffs = []
        for _, row in valid_df.iterrows():
            start = row[col_from]
            end = row[col_to]

            # Only include positive time differences
            if end >= start:
                bh_diff = calculate_business_hours_between(start, end)
                if not pd.isna(bh_diff) and bh_diff >= 0:
                    business_hours_diffs.append(bh_diff)

        return np.mean(business_hours_diffs) if business_hours_diffs else np.nan
    else:
        # Original calendar time calculation
        diff = pd.to_datetime(valid_df[col_to]) - pd.to_datetime(valid_df[col_from])
        diff_positive = diff[diff >= pd.Timedelta(days=0)]

        return diff_positive.mean().total_seconds() / (60 * 60 * 24) if not diff_positive.empty else np.nan

def is_contact_attempt(status_name):
    """
    Uses a heuristic keyword search to determine if a status name represents a patient contact attempt.
    This is case-insensitive.
    """
    if not isinstance(status_name, str):
        return False

    # Define the keywords that signify a direct patient contact attempt
    CONTACT_KEYWORDS = [
        'contact', 'attempt', 'call', 'called',
        'email', 'emailed', 'text', 'sms', 'message sent',
        'voicemail', 'vm', 'left message', 'lm',
        'follow-up', 'outreach', 'spoke to', 'connected'
    ]

    # Check if any keyword exists in the status name (converted to lower case)
    lower_status = status_name.lower()
    return any(keyword in lower_status for keyword in CONTACT_KEYWORDS)

def is_business_hours(timestamp):
    """
    Determines if a given timestamp falls within business hours.

    Business hours are defined as:
    - Monday through Friday (weekday 0-4)
    - 9:00 AM to 5:00 PM (hours 9-16, since hour 17 is 5:00 PM start)

    Args:
        timestamp: pandas Timestamp or datetime object

    Returns:
        bool: True if timestamp is during business hours, False otherwise
    """
    if pd.isna(timestamp):
        return False

    # Check if it's a business day (Monday=0 to Friday=4)
    if timestamp.weekday() not in BUSINESS_DAYS:
        return False

    # Check if it's within business hours (9 AM to 5 PM)
    # Hour 17 is 5:00 PM, so we want hours 9-16 (9 AM to 4:59 PM)
    if timestamp.hour < BUSINESS_HOURS_START or timestamp.hour >= BUSINESS_HOURS_END:
        return False

    return True

def filter_business_hours_only(df, timestamp_col):
    """
    Filters a DataFrame to only include rows where the timestamp column
    falls within business hours.

    Args:
        df: pandas DataFrame
        timestamp_col: string name of the timestamp column to filter on

    Returns:
        pandas DataFrame: Filtered DataFrame with only business hours rows
    """
    if df is None or df.empty:
        return df

    if timestamp_col not in df.columns:
        return df

    # Apply business hours filter
    mask = df[timestamp_col].apply(is_business_hours)
    return df[mask].copy()