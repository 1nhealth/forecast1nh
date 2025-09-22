# pc_calculations.py
import pandas as pd
import numpy as np
# Import the generic lag calculator for reuse
from calculations import calculate_avg_lag_generic

def calculate_heatmap_data(df, ts_col_map, status_history_col):
    """
    Prepares data for two heatmaps:
    1. Pre-Sent-to-Site Contact Attempts by day of week and hour.
    2. Sent to Site events by day of week and hour.
    """
    if df is None or df.empty or ts_col_map is None:
        return pd.DataFrame(), pd.DataFrame()

    # --- Heatmap 1: Pre-StS Contact Attempts ---
    contact_timestamps = []
    sts_ts_col = ts_col_map.get("Sent To Site")

    if status_history_col in df.columns and sts_ts_col in df.columns:
        for _, row in df.iterrows():
            sts_timestamp = row[sts_ts_col]
            history = row[status_history_col]

            if not isinstance(history, list):
                continue

            for event_name, event_dt in history:
                is_contact_attempt = "contact attempt" in event_name.lower()
                
                if is_contact_attempt and (pd.isna(sts_timestamp) or event_dt < sts_timestamp):
                    contact_timestamps.append(event_dt)
    
    # --- Heatmap 2: Sent to Site Events ---
    sts_timestamps = df[sts_ts_col].dropna().tolist()

    # --- Aggregation Helper Function ---
    def aggregate_timestamps(timestamps):
        if not timestamps:
            return pd.DataFrame(np.zeros((7, 24)), 
                                index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                                columns=range(24))
        
        ts_series = pd.Series(pd.to_datetime(timestamps))
        
        agg_df = pd.DataFrame({
            'day_of_week': ts_series.dt.dayofweek, # Monday=0, Sunday=6
            'hour': ts_series.dt.hour
        })
        
        heatmap_grid = pd.crosstab(agg_df['day_of_week'], agg_df['hour'])
        heatmap_grid = heatmap_grid.reindex(index=range(7), columns=range(24), fill_value=0)
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_grid.index = heatmap_grid.index.map(lambda i: day_names[i])
        
        return heatmap_grid

    contact_heatmap_df = aggregate_timestamps(contact_timestamps)
    sts_heatmap_df = aggregate_timestamps(sts_timestamps)
    
    return contact_heatmap_df, sts_heatmap_df

def calculate_average_time_metrics(df, ts_col_map, status_history_col):
    """
    Calculates key average time metrics for PC performance.
    """
    if df is None or df.empty or ts_col_map is None:
        return {"avg_time_to_first_contact": np.nan, 
                "avg_time_between_contacts": np.nan, 
                "avg_time_new_to_sts": np.nan}

    # --- Metric 1: Average Time to First Contact (New -> Pre-Screening Activities) ---
    pof_ts_col = ts_col_map.get("Passed Online Form")
    psa_ts_col = ts_col_map.get("Pre-Screening Activities")
    avg_ttfc = calculate_avg_lag_generic(df, pof_ts_col, psa_ts_col)

    # --- Metric 2: Average Time New -> StS ---
    sts_ts_col = ts_col_map.get("Sent To Site")
    avg_new_to_sts = calculate_avg_lag_generic(df, pof_ts_col, sts_ts_col)
    
    # --- Metric 3: Average Time Between Pre-StS Contact Attempts ---
    all_contact_deltas = []
    if status_history_col in df.columns and sts_ts_col in df.columns:
        for _, row in df.iterrows():
            sts_timestamp = row[sts_ts_col]
            history = row[status_history_col]
            
            if not isinstance(history, list):
                continue

            # Get all pre-StS contact attempt timestamps for this referral
            attempt_timestamps = [
                event_dt for event_name, event_dt in history 
                if "contact attempt" in event_name.lower() 
                and (pd.isna(sts_timestamp) or event_dt < sts_timestamp)
            ]
            
            # If there's more than one attempt, calculate the deltas
            if len(attempt_timestamps) > 1:
                attempt_timestamps.sort()
                deltas = np.diff(attempt_timestamps)
                all_contact_deltas.extend(deltas)

    if all_contact_deltas:
        # Calculate the mean of all deltas and convert from timedelta to days
        avg_between_contacts = pd.Series(all_contact_deltas).mean().total_seconds() / (60 * 60 * 24)
    else:
        avg_between_contacts = np.nan

    return {
        "avg_time_to_first_contact": avg_ttfc,
        "avg_time_between_contacts": avg_between_contacts,
        "avg_time_new_to_sts": avg_new_to_sts
    }