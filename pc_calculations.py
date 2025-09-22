# pc_calculations.py
import pandas as pd
import numpy as np

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
                
                # Check if the event is a contact attempt and occurred before the StS timestamp
                # If StS is NaT, any contact attempt is valid as it hasn't been sent to site yet.
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
        
        # Create a DataFrame with day of week and hour
        agg_df = pd.DataFrame({
            'day_of_week': ts_series.dt.dayofweek, # Monday=0, Sunday=6
            'hour': ts_series.dt.hour
        })
        
        # Create the heatmap grid
        heatmap_grid = pd.crosstab(agg_df['day_of_week'], agg_df['hour'])
        
        # Ensure all days and hours are present
        heatmap_grid = heatmap_grid.reindex(index=range(7), columns=range(24), fill_value=0)
        
        # Map day index to day name
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_grid.index = heatmap_grid.index.map(lambda i: day_names[i])
        
        return heatmap_grid

    contact_heatmap_df = aggregate_timestamps(contact_timestamps)
    sts_heatmap_df = aggregate_timestamps(sts_timestamps)
    
    return contact_heatmap_df, sts_heatmap_df