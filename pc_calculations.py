# pc_calculations.py
import pandas as pd
import numpy as np
from collections import Counter
from calculations import calculate_avg_lag_generic

# ... (existing functions calculate_heatmap_data, calculate_average_time_metrics, calculate_top_status_flows remain unchanged) ...

def calculate_ttfc_effectiveness(df, ts_col_map):
    """
    Analyzes how the time to first contact impacts downstream conversion rates.
    """
    if df is None or df.empty or not ts_col_map:
        return pd.DataFrame()

    pof_ts_col = ts_col_map.get("Passed Online Form")
    if pof_ts_col not in df.columns:
        return pd.DataFrame()

    # Get all other timestamp columns that exist in the dataframe
    other_ts_cols = [v for k, v in ts_col_map.items() if k != "Passed Online Form" and v in df.columns]
    if not other_ts_cols:
        return pd.DataFrame()

    analysis_df = df.copy()
    start_ts = analysis_df[pof_ts_col]

    # For each referral, find the earliest timestamp from any other stage
    def find_first_action(row):
        future_events = row[row > start_ts.loc[row.name]]
        return future_events.min()

    first_action_ts = analysis_df[other_ts_cols].apply(find_first_action, axis=1)
    
    # Calculate the Time to First Contact (TTFC) in minutes
    analysis_df['ttfc_minutes'] = (first_action_ts - start_ts).dt.total_seconds() / 60
    
    # Define the time bins
    bin_edges = [-np.inf, 5, 15, 30, 60, 3*60, 6*60, 12*60, 24*60, np.inf]
    bin_labels = [
        '<= 5 min', '5-15 min', '15-30 min', '30-60 min', 
        '1-3 hours', '3-6 hours', '6-12 hours', '12-24 hours', '> 24 hours'
    ]
    
    analysis_df['ttfc_bin'] = pd.cut(
        analysis_df['ttfc_minutes'], 
        bins=bin_edges, 
        labels=bin_labels, 
        right=True
    )

    # Get downstream columns for aggregation
    sts_col = ts_col_map.get("Sent To Site")
    icf_col = ts_col_map.get("Signed ICF")
    enr_col = ts_col_map.get("Enrolled")

    # Aggregate results by the time bins
    result = analysis_df.groupby('ttfc_bin').agg(
        Attempts=('ttfc_bin', 'size'),
        Total_StS=(sts_col, lambda x: x.notna().sum()),
        Total_ICF=(icf_col, lambda x: x.notna().sum()),
        Total_Enrolled=(enr_col, lambda x: x.notna().sum())
    )
    
    # Ensure all bins are present in the final table, even if they have 0 attempts
    result = result.reindex(bin_labels, fill_value=0)

    # Calculate rates safely, avoiding division by zero
    result['StS_Rate'] = (result['Total_StS'] / result['Attempts'].replace(0, np.nan))
    result['ICF_Rate'] = (result['Total_ICF'] / result['Attempts'].replace(0, np.nan))
    result['Enrollment_Rate'] = (result['Total_Enrolled'] / result['Attempts'].replace(0, np.nan))
    
    result.reset_index(inplace=True)
    result.rename(columns={'ttfc_bin': 'Time to First Contact'}, inplace=True)
    
    return result