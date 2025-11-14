# pc_calculations.py
import pandas as pd
import numpy as np
from collections import Counter

# Import helper functions
from helpers import calculate_avg_lag_generic, is_business_hours

def calculate_heatmap_data(df, ts_col_map, status_history_col, business_hours_only=False):
    """
    Prepares data for two heatmaps:
    1. All Pre-Sent-to-Site Actions (ANY status change) by day of week and hour.
    2. Sent to Site events by day of week and hour.

    Args:
        df: DataFrame with referral data
        ts_col_map: Dictionary mapping stage names to timestamp columns
        status_history_col: Name of column containing parsed status history
        business_hours_only: If True, only include events during business hours (Mon-Fri 9am-5pm)
    """
    if df is None or df.empty or ts_col_map is None:
        return pd.DataFrame(), pd.DataFrame()

    action_timestamps = []
    psa_ts_col = ts_col_map.get("Pre-Screening Activities")
    sts_ts_col = ts_col_map.get("Sent To Site")

    if all(col in df.columns for col in [psa_ts_col, sts_ts_col, status_history_col]):
        for _, row in df.iterrows():
            psa_timestamp = row[psa_ts_col]
            sts_timestamp = row[sts_ts_col]
            history = row[status_history_col]

            if pd.isna(psa_timestamp) or not isinstance(history, list):
                continue

            end_window = sts_timestamp if pd.notna(sts_timestamp) else pd.Timestamp.max

            for _, event_dt in history:
                if psa_timestamp <= event_dt < end_window:
                    # Filter for business hours if requested
                    if business_hours_only:
                        if is_business_hours(event_dt):
                            action_timestamps.append(event_dt)
                    else:
                        action_timestamps.append(event_dt)

    # Get StS timestamps and filter for business hours if requested
    sts_timestamps = df[sts_ts_col].dropna().tolist()
    if business_hours_only:
        sts_timestamps = [ts for ts in sts_timestamps if is_business_hours(ts)]

    def aggregate_timestamps(timestamps):
        if not timestamps:
            return pd.DataFrame(np.zeros((7, 24)), 
                                index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                                columns=range(24))
        ts_series = pd.Series(pd.to_datetime(timestamps))
        agg_df = pd.DataFrame({'day_of_week': ts_series.dt.dayofweek, 'hour': ts_series.dt.hour})
        heatmap_grid = pd.crosstab(agg_df['day_of_week'], agg_df['hour'])
        heatmap_grid = heatmap_grid.reindex(index=range(7), columns=range(24), fill_value=0)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_grid.index = heatmap_grid.index.map(lambda i: day_names[i])
        return heatmap_grid
        
    return aggregate_timestamps(action_timestamps), aggregate_timestamps(sts_timestamps)

def calculate_average_time_metrics(df, ts_col_map, status_history_col, business_hours_only=False):
    """
    Calculates key average time metrics for PC performance.
    "Time Between Contacts" now means time between ANY pre-screening actions.

    Args:
        df: DataFrame with referral data
        ts_col_map: Dictionary mapping stage names to timestamp columns
        status_history_col: Name of column containing parsed status history
        business_hours_only: If True, only include events during business hours (Mon-Fri 9am-5pm)

    Returns:
        dict: {
            "avg_time_to_first_contact": float,
            "avg_time_between_contacts": float,
            "avg_time_new_to_sts": float
        }
    """
    if df is None or df.empty or ts_col_map is None:
        return {
            "avg_time_to_first_contact": np.nan,
            "avg_time_between_contacts": np.nan,
            "avg_time_new_to_sts": np.nan
        }

    pof_ts_col = ts_col_map.get("Passed Online Form")
    psa_ts_col = ts_col_map.get("Pre-Screening Activities")
    sts_ts_col = ts_col_map.get("Sent To Site")

    # Filter to business hours if requested
    if business_hours_only:
        # Filter rows where key timestamps are within business hours
        df_filtered = df.copy()

        # Create business hours mask for each timestamp column
        for col in [pof_ts_col, psa_ts_col, sts_ts_col]:
            if col and col in df_filtered.columns:
                # Mark rows where this timestamp is NOT in business hours
                non_bh_mask = ~df_filtered[col].apply(is_business_hours)
                # Set non-business hours timestamps to NaT
                df_filtered.loc[non_bh_mask, col] = pd.NaT

        df = df_filtered

    # Pass business_hours_only parameter to helper function
    avg_ttfc = calculate_avg_lag_generic(df, pof_ts_col, psa_ts_col, business_hours_only=business_hours_only)
    avg_new_to_sts = calculate_avg_lag_generic(df, pof_ts_col, sts_ts_col, business_hours_only=business_hours_only)

    all_action_deltas = []
    if status_history_col in df.columns and psa_ts_col in df.columns:
        for _, row in df.iterrows():
            psa_timestamp = row[psa_ts_col]
            sts_timestamp = row[sts_ts_col]
            history = row[status_history_col]

            if pd.isna(psa_timestamp) or not isinstance(history, list):
                continue

            end_window = sts_timestamp if pd.notna(sts_timestamp) else pd.Timestamp.max

            # Filter history events to business hours if requested
            if business_hours_only:
                action_timestamps = sorted([
                    event_dt for _, event_dt in history
                    if psa_timestamp <= event_dt < end_window and is_business_hours(event_dt)
                ])
            else:
                action_timestamps = sorted([
                    event_dt for _, event_dt in history
                    if psa_timestamp <= event_dt < end_window
                ])

            all_timestamps = action_timestamps

            # Calculate time between consecutive actions
            if len(all_timestamps) > 1:
                if business_hours_only:
                    # Calculate business hours between each pair of consecutive actions
                    from helpers import calculate_business_hours_between
                    for i in range(len(all_timestamps) - 1):
                        bh_diff = calculate_business_hours_between(all_timestamps[i], all_timestamps[i + 1])
                        if not pd.isna(bh_diff) and bh_diff >= 0:
                            # Convert from days to timedelta for consistency with np.diff
                            all_action_deltas.append(pd.Timedelta(days=bh_diff))
                else:
                    # Calendar time calculation (original behavior)
                    all_action_deltas.extend(np.diff(all_timestamps))

    avg_between_actions = pd.Series(all_action_deltas).mean().total_seconds() / (60 * 60 * 24) if all_action_deltas else np.nan

    return {
        "avg_time_to_first_contact": avg_ttfc,
        "avg_time_between_contacts": avg_between_actions,
        "avg_time_new_to_sts": avg_new_to_sts
    }

def calculate_top_status_flows(df, ts_col_map, status_history_col, min_data_threshold=5):
    """
    Identifies the top 5 most common status flow paths leading to Sent To Site.
    """
    if df is None or df.empty or ts_col_map is None: return []
    sts_ts_col = ts_col_map.get("Sent To Site")
    if sts_ts_col not in df.columns or status_history_col not in df.columns: return []
    successful_referrals = df.dropna(subset=[sts_ts_col]).copy()
    if len(successful_referrals) < min_data_threshold: return []
    all_paths = []
    for _, row in successful_referrals.iterrows():
        sts_timestamp = row[sts_ts_col]
        history = row[status_history_col]
        if not isinstance(history, list): continue
        pre_sts_path = [event_name for event_name, event_dt in history if event_dt <= sts_timestamp]
        if pre_sts_path:
            all_paths.append(" -> ".join(pre_sts_path))
    if not all_paths: return []
    return Counter(all_paths).most_common(5)

def calculate_ttfc_effectiveness(df, ts_col_map, business_hours_only=False):
    """
    Analyzes how the time to first contact impacts downstream conversion rates.

    Args:
        df: DataFrame with referral data
        ts_col_map: Dictionary mapping stage names to timestamp columns
        business_hours_only: If True, only include referrals where first contact was during business hours
    """
    if df is None or df.empty or not ts_col_map: return pd.DataFrame()
    pof_ts_col = ts_col_map.get("Passed Online Form")
    if pof_ts_col not in df.columns: return pd.DataFrame()
    other_ts_cols = [v for k, v in ts_col_map.items() if k != "Passed Online Form" and v in df.columns]
    if not other_ts_cols: return pd.DataFrame()
    analysis_df = df.copy()
    start_ts = analysis_df[pof_ts_col]
    def find_first_action(row):
        future_events = row[row > start_ts.loc[row.name]]
        return future_events.min() if not future_events.empty else pd.NaT
    first_action_ts = analysis_df[other_ts_cols].apply(find_first_action, axis=1)

    # Filter for business hours if requested
    if business_hours_only:
        business_hours_mask = first_action_ts.apply(is_business_hours)
        analysis_df = analysis_df[business_hours_mask].copy()
        first_action_ts = first_action_ts[business_hours_mask]
        start_ts = start_ts[business_hours_mask]

    analysis_df['ttfc_minutes'] = (first_action_ts - start_ts).dt.total_seconds() / 60
    bin_edges = [-np.inf, 5, 15, 30, 60, 3*60, 6*60, 12*60, 24*60, np.inf]
    bin_labels = ['<= 5 min', '5-15 min', '15-30 min', '30-60 min', '1-3 hours', '3-6 hours', '6-12 hours', '12-24 hours', '> 24 hours']
    analysis_df['ttfc_bin'] = pd.cut(analysis_df['ttfc_minutes'], bins=bin_edges, labels=bin_labels, right=True)
    sts_col = ts_col_map.get("Sent To Site")
    icf_col = ts_col_map.get("Signed ICF")
    enr_col = ts_col_map.get("Enrolled")
    result = analysis_df.groupby('ttfc_bin').agg(
        Attempts=('ttfc_bin', 'size'),
        Total_StS=(sts_col, lambda x: x.notna().sum()),
        Total_ICF=(icf_col, lambda x: x.notna().sum()),
        Total_Enrolled=(enr_col, lambda x: x.notna().sum())
    )
    result = result.reindex(bin_labels, fill_value=0).astype(int)
    result['StS_Rate'] = (result['Total_StS'] / result['Attempts'].replace(0, np.nan))
    result['ICF_Rate'] = (result['Total_ICF'] / result['Attempts'].replace(0, np.nan))
    result['Enrollment_Rate'] = (result['Total_Enrolled'] / result['Attempts'].replace(0, np.nan))
    result.reset_index(inplace=True)
    result.rename(columns={'ttfc_bin': 'Time to First Contact'}, inplace=True)
    return result

def calculate_contact_attempt_effectiveness(df, ts_col_map, status_history_col, business_hours_only=False):
    """
    Analyzes how the number of pre-StS status changes (i.e., any action)
    impacts downstream conversions.

    Args:
        df: DataFrame with referral data
        ts_col_map: Dictionary mapping stage names to timestamp columns
        status_history_col: Name of column containing parsed status history
        business_hours_only: If True, only count actions during business hours (Mon-Fri 9am-5pm)
    """
    if df is None or df.empty or ts_col_map is None:
        return pd.DataFrame()

    psa_ts_col = ts_col_map.get("Pre-Screening Activities")
    sts_ts_col = ts_col_map.get("Sent To Site")
    if not all(col in df.columns for col in [psa_ts_col, sts_ts_col, status_history_col]):
        return pd.DataFrame()

    # Get all stage timestamps to determine if referral left first stage
    lost_ts_col = ts_col_map.get("Lost")
    screenfailed_ts_col = ts_col_map.get("Screen Failed")
    appt_ts_col = ts_col_map.get("Appointment Scheduled")
    icf_ts_col = ts_col_map.get("Signed ICF")
    enr_ts_col = ts_col_map.get("Enrolled")

    analysis_df = df.copy()

    def count_pre_sts_actions(row):
        psa_timestamp = row[psa_ts_col]
        sts_timestamp = row[sts_ts_col]
        history = row[status_history_col]

        # Check if referral has left the first stage (Passed Online Form)
        # by checking if they have ANY timestamp for stages beyond the first
        has_left_first_stage = (
            pd.notna(psa_timestamp) or
            pd.notna(sts_timestamp) or
            (lost_ts_col and pd.notna(row.get(lost_ts_col))) or
            (screenfailed_ts_col and pd.notna(row.get(screenfailed_ts_col))) or
            (appt_ts_col and pd.notna(row.get(appt_ts_col))) or
            (icf_ts_col and pd.notna(row.get(icf_ts_col))) or
            (enr_ts_col and pd.notna(row.get(enr_ts_col)))
        )

        # Only referrals still in first stage (Passed Online Form) should have 0 attempts
        if not has_left_first_stage:
            return 0

        # If referral has left first stage, they have at least 1 attempt
        # Handle invalid/empty history gracefully
        if not isinstance(history, list) or len(history) == 0:
            # Data quality issue: Referral left first stage but no history
            # Still count as 1 attempt (the stage transition)
            return 1

        # For referrals that went through PSA, count actions between PSA and StS
        # For referrals that skipped PSA (e.g., went to Lost directly), count from earliest stage change
        if pd.notna(psa_timestamp):
            start_timestamp = psa_timestamp
        else:
            # Find the earliest non-POF stage timestamp
            stage_timestamps = [
                sts_timestamp,
                row.get(lost_ts_col) if lost_ts_col else pd.NaT,
                row.get(screenfailed_ts_col) if screenfailed_ts_col else pd.NaT,
                row.get(appt_ts_col) if appt_ts_col else pd.NaT,
                row.get(icf_ts_col) if icf_ts_col else pd.NaT,
                row.get(enr_ts_col) if enr_ts_col else pd.NaT
            ]
            valid_timestamps = [ts for ts in stage_timestamps if pd.notna(ts)]
            start_timestamp = min(valid_timestamps) if valid_timestamps else pd.Timestamp.max

        end_window = sts_timestamp if pd.notna(sts_timestamp) else pd.Timestamp.max

        # Count actions, filtering for business hours if requested
        # IMPORTANT: Stage transition always counts as attempt #1, regardless of when it occurred
        if business_hours_only:
            # Start with 1 (stage transition), then add business-hour events after transition
            action_count = 1 + sum(1 for _, event_dt in history
                                 if start_timestamp < event_dt < end_window and is_business_hours(event_dt))
        else:
            # Count all events including stage transition
            action_count = sum(1 for _, event_dt in history
                             if start_timestamp <= event_dt < end_window)
            # Ensure at least 1 attempt if referral left first stage
            action_count = max(1, action_count)

        return action_count

    analysis_df['pre_sts_action_count'] = analysis_df.apply(count_pre_sts_actions, axis=1)

    icf_col = ts_col_map.get("Signed ICF")
    enr_col = ts_col_map.get("Enrolled")

    result = analysis_df.groupby('pre_sts_action_count').agg(
        Referral_Count=('pre_sts_action_count', 'size'),
        Total_StS=(sts_ts_col, lambda x: x.notna().sum()),
        Total_ICF=(icf_col, lambda x: x.notna().sum()),
        Total_Enrolled=(enr_col, lambda x: x.notna().sum())
    )

    result['StS_Rate'] = (result['Total_StS'] / result['Referral_Count'].replace(0, np.nan))
    result['ICF_Rate'] = (result['Total_ICF'] / result['Referral_Count'].replace(0, np.nan))
    result['Enrollment_Rate'] = (result['Total_Enrolled'] / result['Referral_Count'].replace(0, np.nan))
    
    result.reset_index(inplace=True)
    
    result.rename(columns={
        'pre_sts_action_count': 'Number of Attempts',
        'Referral_Count': 'Total Referrals'
    }, inplace=True)
    
    return result

def calculate_performance_over_time(df, ts_col_map):
    """
    Calculates key PC performance metrics over time on a weekly basis,
    with transit-time adjustment for Sent to Site %.
    """
    if df is None or df.empty or 'Submitted On_DT' not in df.columns:
        return pd.DataFrame()

    pof_ts_col = ts_col_map.get("Passed Online Form")
    psa_ts_col = ts_col_map.get("Pre-Screening Activities")
    sts_ts_col = ts_col_map.get("Sent To Site")

    if not all(col in df.columns for col in [pof_ts_col, psa_ts_col, sts_ts_col]):
        return pd.DataFrame()

    avg_pof_to_sts_lag = calculate_avg_lag_generic(df, pof_ts_col, sts_ts_col)
    maturity_days = (avg_pof_to_sts_lag * 1.5) if pd.notna(avg_pof_to_sts_lag) else 30

    time_df = df.set_index('Submitted On_DT')

    weekly_summary = time_df.resample('W').apply(lambda week_df: pd.Series({
        'Total Qualified per Week': (
            len(week_df)
        ),
        'Sent to Site % (Transit-Time Adjusted)': (
            week_df[week_df.index + pd.Timedelta(days=maturity_days) < pd.Timestamp.now()]
            .pipe(lambda mature_df: mature_df[sts_ts_col].notna().sum() / len(mature_df) if len(mature_df) > 0 else 0)
        ),
        'Average Time to First Contact (Days)': calculate_avg_lag_generic(
            week_df, pof_ts_col, psa_ts_col
        ),
        'Average Sent to Site per Day': (
            week_df[sts_ts_col].notna().sum() / 7
        ),
        'Total Sent to Site per Week': (
            week_df[sts_ts_col].notna().sum()
        )
    }))

    weekly_summary['Sent to Site % (Transit-Time Adjusted)'] *= 100
    weekly_summary.fillna(method='ffill', inplace=True)

    return weekly_summary

def analyze_heatmap_efficiency(contact_heatmap, sts_heatmap, business_hours_only=False):
    """
    Analyzes the two heatmaps to identify best and worst times for contact attempts.

    Args:
        contact_heatmap: DataFrame with contact counts by day/hour
        sts_heatmap: DataFrame with StS counts by day/hour
        business_hours_only: If True, only analyze business hours (Mon-Fri 9am-5pm)

    Returns:
        dict: {
            'best_times': List of up to 10 time slots with highest efficiency,
            'avoid_times': List of up to 10 time slots with high volume but low efficiency
        }

    Focus on efficiency (success rate) with no overlaps between categories.
    """
    if contact_heatmap.empty or sts_heatmap.empty or contact_heatmap.sum().sum() == 0:
        return {}

    # Convert heatmaps to long format
    contacts_long = contact_heatmap.stack().reset_index()
    contacts_long.columns = ['Day', 'Hour', 'Contacts']
    sts_long = sts_heatmap.stack().reset_index()
    sts_long.columns = ['Day', 'Hour', 'StS']

    merged_df = pd.merge(contacts_long, sts_long, on=['Day', 'Hour'])

    # Filter to business hours only if requested
    if business_hours_only:
        from constants import BUSINESS_HOURS_START, BUSINESS_HOURS_END, BUSINESS_DAYS
        business_day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        merged_df = merged_df[
            (merged_df['Day'].isin(business_day_names)) &
            (merged_df['Hour'] >= BUSINESS_HOURS_START) &
            (merged_df['Hour'] < BUSINESS_HOURS_END)
        ].copy()

    # Check if we have any data after filtering
    if merged_df.empty or merged_df['Contacts'].sum() == 0:
        return {'best_times': [], 'avoid_times': []}

    # Calculate efficiency (success rate)
    merged_df['Efficiency'] = (merged_df['StS'] / merged_df['Contacts']).replace([np.inf, -np.inf], 0).fillna(0)

    # Define minimum volume threshold for statistical significance
    # Use 50th percentile to ensure we have enough data points
    min_contacts_threshold = merged_df['Contacts'].quantile(0.50)

    # Filter for slots with reasonable volume
    valid_slots = merged_df[merged_df['Contacts'] >= min_contacts_threshold].copy()

    if valid_slots.empty:
        return {'best_times': [], 'avoid_times': []}

    # Sort by efficiency
    valid_slots = valid_slots.sort_values(by='Efficiency', ascending=False)

    # BEST TIMES TO CALL: Top 10 slots by efficiency
    best_times_df = valid_slots.head(10)

    # AVOID THESE TIMES: High volume but low efficiency
    # Exclude slots already in best_times
    remaining_slots = valid_slots.iloc[10:]  # Everything after top 10

    # Further filter for high volume (75th percentile or higher)
    high_volume_threshold = merged_df['Contacts'].quantile(0.75)
    avoid_times_df = remaining_slots[
        remaining_slots['Contacts'] >= high_volume_threshold
    ].sort_values(by='Efficiency', ascending=True).head(10)

    def format_hour(hour):
        """Format hour as 12-hour time."""
        if hour == 0: return "12 AM"
        if hour == 12: return "12 PM"
        if hour < 12: return f"{hour} AM"
        return f"{hour-12} PM"

    # Format results
    results = {
        "best_times": [
            f"{row.Day}, {format_hour(row.Hour)}"
            for _, row in best_times_df.iterrows()
        ],
        "avoid_times": [
            f"{row.Day}, {format_hour(row.Hour)}"
            for _, row in avoid_times_df.iterrows()
        ],
    }

    return results