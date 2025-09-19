# calculations.py
import streamlit as st
import pandas as pd
import numpy as np

from constants import * # Import all stage names

def calculate_avg_lag_generic(df, col_from, col_to):
    """
    Safely calculates the average time lag in days between two datetime columns.
    """
    if col_from is None or col_to is None or col_from not in df.columns or col_to not in df.columns:
        return np.nan

    if not all([pd.api.types.is_datetime64_any_dtype(df[col_from]),
                pd.api.types.is_datetime64_any_dtype(df[col_to])]):
        return np.nan

    valid_df = df.dropna(subset=[col_from, col_to])
    if valid_df.empty: return np.nan

    diff = pd.to_datetime(valid_df[col_to]) - pd.to_datetime(valid_df[col_from])
    diff_positive = diff[diff >= pd.Timedelta(days=0)]

    return diff_positive.mean().total_seconds() / (60 * 60 * 24) if not diff_positive.empty else np.nan


@st.cache_data
def calculate_overall_inter_stage_lags(_processed_df, ordered_stages, ts_col_map):
    if _processed_df is None or _processed_df.empty or not ordered_stages or not ts_col_map:
        return {}
    inter_stage_lags = {}
    for i in range(len(ordered_stages) - 1):
        stage_from, stage_to = ordered_stages[i], ordered_stages[i+1]
        ts_col_from, ts_col_to = ts_col_map.get(stage_from), ts_col_map.get(stage_to)
        
        avg_lag = calculate_avg_lag_generic(_processed_df, ts_col_from, ts_col_to)
        inter_stage_lags[f"{stage_from} -> {stage_to}"] = avg_lag
        
    return inter_stage_lags


def calculate_proforma_metrics(_processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    if _processed_df is None or _processed_df.empty: return pd.DataFrame()
    
    processed_df = _processed_df.copy()
    
    cohort_summary = processed_df.groupby("Submission_Month").size().reset_index(name="Total Qualified Referrals_Calc")
    cohort_summary = cohort_summary.set_index("Submission_Month")
    cohort_summary["Ad Spend"] = cohort_summary.index.map(monthly_ad_spend_input).fillna(0)
    
    reached_stage_cols_map = {}
    for stage_name in ordered_stages:
        ts_col = ts_col_map.get(stage_name)
        if ts_col and ts_col in processed_df.columns:
            reached_col = f"Reached_{stage_name.replace(' ', '_')}"
            reached_stage_cols_map[stage_name] = reached_col
            reached_stage_count = processed_df.dropna(subset=[ts_col]).groupby("Submission_Month").size()
            cohort_summary[reached_col] = reached_stage_count
    
    cohort_summary = cohort_summary.fillna(0)
    for col in cohort_summary.columns:
        if col != "Ad Spend": cohort_summary[col] = cohort_summary[col].astype(int)
    cohort_summary["Ad Spend"] = cohort_summary["Ad Spend"].astype(float)
    
    base_count_col = reached_stage_cols_map.get(STAGE_PASSED_ONLINE_FORM, "Total Qualified Referrals_Calc")
    if base_count_col in cohort_summary.columns:
        cohort_summary.rename(columns={base_count_col: "Pre-Screener Qualified"}, inplace=True)
    base_count_col_name = "Pre-Screener Qualified"
    
    proforma_metrics = pd.DataFrame(index=cohort_summary.index)
    if base_count_col_name in cohort_summary.columns:
        proforma_metrics["Ad Spend"] = cohort_summary["Ad Spend"]
        proforma_metrics["Pre-Screener Qualified"] = cohort_summary[base_count_col_name]
        proforma_metrics["Cost per Qualified Pre-screen"] = (cohort_summary["Ad Spend"] / cohort_summary[base_count_col_name].replace(0, np.nan)).round(2)

        for stage, reached_col in reached_stage_cols_map.items():
            metric_name = f"Total {stage}" if stage != STAGE_PASSED_ONLINE_FORM else "Pre-Screener Qualified"
            if reached_col in cohort_summary.columns:
                proforma_metrics[metric_name] = cohort_summary[reached_col]
        
        sts_col = reached_stage_cols_map.get(STAGE_SENT_TO_SITE)
        appt_col = reached_stage_cols_map.get(STAGE_APPOINTMENT_SCHEDULED)
        icf_col = reached_stage_cols_map.get(STAGE_SIGNED_ICF)

        if sts_col in cohort_summary: proforma_metrics["Qualified to StS %"] = (cohort_summary[sts_col] / cohort_summary[base_count_col_name].replace(0, np.nan))
        if sts_col in cohort_summary and appt_col in cohort_summary: proforma_metrics["StS to Appt Sched %"] = (cohort_summary[appt_col] / cohort_summary[sts_col].replace(0, np.nan))
        if appt_col in cohort_summary and icf_col in cohort_summary: proforma_metrics["Appt Sched to ICF %"] = (cohort_summary[icf_col] / cohort_summary[appt_col].replace(0, np.nan))
        if icf_col in cohort_summary:
            proforma_metrics["Qualified to ICF %"] = (cohort_summary[icf_col] / cohort_summary[base_count_col_name].replace(0, np.nan))
            proforma_metrics["Cost Per ICF"] = (cohort_summary["Ad Spend"] / cohort_summary[icf_col].replace(0, np.nan)).round(2)

    return proforma_metrics


def calculate_grouped_performance_metrics(_processed_df, ordered_stages, ts_col_map, grouping_col: str, unclassified_label="Unclassified"):
    if _processed_df is None or _processed_df.empty: return pd.DataFrame()

    df = _processed_df.copy()
    if grouping_col not in df.columns:
        df[grouping_col] = unclassified_label
    df[grouping_col] = df[grouping_col].astype(str).str.strip().replace('', unclassified_label).fillna(unclassified_label)
    
    # --- THIS IS THE CORRECTED LOGIC BLOCK ---
    
    # Define all possible stages that might have counts
    all_possible_stages = [
        STAGE_PASSED_ONLINE_FORM, STAGE_PRE_SCREENING_ACTIVITIES, STAGE_SENT_TO_SITE,
        STAGE_APPOINTMENT_SCHEDULED, STAGE_SIGNED_ICF, STAGE_SCREEN_FAILED, STAGE_ENROLLED
    ]
    
    # Ensure all possible timestamp columns exist, even if they are all null
    for stage in all_possible_stages:
        ts_col = ts_col_map.get(stage)
        if ts_col and ts_col not in df.columns:
            df[ts_col] = pd.NaT

    performance_metrics_list = []
    for group_name, group_df in df.groupby(grouping_col):
        metrics = {grouping_col: group_name}
        
        # Calculate counts for each stage safely
        counts = {}
        for stage in all_possible_stages:
            ts_col = ts_col_map.get(stage)
            if ts_col and ts_col in group_df.columns:
                counts[stage] = group_df[ts_col].notna().sum()
            else:
                counts[stage] = 0

        # Assign counts to metrics dictionary with user-friendly names
        pof_count = counts.get(STAGE_PASSED_ONLINE_FORM, 0)
        psa_count = counts.get(STAGE_PRE_SCREENING_ACTIVITIES, 0)
        sts_count = counts.get(STAGE_SENT_TO_SITE, 0)
        appt_count = counts.get(STAGE_APPOINTMENT_SCHEDULED, 0)
        icf_count = counts.get(STAGE_SIGNED_ICF, 0)
        sf_count = counts.get(STAGE_SCREEN_FAILED, 0)
        enrolled_count = counts.get(STAGE_ENROLLED, 0)

        metrics['Total Qualified'] = pof_count
        metrics['Pre-Screening Activities Count'] = psa_count
        metrics['Sent To Site Count'] = sts_count
        metrics['Appointment Scheduled Count'] = appt_count
        metrics['Signed ICF Count'] = icf_count
        metrics['Screen Failed Count'] = sf_count # Added for completeness
        metrics['Enrollment Count'] = enrolled_count # This now correctly adds the column

        # Calculate Conversion Rates
        metrics['POF -> PSA %'] = (psa_count / pof_count) if pof_count > 0 else 0.0
        metrics['PSA -> StS %'] = (sts_count / psa_count) if psa_count > 0 else 0.0
        metrics['StS -> Appt %'] = (appt_count / sts_count) if sts_count > 0 else 0.0
        metrics['Appt -> ICF %'] = (icf_count / appt_count) if appt_count > 0 else 0.0
        metrics['ICF to Enrollment %'] = (enrolled_count / icf_count) if icf_count > 0 else 0.0
        metrics['Qual -> ICF %'] = (icf_count / pof_count) if pof_count > 0 else 0.0
        metrics['Qual to Enrollment %'] = (enrolled_count / pof_count) if pof_count > 0 else 0.0
        
        # Screen Fail and Lag Metrics
        screen_fail_metric = 'Screen Fail % (from ICF)'
        projection_lag_metric = 'Projection Lag (Days)'
        if grouping_col == 'Site':
             screen_fail_metric = 'Site Screen Fail %'
             projection_lag_metric = 'Site Projection Lag (Days)'

        metrics[screen_fail_metric] = (sf_count / icf_count) if icf_count > 0 else 0.0
        
        projection_segments = [
            (STAGE_PASSED_ONLINE_FORM, STAGE_PRE_SCREENING_ACTIVITIES),
            (STAGE_PRE_SCREENING_ACTIVITIES, STAGE_SENT_TO_SITE),
            (STAGE_SENT_TO_SITE, STAGE_APPOINTMENT_SCHEDULED),
            (STAGE_APPOINTMENT_SCHEDULED, STAGE_SIGNED_ICF)
        ]
        
        total_lag = 0
        valid_segments = 0
        for from_s, to_s in projection_segments:
            lag = calculate_avg_lag_generic(group_df, ts_col_map.get(from_s), ts_col_map.get(to_s))
            if pd.notna(lag):
                total_lag += lag
                valid_segments += 1
        metrics[projection_lag_metric] = total_lag if valid_segments == len(projection_segments) else np.nan
        metrics['Lag Qual -> ICF (Days)'] = calculate_avg_lag_generic(group_df, ts_col_map.get(STAGE_PASSED_ONLINE_FORM), ts_col_map.get(STAGE_SIGNED_ICF))
        
        if grouping_col == 'Site':
            metrics['Avg TTC (Days)'] = np.nan 
            metrics['Avg Funnel Movement Steps'] = 0.0
            
        performance_metrics_list.append(metrics)

    return pd.DataFrame(performance_metrics_list)


def calculate_site_metrics(_processed_df, ordered_stages, ts_col_map):
    if _processed_df is None or 'Site' not in _processed_df.columns:
        st.warning("Cannot calculate site metrics: 'Site' column not found.")
        return pd.DataFrame()
    return calculate_grouped_performance_metrics(
        _processed_df, ordered_stages, ts_col_map,
        grouping_col="Site",
        unclassified_label="Unassigned Site"
    )