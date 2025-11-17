# comparison_calculations.py
"""
Core calculation functions for the Comparison Analysis feature.
Handles date filtering, delta calculations, and statistical significance testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, date

from calculations import (
    calculate_enhanced_site_metrics,
    calculate_site_operational_kpis,
    calculate_site_ttfc_effectiveness,
    calculate_site_contact_attempt_effectiveness,
    calculate_site_performance_over_time,
    calculate_lost_reasons_after_sts,
    calculate_stale_referrals
)
from pc_calculations import (
    calculate_heatmap_data,
    calculate_average_time_metrics,
    calculate_ttfc_effectiveness,
    calculate_contact_attempt_effectiveness,
    calculate_top_status_flows,
    calculate_performance_over_time
)
from forecasting import (
    determine_effective_projection_rates,
    calculate_pipeline_projection
)
from scoring import score_sites, score_performance_groups


def filter_by_date_range(df: pd.DataFrame, start_date, end_date, date_column='Submitted On_DT') -> pd.DataFrame:
    """
    Filter dataframe by date range.

    Args:
        df: DataFrame to filter
        start_date: Start date (datetime, date, or string)
        end_date: End date (datetime, date, or string)
        date_column: Column name containing dates

    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()

    # Convert to pandas Timestamp
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Filter
    mask = (df_filtered[date_column] >= start_ts) & (df_filtered[date_column] <= end_ts)

    return df_filtered[mask].copy()


def validate_date_ranges(start_a, end_a, start_b, end_b, df: pd.DataFrame, date_column='Submitted On_DT') -> Dict[str, Any]:
    """
    Validate date ranges and check for data sufficiency.

    Returns dict with:
    - valid: bool
    - warnings: list of warning messages
    - errors: list of error messages
    - period_a_count: number of records in period A
    - period_b_count: number of records in period B
    """
    warnings = []
    errors = []

    # Check date logic
    if start_a >= end_a:
        errors.append("Period A: Start date must be before end date")
    if start_b >= end_b:
        errors.append("Period B: Start date must be before end date")

    # Check for date range overlap
    if not (end_a < start_b or end_b < start_a):
        warnings.append("Warning: Date ranges overlap. Results may be misleading.")

    # Check data availability
    df_a = filter_by_date_range(df, start_a, end_a, date_column)
    df_b = filter_by_date_range(df, start_b, end_b, date_column)

    period_a_count = len(df_a)
    period_b_count = len(df_b)

    # Data sufficiency warnings
    if period_a_count < 10:
        warnings.append(f"Period A has only {period_a_count} referrals. Statistical tests may be unreliable.")
    if period_b_count < 10:
        warnings.append(f"Period B has only {period_b_count} referrals. Statistical tests may be unreliable.")

    if period_a_count == 0:
        errors.append("Period A has no data")
    if period_b_count == 0:
        errors.append("Period B has no data")

    # Check time period duration
    days_a = (end_a - start_a).days
    days_b = (end_b - start_b).days

    if days_a < 7:
        warnings.append(f"Period A is only {days_a} days long. Consider a longer time period.")
    if days_b < 7:
        warnings.append(f"Period B is only {days_b} days long. Consider a longer time period.")

    return {
        'valid': len(errors) == 0,
        'warnings': warnings,
        'errors': errors,
        'period_a_count': period_a_count,
        'period_b_count': period_b_count,
        'period_a_days': days_a,
        'period_b_days': days_b
    }


def merge_and_calculate_deltas(df_a: pd.DataFrame, df_b: pd.DataFrame,
                               key_column: str,
                               metric_configs: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """
    Merge two dataframes and calculate deltas for numeric columns.

    Args:
        df_a: Period A dataframe
        df_b: Period B dataframe
        key_column: Column to merge on (e.g., 'Site', 'UTM Source')
        metric_configs: Optional dict specifying which metrics are "inverse" (lower is better)
                       Format: {'metric_name': {'inverse': True/False}}

    Returns:
        Merged dataframe with delta columns and improvement indicators
    """
    # Merge dataframes
    merged = df_a.merge(df_b, on=key_column, how='outer', suffixes=('_A', '_B'))

    # Fill NaN for entities that only exist in one period
    for col in merged.columns:
        if col.endswith('_A') or col.endswith('_B'):
            merged[col] = merged[col].fillna(0)

    # Calculate deltas for numeric columns
    numeric_cols_a = [col for col in df_a.columns if col != key_column and pd.api.types.is_numeric_dtype(df_a[col])]

    for col in numeric_cols_a:
        col_a = f"{col}_A"
        col_b = f"{col}_B"

        if col_a in merged.columns and col_b in merged.columns:
            # Absolute delta
            merged[f'{col}_Delta'] = merged[col_b] - merged[col_a]

            # Percentage delta
            merged[f'{col}_Delta_Pct'] = np.where(
                merged[col_a] != 0,
                ((merged[col_b] - merged[col_a]) / merged[col_a]) * 100,
                np.nan
            )

            # Determine if change is improvement (depends on metric type)
            is_inverse = False
            if metric_configs and col in metric_configs:
                is_inverse = metric_configs[col].get('inverse', False)

            # For inverse metrics (lower is better), improvement is negative delta
            if is_inverse:
                merged[f'{col}_Improved'] = merged[f'{col}_Delta'] < 0
            else:
                merged[f'{col}_Improved'] = merged[f'{col}_Delta'] > 0

    # Calculate rank changes if Score column exists
    if 'Score_A' in merged.columns and 'Score_B' in merged.columns:
        # Sort by score descending to get ranks
        merged['Rank_A'] = merged['Score_A'].rank(ascending=False, method='min').astype(int)
        merged['Rank_B'] = merged['Score_B'].rank(ascending=False, method='min').astype(int)

        # Positive = moved up (lower rank number), Negative = moved down
        merged['Rank_Change'] = merged['Rank_A'] - merged['Rank_B']

        # Direction indicator
        merged['Rank_Direction'] = merged['Rank_Change'].apply(
            lambda x: '↑' if x > 0 else '↓' if x < 0 else '→'
        )

    return merged


def calculate_statistical_significance(data_a: pd.Series, data_b: pd.Series,
                                       metric_name: str,
                                       test_type: str = 'auto') -> Dict[str, Any]:
    """
    Perform statistical significance testing between two samples.

    Args:
        data_a: Period A data series
        data_b: Period B data series
        metric_name: Name of the metric being tested
        test_type: 'ttest' for continuous, 'proportion' for rates, 'auto' to decide automatically

    Returns:
        Dict with p_value, is_significant, test_used, significance_level
    """
    # Remove NaN values
    data_a_clean = data_a.dropna()
    data_b_clean = data_b.dropna()

    if len(data_a_clean) < 2 or len(data_b_clean) < 2:
        return {
            'p_value': None,
            'is_significant': False,
            'test_used': 'insufficient_data',
            'significance_level': 'N/A',
            'warning': 'Insufficient data for statistical testing'
        }

    try:
        if test_type == 'auto':
            # Auto-detect test type based on metric name and data
            if '%' in metric_name or 'Rate' in metric_name or 'Conversion' in metric_name:
                test_type = 'proportion'
            else:
                test_type = 'ttest'

        if test_type == 'ttest':
            # Independent samples t-test
            t_stat, p_value = stats.ttest_ind(data_a_clean, data_b_clean, nan_policy='omit')
            test_used = 't-test'

        elif test_type == 'proportion':
            # For proportions, use Mann-Whitney U test (non-parametric)
            # More appropriate for percentage/rate data
            u_stat, p_value = stats.mannwhitneyu(data_a_clean, data_b_clean, alternative='two-sided')
            test_used = 'Mann-Whitney U'

        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Determine significance level
        is_significant = p_value < 0.05

        if p_value < 0.001:
            sig_level = '***'
        elif p_value < 0.01:
            sig_level = '**'
        elif p_value < 0.05:
            sig_level = '*'
        else:
            sig_level = 'ns'

        return {
            'p_value': p_value,
            'is_significant': is_significant,
            'test_used': test_used,
            'significance_level': sig_level,
            'warning': None
        }

    except Exception as e:
        return {
            'p_value': None,
            'is_significant': False,
            'test_used': 'error',
            'significance_level': 'N/A',
            'warning': f'Error during statistical test: {str(e)}'
        }


def calculate_comparison_for_site_performance(
    df: pd.DataFrame,
    date_range_a: Tuple,
    date_range_b: Tuple,
    ordered_stages: List[str],
    ts_col_map: Dict,
    weights: Dict,
    business_hours_only: bool = False,
    compare_full_table: bool = True,
    selected_metrics: Optional[List[str]] = None,
    contact_status_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate site performance comparison between two date ranges.

    Returns dict with:
    - period_a: DataFrame for period A
    - period_b: DataFrame for period B
    - comparison: Merged DataFrame with deltas
    - type: 'full_table' or 'selected_metrics'
    - validation: Validation results
    """
    start_a, end_a = date_range_a
    start_b, end_b = date_range_b

    # Validate date ranges
    validation = validate_date_ranges(start_a, end_a, start_b, end_b, df)

    if not validation['valid']:
        return {
            'error': True,
            'validation': validation
        }

    # Filter data by date ranges
    df_a = filter_by_date_range(df, start_a, end_a)
    df_b = filter_by_date_range(df, start_b, end_b)

    if compare_full_table:
        # Calculate full enhanced metrics + scoring for both periods
        enhanced_a = calculate_enhanced_site_metrics(
            df_a, ordered_stages, ts_col_map,
            "Parsed_Lead_Status_History",
            business_hours_only=business_hours_only
        )

        enhanced_b = calculate_enhanced_site_metrics(
            df_b, ordered_stages, ts_col_map,
            "Parsed_Lead_Status_History",
            business_hours_only=business_hours_only
        )

        # Score sites
        ranked_a = score_sites(enhanced_a, weights)
        ranked_b = score_sites(enhanced_b, weights)

        # Define metric configs (which metrics are inverse/lower is better)
        metric_configs = {
            'Average time to first site action': {'inverse': True},
            'Avg time from StS to Appt Sched.': {'inverse': True},
            'Avg. Time Between Site Contacts': {'inverse': True},
            'Avg time from StS to ICF': {'inverse': True},
            'Avg time from StS to Enrollment': {'inverse': True},
            'Total Referrals Awaiting First Site Action': {'inverse': True},
            'SF or Lost After ICF %': {'inverse': True},
            'StS to Lost %': {'inverse': True},
            'SF or Lost After ICF Count': {'inverse': True},
            'Lost After StS': {'inverse': True},
            'Total Lost Count': {'inverse': True}
        }

        # Calculate deltas
        comparison_df = merge_and_calculate_deltas(ranked_a, ranked_b, 'Site', metric_configs)

        return {
            'error': False,
            'period_a': ranked_a,
            'period_b': ranked_b,
            'comparison': comparison_df,
            'type': 'full_table',
            'validation': validation
        }
    else:
        # Calculate only selected metrics
        # This would be implemented for individual metric selection
        return {
            'error': True,
            'validation': {'errors': ['Individual metric selection not yet implemented']}
        }


def calculate_comparison_for_ad_performance(
    df: pd.DataFrame,
    date_range_a: Tuple,
    date_range_b: Tuple,
    ordered_stages: List[str],
    ts_col_map: Dict,
    weights: Dict,
    table_type: str = 'source'
) -> Dict[str, Any]:
    """
    Calculate ad performance comparison between two date ranges.

    Args:
        table_type: 'source' for UTM Source, 'combo' for UTM Source/Medium

    Returns comparison results dict
    """
    from calculations import calculate_enhanced_ad_metrics

    start_a, end_a = date_range_a
    start_b, end_b = date_range_b

    # Validate date ranges
    validation = validate_date_ranges(start_a, end_a, start_b, end_b, df)

    if not validation['valid']:
        return {
            'error': True,
            'validation': validation
        }

    # Filter data
    df_a = filter_by_date_range(df, start_a, end_a)
    df_b = filter_by_date_range(df, start_b, end_b)

    # Calculate enhanced metrics
    if table_type == 'source':
        enhanced_a = calculate_enhanced_ad_metrics(
            df_a, ordered_stages, ts_col_map,
            "UTM Source", "Unclassified Source"
        )
        enhanced_b = calculate_enhanced_ad_metrics(
            df_b, ordered_stages, ts_col_map,
            "UTM Source", "Unclassified Source"
        )
        key_col = 'UTM Source'
    else:  # combo
        # Create combined column like in app.py
        df_a_combo = df_a.copy()
        df_a_combo['UTM Source/Medium'] = df_a_combo['UTM Source'].astype(str).fillna("Unclassified") + ' / ' + df_a_combo['UTM Medium'].astype(str).fillna("Unclassified")

        df_b_combo = df_b.copy()
        df_b_combo['UTM Source/Medium'] = df_b_combo['UTM Source'].astype(str).fillna("Unclassified") + ' / ' + df_b_combo['UTM Medium'].astype(str).fillna("Unclassified")

        enhanced_a = calculate_enhanced_ad_metrics(
            df_a_combo, ordered_stages, ts_col_map,
            "UTM Source/Medium", "Unclassified Combo"
        )
        enhanced_b = calculate_enhanced_ad_metrics(
            df_b_combo, ordered_stages, ts_col_map,
            "UTM Source/Medium", "Unclassified Combo"
        )
        key_col = 'UTM Source/Medium'

    # Score performance groups
    ranked_a = score_performance_groups(enhanced_a, weights, key_col)
    ranked_b = score_performance_groups(enhanced_b, weights, key_col)

    # Metric configs
    metric_configs = {
        'Average time to first site action': {'inverse': True},
        'Avg time from StS to Appt Sched.': {'inverse': True},
        'Screen Fail % (from Qualified)': {'inverse': True}
    }

    # Calculate deltas
    comparison_df = merge_and_calculate_deltas(ranked_a, ranked_b, key_col, metric_configs)

    return {
        'error': False,
        'period_a': ranked_a,
        'period_b': ranked_b,
        'comparison': comparison_df,
        'type': f'{table_type}_table',
        'validation': validation,
        'key_column': key_col
    }


def calculate_comparison_for_pc_performance(
    df: pd.DataFrame,
    date_range_a: Tuple,
    date_range_b: Tuple,
    ts_col_map: Dict,
    comparison_type: str = 'time_metrics',
    business_hours_only: bool = False,
    contact_status_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate PC performance comparison.

    Args:
        comparison_type: 'heatmap', 'time_metrics', 'contact_effectiveness', 'weekly_trends'
    """
    start_a, end_a = date_range_a
    start_b, end_b = date_range_b

    # Validate
    validation = validate_date_ranges(start_a, end_a, start_b, end_b, df)

    if not validation['valid']:
        return {
            'error': True,
            'validation': validation
        }

    # Filter data
    df_a = filter_by_date_range(df, start_a, end_a)
    df_b = filter_by_date_range(df, start_b, end_b)

    if comparison_type == 'time_metrics':
        # Calculate time metrics for both periods
        from pc_calculations import calculate_average_time_metrics

        metrics_a = calculate_average_time_metrics(df_a, ts_col_map, "Parsed_Lead_Status_History", business_hours_only=business_hours_only)
        metrics_b = calculate_average_time_metrics(df_b, ts_col_map, "Parsed_Lead_Status_History", business_hours_only=business_hours_only)

        return {
            'error': False,
            'period_a': metrics_a,
            'period_b': metrics_b,
            'type': 'time_metrics',
            'validation': validation
        }

    elif comparison_type == 'contact_effectiveness':
        # Calculate contact effectiveness for both periods
        effectiveness_a = calculate_contact_attempt_effectiveness(
            df_a, ts_col_map, "Parsed_Lead_Status_History",
            business_hours_only
        )
        effectiveness_b = calculate_contact_attempt_effectiveness(
            df_b, ts_col_map, "Parsed_Lead_Status_History",
            business_hours_only
        )

        # Merge and calculate deltas
        metric_configs = {}  # Define as needed
        comparison_df = merge_and_calculate_deltas(
            effectiveness_a, effectiveness_b,
            'Number of Attempts',
            metric_configs
        )

        return {
            'error': False,
            'period_a': effectiveness_a,
            'period_b': effectiveness_b,
            'comparison': comparison_df,
            'type': 'contact_effectiveness',
            'validation': validation
        }

    elif comparison_type == 'time_to_contact_effectiveness':
        # Calculate time to first contact effectiveness for both periods
        from pc_calculations import calculate_ttfc_effectiveness

        ttfc_a = calculate_ttfc_effectiveness(df_a, ts_col_map, business_hours_only=business_hours_only)
        ttfc_b = calculate_ttfc_effectiveness(df_b, ts_col_map, business_hours_only=business_hours_only)

        # Merge and calculate deltas
        metric_configs = {}  # Define as needed
        comparison_df = merge_and_calculate_deltas(
            ttfc_a, ttfc_b,
            'Time to First Contact',
            metric_configs
        )

        return {
            'error': False,
            'period_a': ttfc_a,
            'period_b': ttfc_b,
            'comparison': comparison_df,
            'type': 'time_to_contact_effectiveness',
            'validation': validation
        }

    else:
        return {
            'error': True,
            'validation': {'errors': [f'Comparison type "{comparison_type}" not yet implemented']}
        }


def calculate_comparison_for_funnel(
    df: pd.DataFrame,
    date_range_a: Tuple,
    date_range_b: Tuple,
    ordered_stages: List[str],
    ts_col_map: Dict,
    inter_stage_lags: Dict,
    comparison_type: str = 'full_projection',
    rate_method: str = 'Rolling Historical Average',
    rolling_window: int = 3,
    manual_rates: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Calculate funnel analysis comparison.

    Args:
        comparison_type: 'full_projection' or 'stage_breakdown'
    """
    start_a, end_a = date_range_a
    start_b, end_b = date_range_b

    # Validate
    validation = validate_date_ranges(start_a, end_a, start_b, end_b, df)

    if not validation['valid']:
        return {
            'error': True,
            'validation': validation
        }

    # Filter data
    df_a = filter_by_date_range(df, start_a, end_a)
    df_b = filter_by_date_range(df, start_b, end_b)

    # Calculate projection for both periods
    effective_rates_a, rates_desc_a = determine_effective_projection_rates(
        df_a, ordered_stages, ts_col_map,
        rate_method, rolling_window, manual_rates or {}, inter_stage_lags
    )

    effective_rates_b, rates_desc_b = determine_effective_projection_rates(
        df_b, ordered_stages, ts_col_map,
        rate_method, rolling_window, manual_rates or {}, inter_stage_lags
    )

    results_a = calculate_pipeline_projection(
        _processed_df=df_a,
        ordered_stages=ordered_stages,
        ts_col_map=ts_col_map,
        inter_stage_lags=inter_stage_lags,
        conversion_rates=effective_rates_a,
        lag_assumption_model=None
    )

    results_b = calculate_pipeline_projection(
        _processed_df=df_b,
        ordered_stages=ordered_stages,
        ts_col_map=ts_col_map,
        inter_stage_lags=inter_stage_lags,
        conversion_rates=effective_rates_b,
        lag_assumption_model=None
    )

    return {
        'error': False,
        'period_a': {
            'results': results_a,
            'rates': effective_rates_a,
            'rates_desc': rates_desc_a
        },
        'period_b': {
            'results': results_b,
            'rates': effective_rates_b,
            'rates_desc': rates_desc_b
        },
        'type': comparison_type,
        'validation': validation
    }
