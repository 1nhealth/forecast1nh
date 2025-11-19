"""
AI-Powered Insights Module for Clinical Trial Recruitment Dashboard

SECURITY & PRIVACY NOTES:
- Insights cached in session state (isolated per user, cleared on session end)
- Data hash validation ensures insights regenerate when data changes
- Only AGGREGATED metrics sent to Gemini API (no raw PII/PHI)
- Cache explicitly cleared on new data upload
- Streamlit sessions are isolated - no cross-user data leakage
- Cache persists ONLY during active browser session
"""

import hashlib
import json
import streamlit as st
import pandas as pd
import google.generativeai as genai


def compute_data_hash(data_dict: dict) -> str:
    """
    Compute hash of data to detect changes.
    Converts DataFrames and other types to JSON for consistent hashing.

    Args:
        data_dict: Dictionary containing data to hash

    Returns:
        SHA256 hash string
    """
    serializable = {}
    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            # Use only top/bottom rows for hash (not full DF)
            serializable[key] = value.head(10).to_dict()
        elif isinstance(value, (list, dict, str, int, float, bool)):
            serializable[key] = value
        elif pd.isna(value):
            serializable[key] = None
        else:
            serializable[key] = str(value)

    # Create stable JSON string and hash
    json_str = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_cached_or_generate_insights(
    page_name: str,
    data_dict: dict,
    generate_func: callable,
    focus: str = "ICF",
    force_regenerate: bool = False
) -> str:
    """
    Get cached insights if data unchanged, otherwise regenerate.

    Args:
        page_name: Unique identifier (e.g., 'site_performance')
        data_dict: Data to analyze (used for hashing)
        generate_func: Function to call for generation
        focus: Analysis focus - "ICF" or "Enrollment"
        force_regenerate: Skip cache and regenerate

    Returns:
        AI-generated insights text
    """
    cache_key = f"{page_name}_ai_insights_{focus}"
    hash_key = f"{page_name}_data_hash_{focus}"

    # Compute current data hash
    current_hash = compute_data_hash(data_dict)

    # Check if we have valid cached insights
    if not force_regenerate:
        if cache_key in st.session_state and hash_key in st.session_state:
            if st.session_state[hash_key] == current_hash:
                # Data unchanged - return cached insights
                return st.session_state[cache_key]

    # Generate fresh insights
    insights = generate_func(data_dict)

    # Cache results with hash
    st.session_state[cache_key] = insights
    st.session_state[hash_key] = current_hash

    return insights


def clear_all_ai_caches():
    """Clear all AI-related caches from session state."""
    ai_keys = [
        k for k in list(st.session_state.keys())
        if '_ai_insights' in k or '_data_hash' in k
    ]
    for key in ai_keys:
        del st.session_state[key]


def clear_page_ai_cache(page_name: str):
    """Clear cache for specific page."""
    cache_key = f"{page_name}_ai_insights"
    hash_key = f"{page_name}_data_hash"

    if cache_key in st.session_state:
        del st.session_state[cache_key]
    if hash_key in st.session_state:
        del st.session_state[hash_key]


def get_cache_status() -> dict:
    """Get status of all AI caches."""
    pages = ['site_performance', 'ad_performance', 'pc_performance', 'funnel_analysis']
    status = {}

    for page in pages:
        cache_key = f"{page}_ai_insights"
        status[page] = cache_key in st.session_state

    return status


# ============================================================================
# DATA SANITIZATION FUNCTIONS
# ============================================================================

def sanitize_site_data(ranked_sites_df: pd.DataFrame, focus: str = "ICF", top_n: int = 5, bottom_n: int = 5) -> dict:
    """
    Extract aggregated site metrics for AI analysis using intelligent filtering and mean-based comparison.

    Features:
    - Dynamic minimum threshold based on median StS count (scales with study size)
    - Mean-based site selection (above/below mean on focus metric)
    - Filters out sites with insufficient data
    - Prevents overlap when total sites < (top_n + bottom_n)

    Args:
        ranked_sites_df: DataFrame with ranked site performance
        focus: "ICF" or "Enrollment" - determines which metric to use for comparison
        top_n: Number of top performers to include (above mean)
        bottom_n: Number of bottom performers to include (below mean)

    Returns:
        Dictionary with aggregated metrics and filtering metadata
    """
    if ranked_sites_df is None or len(ranked_sites_df) == 0:
        return {'error': 'No site data available'}

    # Get available columns safely
    def safe_mean(col_name, default=0):
        if col_name in ranked_sites_df.columns:
            return ranked_sites_df[col_name].mean()
        return default

    def safe_get(row, col_name, default=0):
        """Safely get a value from a row, converting to float if needed."""
        if col_name not in ranked_sites_df.columns:
            return default
        val = row.get(col_name, default)
        if pd.isna(val):
            return default
        try:
            return float(val)
        except:
            return default

    # Define all operational metrics to extract (NO scores or grades)
    operational_metrics = [
        'Total Qualified', 'StS Count', 'Appt Count', 'ICF Count', 'Enrollment Count',
        'Qualified to StS %', 'Qualified to Appt %', 'Qualified to ICF %', 'Qualified to Enrollment %',
        'StS to Appt %', 'StS to ICF %', 'StS to Enrollment %', 'ICF to Enrollment %',
        'Average time to first site action', 'Avg time from StS to Appt Sched.',
        'Avg time from StS to ICF', 'Avg time from StS to Enrollment',
        'StS Contact Rate %', 'SF or Lost After ICF %', 'StS to Lost %'
    ]

    # DYNAMIC THRESHOLD CALCULATION
    # Use median StS count to determine appropriate minimum (scales with study size)
    median_sts = ranked_sites_df['StS Count'].median() if 'StS Count' in ranked_sites_df.columns else 10
    min_sts_threshold = max(3, int(median_sts * 0.20))  # At least 3, or 20% of median

    # Filter sites with sufficient data
    filtered_df = ranked_sites_df[ranked_sites_df['StS Count'] >= min_sts_threshold].copy()

    if len(filtered_df) == 0:
        # Fallback: if no sites meet threshold, use all sites with at least 1 StS
        min_sts_threshold = 1
        filtered_df = ranked_sites_df[ranked_sites_df['StS Count'] >= 1].copy()

    # Determine focus metric based on focus parameter
    if focus == "ICF":
        focus_metric = 'StS to ICF %'
    else:  # Enrollment
        focus_metric = 'ICF to Enrollment %'

    # Calculate mean of focus metric for filtered sites
    if focus_metric in filtered_df.columns:
        mean_focus_value = filtered_df[focus_metric].mean()
    else:
        mean_focus_value = 0

    # Split sites into above/below mean
    above_mean_df = filtered_df[filtered_df[focus_metric] > mean_focus_value].copy()
    below_mean_df = filtered_df[filtered_df[focus_metric] < mean_focus_value].copy()

    # Sort by focus metric value (not composite score)
    above_mean_df = above_mean_df.sort_values(focus_metric, ascending=False)
    below_mean_df = below_mean_df.sort_values(focus_metric, ascending=True)

    # Prevent overlap when total sites < (top_n + bottom_n)
    total_filtered = len(filtered_df)
    if total_filtered < (top_n + bottom_n):
        # Reduce to half of available sites each
        adjusted_n = max(1, total_filtered // 2)
        top_n = min(top_n, adjusted_n)
        bottom_n = min(bottom_n, adjusted_n)

    # Select top performers (above mean, highest values)
    top_sites_df = above_mean_df.head(min(top_n, len(above_mean_df)))

    # Select bottom performers (below mean, lowest values)
    bottom_sites_df = below_mean_df.head(min(bottom_n, len(below_mean_df)))

    # Build top sites summary with ALL operational metrics
    top_sites = []
    for _, row in top_sites_df.iterrows():
        site_info = {'site_name': row.get('Site', 'Unknown')}
        for metric in operational_metrics:
            site_info[metric] = safe_get(row, metric, 0)
        top_sites.append(site_info)

    # Build bottom sites summary with ALL operational metrics
    bottom_sites = []
    for _, row in bottom_sites_df.iterrows():
        site_info = {'site_name': row.get('Site', 'Unknown')}
        for metric in operational_metrics:
            site_info[metric] = safe_get(row, metric, 0)
        bottom_sites.append(site_info)

    # Calculate averages for all operational metrics (using FILTERED data)
    avg_metrics = {}
    for metric in operational_metrics:
        if metric in filtered_df.columns:
            avg_metrics[f'avg_{metric}'] = float(filtered_df[metric].mean())
        else:
            avg_metrics[f'avg_{metric}'] = 0

    return {
        'top_sites': top_sites,
        'bottom_sites': bottom_sites,
        'total_sites': len(filtered_df),
        'total_sites_unfiltered': len(ranked_sites_df),
        'sites_excluded': len(ranked_sites_df) - len(filtered_df),
        'min_sts_threshold': min_sts_threshold,
        'median_sts': float(median_sts),
        'focus_metric': focus_metric,
        'mean_focus_value': float(mean_focus_value),
        'sites_above_mean': len(above_mean_df),
        'sites_below_mean': len(below_mean_df),
        'sites_at_mean': total_filtered - len(above_mean_df) - len(below_mean_df),
        **avg_metrics  # Unpack all average metrics
    }


def sanitize_ad_data(ranked_ads_df: pd.DataFrame, top_n: int = 5, bottom_n: int = 5) -> dict:
    """
    Extract aggregated ad performance metrics for AI analysis.

    Args:
        ranked_ads_df: DataFrame with ranked ad source performance
        top_n: Number of top performers to include
        bottom_n: Number of bottom performers to include

    Returns:
        Dictionary with aggregated metrics
    """
    if ranked_ads_df is None or len(ranked_ads_df) == 0:
        return {'error': 'No ad data available'}

    # Determine source column name (could be 'UTM Source', 'source', 'utm_source', or similar)
    source_col = None
    for col in ['UTM Source', 'source', 'utm_source', 'ad_source', 'Source']:
        if col in ranked_ads_df.columns:
            source_col = col
            break

    if source_col is None:
        source_col = ranked_ads_df.columns[0]  # Fallback to first column

    def safe_get(row, col_name, default=0):
        """Safely get a value from a row, converting to appropriate type."""
        if col_name not in ranked_ads_df.columns:
            return default
        val = row.get(col_name, default)
        if pd.isna(val):
            return default
        try:
            return float(val)
        except:
            return default

    def safe_mean(col_name, default=0):
        if col_name in ranked_ads_df.columns:
            return ranked_ads_df[col_name].mean()
        return default

    # Define all operational metrics to extract (NO scores or grades)
    operational_metrics = [
        'Total Qualified', 'StS Count', 'Appt Count', 'ICF Count', 'Enrollment Count',
        'Qualified to StS %', 'StS to Appt %', 'Qualified to ICF %', 'Qualified to Enrollment %', 'ICF to Enrollment %',
        'Average time to first site action', 'Avg time from StS to Appt Sched.',
        'Screen Fail % (from Qualified)'
    ]

    # Build top sources summary with ALL operational metrics
    top_sources = []
    for _, row in ranked_ads_df.head(top_n).iterrows():
        source_info = {'source': str(row.get(source_col, 'Unknown'))}
        for metric in operational_metrics:
            source_info[metric] = safe_get(row, metric, 0)
        top_sources.append(source_info)

    # Build bottom sources summary with ALL operational metrics
    bottom_sources = []
    for _, row in ranked_ads_df.tail(bottom_n).iterrows():
        source_info = {'source': str(row.get(source_col, 'Unknown'))}
        for metric in operational_metrics:
            source_info[metric] = safe_get(row, metric, 0)
        bottom_sources.append(source_info)

    # Calculate averages for all operational metrics
    avg_metrics = {}
    for metric in operational_metrics:
        avg_metrics[f'avg_{metric}'] = float(safe_mean(metric, 0))

    return {
        'top_sources': top_sources,
        'bottom_sources': bottom_sources,
        'total_sources': len(ranked_ads_df),
        **avg_metrics  # Unpack all average metrics
    }


def sanitize_pc_data(pc_metrics_df: pd.DataFrame, top_n: int = 3, bottom_n: int = 3) -> dict:
    """
    Extract aggregated PC performance metrics for AI analysis.
    This handles effectiveness dataframes (TTFC or Contact Attempts), not individual PC rankings.

    Args:
        pc_metrics_df: DataFrame with PC effectiveness metrics (either ttfc_df or attempt_effectiveness_df)
        top_n: Number of best-performing segments to include
        bottom_n: Number of worst-performing segments to include

    Returns:
        Dictionary with aggregated metrics
    """
    if pc_metrics_df is None or len(pc_metrics_df) == 0:
        return {'error': 'No PC data available'}

    def safe_get(row, col_name, default=0):
        """Safely get a value from a row."""
        if col_name not in pc_metrics_df.columns:
            return default
        val = row.get(col_name, default)
        if pd.isna(val):
            return default
        try:
            return float(val)
        except:
            return default

    # Determine what type of effectiveness data this is
    if 'Time to First Contact' in pc_metrics_df.columns:
        # TTFC Effectiveness data
        segment_col = 'Time to First Contact'
        metrics_type = 'ttfc'
    elif 'Number of Attempts' in pc_metrics_df.columns:
        # Contact Attempt Effectiveness data
        segment_col = 'Number of Attempts'
        metrics_type = 'attempts'
    else:
        # Unknown format
        segment_col = pc_metrics_df.columns[0]
        metrics_type = 'unknown'

    # Operational metrics available in both types
    operational_metrics = ['Attempts', 'Total Referrals', 'Total Sent to Site', 'StS Rate',
                          'Total ICFs', 'ICF Rate', 'Total Enrollments', 'Enrollment Rate']

    # Map alternate column names
    col_map = {
        'Total_StS': 'Total Sent to Site',
        'StS_Rate': 'StS Rate',
        'Total_ICF': 'Total ICFs',
        'ICF_Rate': 'ICF Rate',
        'Total_Enrolled': 'Total Enrollments',
        'Enrollment_Rate': 'Enrollment Rate'
    }

    # Extract all segments with their metrics (not just top/bottom)
    all_segments = []
    for _, row in pc_metrics_df.iterrows():
        segment_info = {
            'segment': str(row.get(segment_col, 'Unknown')),
            'metrics_type': metrics_type
        }

        # Try each metric with potential alternate names
        for metric in operational_metrics:
            # Try direct column name first
            if metric in pc_metrics_df.columns:
                segment_info[metric] = safe_get(row, metric, 0)
            # Try mapped name
            elif metric in col_map.values():
                # Find original name
                orig_name = [k for k, v in col_map.items() if v == metric]
                if orig_name and orig_name[0] in pc_metrics_df.columns:
                    segment_info[metric] = safe_get(row, orig_name[0], 0)
                else:
                    segment_info[metric] = 0
            else:
                segment_info[metric] = 0

        all_segments.append(segment_info)

    return {
        'metrics_type': metrics_type,
        'segment_column': segment_col,
        'all_segments': all_segments,
        'total_segments': len(all_segments)
    }


def sanitize_funnel_data(funnel_results: dict) -> dict:
    """
    Extract aggregated funnel projection metrics for AI analysis.

    Args:
        funnel_results: Dictionary from calculate_pipeline_projection()
                       Contains: results_df, total_icf_yield, total_enroll_yield

    Returns:
        Dictionary with aggregated projection metrics
    """
    if funnel_results is None or len(funnel_results) == 0:
        return {'error': 'No funnel data available'}

    results_df = funnel_results.get('results_df', pd.DataFrame())

    if results_df.empty:
        return {'error': 'No projection results available'}

    # Extract monthly projection data
    monthly_icf = results_df['Projected_ICF_Landed'].tolist() if 'Projected_ICF_Landed' in results_df else []
    monthly_enroll = results_df['Projected_Enrollments_Landed'].tolist() if 'Projected_Enrollments_Landed' in results_df else []

    return {
        'total_icf_yield': float(funnel_results.get('total_icf_yield', 0)),
        'total_enroll_yield': float(funnel_results.get('total_enroll_yield', 0)),
        'num_months_projected': len(results_df),
        'peak_monthly_icf': float(max(monthly_icf)) if monthly_icf else 0,
        'peak_monthly_enrollment': float(max(monthly_enroll)) if monthly_enroll else 0,
        'avg_monthly_icf': float(sum(monthly_icf) / len(monthly_icf)) if monthly_icf else 0,
        'avg_monthly_enrollment': float(sum(monthly_enroll) / len(monthly_enroll)) if monthly_enroll else 0,
        'final_cumulative_icf': float(results_df['Cumulative_ICF_Landed'].iloc[-1]) if 'Cumulative_ICF_Landed' in results_df and len(results_df) > 0 else 0,
        'final_cumulative_enrollment': float(results_df['Cumulative_Enrollments_Landed'].iloc[-1]) if 'Cumulative_Enrollments_Landed' in results_df and len(results_df) > 0 else 0,
    }


# ============================================================================
# AI GENERATION FUNCTIONS
# ============================================================================

def generate_site_insights(sanitized_data: dict, focus: str = "ICF") -> str:
    """
    Generate AI insights for site performance.

    Args:
        sanitized_data: Aggregated site metrics
        focus: Analysis focus - "ICF" or "Enrollment"

    Returns:
        AI-generated insights text
    """
    if 'error' in sanitized_data:
        return f"Unable to generate insights: {sanitized_data['error']}"

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=genai.GenerationConfig(temperature=0.2)
        )

        if focus == "ICF":
            prompt = f"""You are analyzing site ICF performance data.

DATA FILTERING APPLIED:
- Minimum StS threshold: {sanitized_data.get('min_sts_threshold', 10)} (dynamically calculated as 20% of median StS: {sanitized_data.get('median_sts', 0):.0f})
- Sites analyzed: {sanitized_data['total_sites']} of {sanitized_data.get('total_sites_unfiltered', 0)} total
- Sites excluded (insufficient data): {sanitized_data.get('sites_excluded', 0)}
- Focus metric: {sanitized_data.get('focus_metric', 'StS to ICF %')}

STUDY AVERAGES (filtered sites only):
- Study Mean {sanitized_data.get('focus_metric', 'StS to ICF %')}: {sanitized_data.get('mean_focus_value', 0):.1%}
- Sites above mean: {sanitized_data.get('sites_above_mean', 0)}
- Sites below mean: {sanitized_data.get('sites_below_mean', 0)}
- Study Average StS to ICF %: {sanitized_data.get('avg_StS to ICF %', 0):.1%}
- Study Average StS to Appt %: {sanitized_data.get('avg_StS to Appt %', 0):.1%}
- Study Average time to first site action: {sanitized_data.get('avg_Average time to first site action', 0):.1f} days

ABOVE MEAN PERFORMERS (sorted by {sanitized_data.get('focus_metric', 'StS to ICF %')}):
{json.dumps(sanitized_data['top_sites'], indent=2)}

BELOW MEAN PERFORMERS (sorted by {sanitized_data.get('focus_metric', 'StS to ICF %')}):
{json.dumps(sanitized_data['bottom_sites'], indent=2)}

INSTRUCTIONS:
- StS means "Sent to Site" (NOT "screened to site")
- Use ### for section headers
- Convert decimal days to readable format (e.g., "2.5 days" = "2 days 12 hours")
- Calculate ALL numbers from the data above - do NOT use example numbers
- Start directly with section 1 below - NO intro text

YOUR ANALYSIS STRUCTURE:

### 1. Key Metrics Driving ICF Performance
Identify the 2-3 metrics that most strongly differentiate high vs low performing sites.
Compare above-mean performers to below-mean performers with actual numbers from the data.
Example: "Above-mean sites average 45% StS to ICF vs below-mean sites 12%"

### 2. Distribution vs Study Mean
The study mean StS to ICF rate is {sanitized_data.get('mean_focus_value', 0):.1%}.
You have {len(sanitized_data.get('top_sites', []))} sites above mean and {len(sanitized_data.get('bottom_sites', []))} sites below mean.
List sites by name in each category.
Format: "X sites above mean: [list names]. Y sites below mean: [list names]"

### 3. Top Site Concentration
Calculate what % of total study ICFs come from the above-mean performing sites.
List these sites by name with their ICF counts.
Show the math: "Above-mean sites: X ICFs ÷ Y total ICFs = Z%"

### 4. Performance Gap Analysis
Compare above-mean sites vs below-mean sites on key metrics:
- StS to ICF %
- StS to Appt %
- Average time to first site action
- Any other relevant metrics in the data

Identify 2-3 specific gaps where below-mean sites could improve.
Use actual numbers: "Below-mean sites: X%, Above-mean sites: Y%, Gap: Z%"

### 5. Improvement Impact Calculation
If the below-mean sites improved their StS to ICF % to match the study mean of {sanitized_data.get('mean_focus_value', 0):.1%}:
- Calculate their current total ICFs
- Calculate what they would generate at the study mean rate
- Show the gain in additional ICFs

Format: "Below-mean sites current: X ICFs. At study mean: Y ICFs. Gain: Z additional ICFs"

CRITICAL: Use ONLY the actual data provided above. Show all calculations clearly."""

        else:  # focus == "Enrollment"
            prompt = f"""You are analyzing site Enrollment performance data.

DATA FILTERING APPLIED:
- Minimum StS threshold: {sanitized_data.get('min_sts_threshold', 10)} (dynamically calculated as 20% of median StS: {sanitized_data.get('median_sts', 0):.0f})
- Sites analyzed: {sanitized_data['total_sites']} of {sanitized_data.get('total_sites_unfiltered', 0)} total
- Sites excluded (insufficient data): {sanitized_data.get('sites_excluded', 0)}
- Focus metric: {sanitized_data.get('focus_metric', 'ICF to Enrollment %')}

STUDY AVERAGES (filtered sites only):
- Study Mean {sanitized_data.get('focus_metric', 'ICF to Enrollment %')}: {sanitized_data.get('mean_focus_value', 0):.1%}
- Sites above mean: {sanitized_data.get('sites_above_mean', 0)}
- Sites below mean: {sanitized_data.get('sites_below_mean', 0)}
- Study Average ICF to Enrollment %: {sanitized_data.get('avg_ICF to Enrollment %', 0):.1%}
- Study Average SF or Lost After ICF %: {sanitized_data.get('avg_SF or Lost After ICF %', 0):.1%}

ABOVE MEAN PERFORMERS (sorted by {sanitized_data.get('focus_metric', 'ICF to Enrollment %')}):
{json.dumps(sanitized_data['top_sites'], indent=2)}

BELOW MEAN PERFORMERS (sorted by {sanitized_data.get('focus_metric', 'ICF to Enrollment %')}):
{json.dumps(sanitized_data['bottom_sites'], indent=2)}

INSTRUCTIONS:
- StS means "Sent to Site" (NOT "screened to site")
- Use ### for section headers
- Convert decimal days to readable format (e.g., "2.5 days" = "2 days 12 hours")
- Calculate ALL numbers from the data above - do NOT use example numbers
- Start directly with section 1 below - NO intro text

### 1. Key Metrics Driving Enrollment Performance
Identify the 2-3 metrics that most strongly differentiate high vs low performing sites.
Compare above-mean performers to below-mean performers with actual numbers from the data.
Example: "Above-mean sites average 85% ICF to Enrollment vs below-mean sites 45%"

### 2. Distribution vs Study Mean
The study mean ICF to Enrollment rate is {sanitized_data.get('mean_focus_value', 0):.1%}.
You have {len(sanitized_data.get('top_sites', []))} sites above mean and {len(sanitized_data.get('bottom_sites', []))} sites below mean.
List sites by name in each category.
Format: "X sites above mean: [list names]. Y sites below mean: [list names]"

### 3. Top Site Concentration
Calculate what % of total study Enrollments come from the above-mean performing sites.
List these sites by name with their Enrollment counts.
Show the math: "Above-mean sites: X Enrollments ÷ Y total Enrollments = Z%"

### 4. Performance Gap Analysis
Compare above-mean sites vs below-mean sites on key metrics:
- ICF to Enrollment %
- SF or Lost After ICF % (screen fail rate)
- Any other relevant metrics in the data

Identify 2-3 specific gaps where below-mean sites could improve.
Use actual numbers: "Below-mean sites: X%, Above-mean sites: Y%, Gap: Z%"

### 5. Improvement Impact Calculation
If the below-mean sites improved their ICF to Enrollment % to match the study mean of {sanitized_data.get('mean_focus_value', 0):.1%}:
- Calculate their current total Enrollments
- Calculate what they would generate at the study mean rate
- Show the gain in additional Enrollments

Format: "Below-mean sites current: X Enrollments. At study mean: Y Enrollments. Gain: Z additional Enrollments"

CRITICAL: Use ONLY the actual data provided above. Show all calculations clearly."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating insights: {str(e)}\n\nPlease check your GEMINI_API_KEY in .streamlit/secrets.toml"


def generate_ad_insights(sanitized_data: dict, focus: str = "ICF") -> str:
    """
    Generate AI insights for ad performance.

    Args:
        sanitized_data: Aggregated ad metrics
        focus: Analysis focus - "ICF" or "Enrollment"

    Returns:
        AI-generated insights text
    """
    if 'error' in sanitized_data:
        return f"Unable to generate insights: {sanitized_data['error']}"

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=genai.GenerationConfig(temperature=0.2)
        )

        if focus == "ICF":
            prompt = f"""You are a Chief Data Scientist analyzing ad sources to maximize ICF conversions.

FOCUS: DRIVING ICF CONVERSIONS
Identify which ad sources deliver the highest quality leads that convert to ICF.

AGGREGATED AD METRICS (OPERATIONAL AVERAGES):
- Total Ad Sources: {sanitized_data['total_sources']}
- Average Qualified to ICF %: {sanitized_data.get('avg_Qualified to ICF %', 0):.1%}
- Average StS to Appt %: {sanitized_data.get('avg_StS to Appt %', 0):.1%}
- Average Total Qualified per Source: {sanitized_data.get('avg_Total Qualified', 0):.1f}
- Average ICF Count per Source: {sanitized_data.get('avg_ICF Count', 0):.1f}

TOP PERFORMING AD SOURCES (by ICF metrics):
{json.dumps(sanitized_data['top_sources'], indent=2)}

LOWEST PERFORMING AD SOURCES (by ICF metrics):
{json.dumps(sanitized_data['bottom_sources'], indent=2)}

CRITICAL TERMINOLOGY:
StS = "Sent to Site" (NOT "screened to site" or "screen to site")
ALWAYS use "Sent to Site" or "StS" - NEVER use incorrect terminology.

TIME FORMAT RULE:
Convert all decimal days/times to human-readable format in your analysis.

FORMATTING RULE:
Use ### for all section headers (NOT # or ##)
✅ GOOD: "### 1. EXECUTIVE OVERVIEW"
❌ BAD: "# 1. EXECUTIVE OVERVIEW" or "## 1. EXECUTIVE OVERVIEW"

DATA SOURCES TO ANALYZE:
You have access to ad source performance data - analyze ALL available metrics:

1. Ad Source Ranking Table: All operational conversion metrics (NO scores/grades)
   - ICF conversion rates (Qualified to ICF %, StS to Appt %, etc.)
   - Volume metrics (Total Qualified, StS Count, ICF Count)
   - Quality vs quantity trade-offs
   - Any cost or efficiency metrics visible

REQUIRED: Analyze quality vs quantity trade-offs with specific ROI calculations

KEY METRICS FOR ICF SUCCESS:
- Qualified to ICF % (primary ICF quality metric)
- StS to Appt % (appointment show-up rate driving ICF)
- Volume of ICFs delivered per source
- Quality vs quantity: high ICF rate with lower volume vs low ICF rate with high volume

IGNORE FOR THIS ANALYSIS:
- ICF to Enrollment rates (post-ICF conversion, not ICF-relevant)
- Screen fail after ICF (not relevant to reaching ICF)

DATA-DRIVEN ANALYSIS RULES FOR ICF:
1. USE DISTRIBUTION-BASED COMPARISONS (calculate ALL statistics from actual data):
   - ✅ GOOD: "The average Qualified to ICF rate is [calculate from actual data], with [calculate %] of ad sources falling below this average"
   - ✅ GOOD: "Source X's [actual from data] Qualified→ICF rate places them in the top [calculate %] of all sources"
   - ✅ GOOD: "The median (50th percentile) is [calculate], meaning half of sources perform above and half below"
   - ❌ BAD: "Sources average 35% ICF conversion, indicating significant room for improvement" (no baseline context)
   - ❌ BAD: "The median is 35%, with 60% of sources below this threshold" (WRONG - median = 50th percentile by definition!)
   - CRITICAL: CALCULATE all statistics from the actual data provided - NEVER echo example numbers from these instructions
   - STATISTICAL ACCURACY: Median = 50th percentile (exactly 50% above, 50% below). Average/mean can have any distribution.
   - REQUIRED: Use average/mean with distribution context ("X% of sources fall below the average of Y%")
   - FORBIDDEN: Saying "room for improvement" without baseline comparison

2. IDENTIFY CONCENTRATION & TOP PERFORMERS:
   - ✅ GOOD: "The top 3 ad sources (list names) with the highest Qualified→ICF rates are driving 72% of the study's total ICFs"
   - ✅ GOOD: "2 high-quality sources account for 65% of ICF volume, while bottom 8 sources contribute only 15%"
   - ❌ BAD: "Higher ICF conversion rates lead to more ICFs" (obvious, no concentration analysis)
   - REQUIRED: Identify which sources drive majority of ICF results
   - REQUIRED: Multi-factor causation for why top sources succeed (quality + volume analysis)

3. MULTI-FACTOR CAUSATION (NOT single correlations):
   - ✅ GOOD: "The top 3 sources driving 72% of ICFs share these factors: (1) Qualified→ICF rates >50% vs study median of 35%, (2) StS to Appt rates >60% vs median of 45%, (3) moderate-high volume (150-250 qualified/month) providing both quality AND scale"
   - ❌ BAD: "Quality sources perform better than volume sources" (single dimension, vague)
   - REQUIRED: Link multiple metrics (conversion quality + volume + efficiency) to explain performance

4. Cite specific ICF rates: "Source X: 52% Qualified→ICF vs Source Y: 18%"
5. Show ICF causality: "High-quality Source A delivers 2.3x ICF rate vs high-volume Source B"
6. Calculate ICF ROI: "Source X delivers $Y cost-per-ICF vs study median $Z"
7. Quantify ICF reallocation: "Shifting 20% budget from X to Y would yield Z more ICFs"
8. Use ACTUAL source names from the data
9. Show ICF-specific calculations
10. FIND TIPPING POINTS: Identify quality/volume thresholds for optimal ICF generation
   - ✅ GOOD: "Sources with >50% Qualified→ICF rate deliver 2.8x more ICFs per dollar vs <30% sources (50% is efficiency threshold)"
   - ✅ GOOD: "Medium-volume high-quality sources (100-200 qualified/month at >45% ICF rate) outperform high-volume low-quality (300+ at <25%)"
   - ❌ BAD: "Quality sources perform better" (no threshold, not actionable)

FORBIDDEN OBVIOUS INSIGHTS (these provide NO value):
❌ "Higher ICF conversion rates lead to more ICFs" (circular logic)
❌ "More qualified referrals lead to more ICFs" (volume correlation)
❌ "Quality correlates with ICF success" (vague, no threshold)
❌ Any statement without specific quality/volume trade-off analysis

CRITICAL DATA AVAILABILITY RULE:
NEVER speculate about missing data or mention what you DON'T have.
❌ FORBIDDEN: "While the data doesn't provide cost metrics..."
❌ FORBIDDEN: "Further data would likely reveal..."
❌ FORBIDDEN: "The provided data doesn't explicitly detail..."
❌ FORBIDDEN: "Although we lack information on..."
✅ REQUIRED: ONLY analyze metrics that ARE provided in the data
✅ REQUIRED: If a metric isn't in the data, don't mention it at all
✅ REQUIRED: Work with what you HAVE, not what you wish you had

CRITICAL: When recommending budget shifts, provide SPECIFIC, NUMBERED ACTION PLANS with ROI calculations
❌ BAD: "Consider increasing digital ad spend"
❌ BAD: "Reallocate budget to higher-performing sources"
✅ GOOD: "The key opportunity areas for ICF improvement are: reallocating budget from low-quality/high-volume to high-quality/moderate-volume sources. The top 3 sources (list names) driving 72% of ICFs have median Qualified→ICF of 58% vs bottom 5 sources at 18%. Recommended actions:
1. Reallocate 30% of Facebook budget ($12K/month) to physician_referral source
   - Facebook: 320 qualified × 28% = 90 ICFs ($450 per ICF)
   - Physician: 180 qualified × 67% = 121 ICFs ($260 per ICF)
2. Expected Facebook impact: 320→224 qualified (-27 ICFs)
3. Expected physician impact: 180→275 qualified (+64 ICFs)
4. Net result: +37 ICFs monthly (+41% improvement) at 42% lower cost-per-ICF, achieving study median efficiency"

REQUIRED ELEMENTS FOR ACTION PLANS:
- List specific underperforming sources by name
- Reference study median/percentiles for benchmarking (NOT invented targets)
- Provide numbered, specific actions with full ROI calculations
- Quantify expected impact: "would result in X additional ICFs per month at Y% lower cost"
- Multi-source reallocation analysis (not single-source focus)

EXAMPLE OF EXCELLENT ICF INSIGHT WITH ROI ANALYSIS:
"Sources with >50% Qualified→ICF rate deliver 2.8x cost efficiency vs <30% sources (50% is quality threshold). utm_source=physician_referral: 67% ICF rate, 180 qualified/month = 121 ICFs. utm_source=facebook: 28% ICF rate, 320 qualified/month = 90 ICFs. Despite Facebook's 78% higher volume, physician source delivers 34% more ICFs.

PRESCRIPTION for budget optimization:
(a) Shift 30% of Facebook budget ($12K/month) to physician outreach expansion
(b) Expected Facebook impact: 320→224 qualified (-27 ICFs)
(c) Expected physician impact: 180→275 qualified (+64 ICFs)
Net result: +37 ICFs monthly (+41% improvement) at 42% lower cost-per-ICF (from $450 to $260)."

EXAMPLE OF BAD ICF INSIGHT:
❌ "Consider increasing digital ad spend" (vague, no ICF impact data, no ROI)
❌ "Quality sources perform better" (no threshold, no actionable recommendation)

CRITICAL INSTRUCTION:
Focus EXCLUSIVELY on ICF-driving operational metrics. NO scores or rankings.

IMPORTANT: Start your response DIRECTLY with section 1 below. NO introductory sentences.

Provide data-driven insights focused EXCLUSIVELY on ICF:

1. EXECUTIVE OVERVIEW (2-3 sentences)
   - Quantify ICF conversion rates and volumes across ad sources
   - Identify quality vs. volume patterns in ICF generation
   - NO mention of enrollment

2. ICF SUCCESS DRIVERS (Top 3 ad sources/patterns driving ICF)
   - Compare ICF conversion rates: "Source A: 67% Qualified→ICF vs Source B: 23%"
   - Quality patterns: "Source X delivers 52% ICF rate despite 40% lower volume"
   - Calculate cost-per-ICF efficiency

3. ICF OPPORTUNITIES (Top 3 quantified ICF improvement areas)
   - Identify low-ICF sources vs high-ICF sources with SPECIFIC numbers
   - Calculate ICF reallocation impact: "Moving $X from Source A to B would yield N additional ICFs"
   - Quantify which sources to scale for ICF maximization

4. FORWARD OUTLOOK (2-3 sentences)
   - Calculate optimal budget mix for ICF generation
   - Quantify ICF impact of recommended reallocations
   - Provide specific percentages and expected ICF yields

Remember: ACT AS A DATA SCIENTIST. Focus on ICF ROI, not generic marketing advice."""

        else:  # focus == "Enrollment"
            prompt = f"""You are a Chief Data Scientist analyzing ad sources to maximize Enrollment conversions.

FOCUS: DRIVING ENROLLMENT CONVERSIONS
Identify which ad sources deliver leads that successfully complete enrollment.

AGGREGATED AD METRICS (OPERATIONAL AVERAGES):
- Total Ad Sources: {sanitized_data['total_sources']}
- Average Qualified to Enrollment %: {sanitized_data.get('avg_Qualified to Enrollment %', 0):.1%}
- Average ICF to Enrollment %: {sanitized_data.get('avg_ICF to Enrollment %', 0):.1%}
- Average Total Qualified per Source: {sanitized_data.get('avg_Total Qualified', 0):.1f}
- Average Enrollment Count per Source: {sanitized_data.get('avg_Enrollment Count', 0):.1f}

TOP PERFORMING AD SOURCES (by Enrollment metrics):
{json.dumps(sanitized_data['top_sources'], indent=2)}

LOWEST PERFORMING AD SOURCES (by Enrollment metrics):
{json.dumps(sanitized_data['bottom_sources'], indent=2)}

CRITICAL TERMINOLOGY:
StS = "Sent to Site" (NOT "screened to site" or "screen to site")
ALWAYS use "Sent to Site" or "StS" - NEVER use incorrect terminology.

TIME FORMAT RULE:
Convert all decimal days/times to human-readable format in your analysis.

FORMATTING RULE:
Use ### for all section headers (NOT # or ##)
✅ GOOD: "### 1. EXECUTIVE OVERVIEW"
❌ BAD: "# 1. EXECUTIVE OVERVIEW" or "## 1. EXECUTIVE OVERVIEW"

DATA SOURCES TO ANALYZE:
You have access to ad source performance data - analyze ALL available metrics:

1. Ad Source Ranking Table: All operational conversion metrics (NO scores/grades)
   - Enrollment conversion rates (Qualified to Enrollment %, ICF to Enrollment %, etc.)
   - Volume metrics (Total Qualified, ICF Count, Enrollment Count)
   - Screen fail rates by source
   - Quality vs quantity trade-offs for enrollment

REQUIRED: Analyze quality vs quantity trade-offs with specific enrollment ROI calculations

KEY METRICS FOR ENROLLMENT SUCCESS:
- Qualified to Enrollment % (complete funnel enrollment rate)
- ICF to Enrollment % (post-ICF enrollment success)
- Volume of enrollments delivered per source
- Screen fail rates (enrollment blockers)

IGNORE FOR THIS ANALYSIS:
- StS to Appointment rates (pre-enrollment step, not enrollment-relevant)
- Initial contact metrics (not enrollment-specific)

DATA-DRIVEN ANALYSIS RULES FOR ENROLLMENT:
1. USE DISTRIBUTION-BASED COMPARISONS (calculate ALL statistics from actual data):
   - ✅ GOOD: "The average Qualified to Enrollment rate is [calculate from actual data], with [calculate %] of ad sources falling below this average"
   - ✅ GOOD: "Source X's [actual from data] Qualified→Enrollment rate places them in the top [calculate %] of all sources"
   - ✅ GOOD: "The median (50th percentile) is [calculate], meaning half of sources perform above and half below"
   - ❌ BAD: "Sources average 28% enrollment rate, indicating significant room for improvement" (no baseline context)
   - ❌ BAD: "The median is 28%, with 68% of sources below this threshold" (WRONG - median = 50th percentile by definition!)
   - CRITICAL: CALCULATE all statistics from the actual data provided - NEVER echo example numbers from these instructions
   - STATISTICAL ACCURACY: Median = 50th percentile (exactly 50% above, 50% below). Average/mean can have any distribution.
   - REQUIRED: Use average/mean with distribution context ("X% of sources fall below the average of Y%")
   - FORBIDDEN: Saying "room for improvement" without baseline comparison

2. IDENTIFY CONCENTRATION & TOP PERFORMERS:
   - ✅ GOOD: "The top 3 ad sources (list names) with the lowest screen fail rates are driving 78% of the study's total enrollments"
   - ✅ GOOD: "2 high-quality sources account for 70% of enrollment volume, while bottom 7 sources contribute only 12%"
   - ❌ BAD: "Lower screen fail rates lead to more enrollments" (obvious, no concentration analysis)
   - REQUIRED: Identify which sources drive majority of enrollment results
   - REQUIRED: Multi-factor causation for why top sources succeed (quality + volume + screen fail analysis)

3. MULTI-FACTOR CAUSATION (NOT single correlations):
   - ✅ GOOD: "The top 3 sources driving 78% of enrollments share these factors: (1) ICF→Enrollment rates >45% vs study median of 28%, (2) screen fail rates <12% vs median of 23%, (3) moderate-high volume (120-200 qualified/month) providing both quality AND scale"
   - ❌ BAD: "Better quality sources enroll more participants" (single dimension, vague)
   - REQUIRED: Link multiple metrics (enrollment quality + screen fail + volume) to explain performance

4. Cite specific enrollment rates: "Source X: 42% Qualified→Enrollment vs Source Y: 12%"
5. Show enrollment causality: "Quality Source A delivers 3.5x enrollment vs high-volume Source B"
6. Calculate enrollment ROI: "Source X delivers $Y cost-per-enrollment vs study median $Z"
7. Quantify enrollment reallocation: "Shifting 25% budget from X to Y would yield Z more enrollments"
8. Use ACTUAL source names from the data
9. Show enrollment-specific calculations
10. FIND TIPPING POINTS: Identify quality/screen fail thresholds for optimal enrollment
   - ✅ GOOD: "Sources with <15% screen fail rate deliver 2.5x enrollments per dollar vs >25% screen fail sources"
   - ✅ GOOD: "Sources with >40% ICF→Enrollment rate yield 3.1x ROI vs <20% sources (40% is efficiency threshold)"
   - ❌ BAD: "Better quality sources enroll more participants" (no threshold, not actionable)

FORBIDDEN OBVIOUS INSIGHTS (these provide NO value):
❌ "Higher enrollment rates lead to more enrollments" (circular logic)
❌ "Lower screen fail rates lead to more enrollments" (obvious causation)
❌ "More qualified referrals lead to more enrollments" (volume correlation)
❌ Any statement without specific quality/screen fail threshold and ROI

CRITICAL DATA AVAILABILITY RULE:
NEVER speculate about missing data or mention what you DON'T have.
❌ FORBIDDEN: "While the data doesn't provide cost metrics..."
❌ FORBIDDEN: "Further data would likely reveal..."
❌ FORBIDDEN: "The provided data doesn't explicitly detail..."
❌ FORBIDDEN: "Although we lack information on..."
✅ REQUIRED: ONLY analyze metrics that ARE provided in the data
✅ REQUIRED: If a metric isn't in the data, don't mention it at all
✅ REQUIRED: Work with what you HAVE, not what you wish you had

CRITICAL: When recommending budget shifts, provide SPECIFIC, NUMBERED ACTION PLANS with ROI calculations
❌ BAD: "Diversify marketing channels"
❌ BAD: "Reallocate budget to sources with better enrollment rates"
✅ GOOD: "The key opportunity areas for enrollment improvement are: reallocating budget from high-screen-fail sources to low-screen-fail sources. The top 3 sources (list names) driving 78% of enrollments have median screen fail of 12% vs bottom 5 sources at 31%. Recommended actions:
1. Reallocate 40% of Facebook budget ($16K/month) to community_events source
   - Facebook: 280 qualified, 31% screen fail × 18% enrollment = 50 enrollments ($850 per enrollment)
   - Community: 95 qualified, 12% screen fail × 52% enrollment = 49 enrollments ($360 per enrollment)
2. Expected Facebook impact: 280→168 qualified (-20 enrollments)
3. Expected community impact: 95→201 qualified (+55 enrollments)
4. Net result: +35 enrollments monthly (+70% improvement) at 58% lower cost-per-enrollment, achieving study median quality"

REQUIRED ELEMENTS FOR ACTION PLANS:
- List specific underperforming sources by name
- Reference study median/percentiles for benchmarking (NOT invented targets)
- Provide numbered, specific actions with full ROI calculations
- Quantify expected impact: "would result in X additional enrollments per month at Y% lower cost"
- Multi-source reallocation analysis with screen fail consideration

EXAMPLE OF EXCELLENT ENROLLMENT INSIGHT WITH ROI ANALYSIS:
"Sources with <15% screen fail rate deliver 2.5x enrollment efficiency vs >25% screen fail (15% is quality threshold). utm_source=community_events: 52% enrollment rate, 12% screen fail, 95 qualified/month = 49 enrollments. utm_source=facebook: 18% enrollment rate, 31% screen fail, 280 qualified/month = 50 enrollments. Despite Facebook's 195% higher volume, community source delivers equivalent enrollments at superior quality.

PRESCRIPTION for budget optimization:
(a) Shift 40% of Facebook budget ($16K/month) to community events expansion
(b) Expected Facebook impact: 280→168 qualified (-20 enrollments)
(c) Expected community impact: 95→201 qualified (+55 enrollments, assuming linear scaling)
Net result: +35 enrollments monthly (+70% improvement) at 58% lower cost-per-enrollment (from $850 to $360). Additionally, 12% screen fail vs 31% indicates better long-term participant retention."

EXAMPLE OF BAD ENROLLMENT INSIGHT:
❌ "Diversify marketing channels" (vague, no enrollment impact data, no ROI)
❌ "Better quality sources enroll more" (no threshold, no actionable recommendation)

CRITICAL INSTRUCTION:
Focus EXCLUSIVELY on enrollment-driving operational metrics. NO scores or rankings.

IMPORTANT: Start your response DIRECTLY with section 1 below. NO introductory sentences.

Provide data-driven insights focused EXCLUSIVELY on Enrollment:

1. EXECUTIVE OVERVIEW (2-3 sentences)
   - Quantify enrollment rates and volumes across ad sources
   - Identify quality vs. volume patterns in enrollment generation
   - NO mention of pre-ICF metrics

2. ENROLLMENT SUCCESS DRIVERS (Top 3 ad sources/patterns driving enrollment)
   - Compare enrollment rates: "Source A: 52% Qualified→Enrollment vs Source B: 18%"
   - Quality patterns: "Source X delivers 42% enrollment rate with lower screen fails"
   - Calculate cost-per-enrollment efficiency

3. ENROLLMENT OPPORTUNITIES (Top 3 quantified enrollment improvement areas)
   - Identify low-enrollment sources vs high-enrollment sources with SPECIFIC numbers
   - Calculate enrollment reallocation impact: "Moving $X from Source A to B would yield N additional enrollments"
   - Quantify which sources to scale for enrollment maximization

4. FORWARD OUTLOOK (2-3 sentences)
   - Calculate optimal budget mix for enrollment generation
   - Quantify enrollment impact of recommended reallocations
   - Provide specific percentages and expected enrollment yields

Remember: ACT AS A DATA SCIENTIST. Focus on enrollment ROI, not generic marketing advice."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating insights: {str(e)}\n\nPlease check your GEMINI_API_KEY in .streamlit/secrets.toml"


def generate_pc_insights(sanitized_data: dict, focus: str = "ICF") -> str:
    """
    Generate AI insights for PC performance.

    Args:
        sanitized_data: Aggregated PC metrics
        focus: Analysis focus - "ICF" or "Enrollment"

    Returns:
        AI-generated insights text
    """
    if 'error' in sanitized_data:
        return f"Unable to generate insights: {sanitized_data['error']}"

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=genai.GenerationConfig(temperature=0.2)
        )

        # Extract time metrics if available
        time_metrics_text = ""
        if 'time_metrics' in sanitized_data:
            tm = sanitized_data['time_metrics']
            time_metrics_text = f"""
TIME-BASED METRICS:
- Average Time to First Contact: {tm.get('avg_time_to_first_contact', 'N/A')} days
- Average Time Between Contact Attempts: {tm.get('avg_time_between_contacts', 'N/A')} days
- Average Time from New to Sent To Site: {tm.get('avg_time_new_to_sts', 'N/A')} days
"""

        # Extract heatmap insights if available
        heatmap_text = ""
        if 'heatmap_insights' in sanitized_data and sanitized_data['heatmap_insights']:
            hi = sanitized_data['heatmap_insights']
            heatmap_text = f"""
CALL TIMING ANALYSIS:
- Best times for calls: {', '.join(hi.get('best_times', []))}
- Times to avoid: {', '.join(hi.get('avoid_times', []))}
- Peak activity hours: {hi.get('peak_hours', 'N/A')}
"""

        # Format effectiveness segments data
        metrics_type = sanitized_data.get('metrics_type', 'unknown')
        segment_label = "Time to First Contact" if metrics_type == 'ttfc' else "Number of Contact Attempts"

        if focus == "ICF":
            prompt = f"""You are a Chief Data Scientist analyzing PC behavior patterns to maximize ICF conversions.

FOCUS: DRIVING ICF CONVERSIONS
Identify PC behaviors that drive referrals to ICF completion.

EFFECTIVENESS ANALYSIS BY {segment_label.upper()}:
{json.dumps(sanitized_data.get('all_segments', []), indent=2)}

{time_metrics_text}

{heatmap_text}

CRITICAL TERMINOLOGY:
StS = "Sent to Site" (NOT "screened to site" or "screen to site")
ALWAYS use "Sent to Site" or "StS" - NEVER use incorrect terminology.

TIME FORMAT RULE:
Convert all decimal days/times to human-readable format in your analysis.
- Example: "2.91 days" → "2 days 22 hours"
- Example: "0.5 days" → "12 hours"

FORMATTING RULE:
Use ### for all section headers (NOT # or ##)
✅ GOOD: "### 1. EXECUTIVE OVERVIEW"
❌ BAD: "# 1. EXECUTIVE OVERVIEW" or "## 1. EXECUTIVE OVERVIEW"

DATA SOURCES TO CROSS-REFERENCE:
You have access to multiple PC team performance data sources - analyze ALL of them together:

1. Time to First Contact Effectiveness (PC TEAM behavior):
   - ICF/Enrollment rates by PC response time segments (<1 day, 1-2 days, 3+ days, etc.)
   - Identifies PC response speed tipping points

2. Contact Attempt Effectiveness (PC TEAM behavior):
   - ICF/Enrollment rates by PC contact frequency (1-2 attempts, 3-4, 5+, etc.)
   - Identifies optimal PC persistence patterns

3. Call Timing Heatmap Data (if available):
   - Best/worst hours for call success
   - Peak activity patterns

4. Any other PC operational metrics visible on the page

REQUIRED: Cross-reference ALL sources for multi-dimensional PC team recommendations

KEY METRICS FOR ICF SUCCESS:
- ICF Rate by response speed segments (e.g., <1 day vs 3+ days)
- ICF Rate by contact attempt frequency
- Time to first contact impact on ICF conversion
- Call timing effectiveness for ICF (heatmap data)
- Sent to Site rates (getting referrals to sites for ICF)

IGNORE FOR THIS ANALYSIS:
- Enrollment rates (post-ICF metric)
- Screen fail after ICF (not PC-controllable)

DATA-DRIVEN ANALYSIS RULES FOR ICF:
1. USE DISTRIBUTION-BASED COMPARISONS (calculate ALL statistics from actual data):
   - ✅ GOOD: "The average response time is [calculate from actual data], with [calculate %] of referrals receiving first contact after this average"
   - ✅ GOOD: "Referrals contacted within 1 day represent the top [calculate %] of response times"
   - ✅ GOOD: "The median (50th percentile) response time is [calculate], meaning half of referrals get faster response and half slower"
   - ❌ BAD: "Average response time is 2.3 days, indicating room for improvement" (no baseline context)
   - ❌ BAD: "The median is 2.3 days, with 65% of referrals after this threshold" (WRONG - median = 50th percentile by definition!)
   - CRITICAL: CALCULATE all statistics from the actual data provided - NEVER echo example numbers from these instructions
   - STATISTICAL ACCURACY: Median = 50th percentile (exactly 50% above, 50% below). Average/mean can have any distribution.
   - REQUIRED: Use average/mean with distribution context ("X% of referrals fall below/above the average")
   - FORBIDDEN: Saying "room for improvement" without baseline comparison

2. IDENTIFY CONCENTRATION & BEHAVIORAL PATTERNS:
   - ✅ GOOD: "The 35% of referrals (280/month) receiving <1 day response are generating 72% of the study's ICFs"
   - ✅ GOOD: "Referrals with 3-4 contact attempts (25% of volume) account for 68% of ICF outcomes"
   - ❌ BAD: "More contact attempts lead to higher ICF rates" (obvious, no concentration analysis)
   - REQUIRED: Identify which behavioral segments drive majority of ICF results
   - REQUIRED: Multi-factor causation for why certain behaviors succeed

3. MULTI-FACTOR CAUSATION (NOT single correlations):
   - ✅ GOOD: "The 72% of ICFs from fast-response referrals correlate with: (1) response time <1 day vs study median of 2.3 days, (2) 3-4 contact attempts vs median of 2, (3) morning call timing (9-11am) vs median afternoon timing"
   - ❌ BAD: "Faster PC response correlates with better ICF rates" (single factor, obvious)
   - REQUIRED: Link multiple PC behaviors (timing + attempts + call windows) to explain performance

4. Cite specific ICF rates from segments: "Response within 1 day: 58% ICF rate vs 3+ days: 24%"
5. Show ICF causality: "Faster response drives 2.4x higher ICF rates because [data shows...]"
6. Quantify ICF behavioral differences: "3 contact attempts yield X% ICF vs 1 attempt yields Y% ICF"
7. Calculate ICF impact: "Moving referrals from 3+ day to <1 day response would yield N additional ICFs"
8. NO generic training - specify WHAT behaviors drive ICF
9. Show ICF-specific math
10. FIND TIPPING POINTS: Identify PC behavior thresholds where ICF outcomes shift dramatically
   - ✅ GOOD: "PC response <1 day: 58% ICF rate vs >2 days: 24% ICF (<1 day = 2.4x tipping point)"
   - ✅ GOOD: "3-4 contact attempts: 67% ICF vs 1-2 attempts: 23% ICF (3+ attempts is threshold)"
   - ✅ GOOD: "Morning calls (9-11am): 67% ICF vs afternoon (2-5pm): 31% ICF (morning is optimal window)"
   - ❌ BAD: "Faster PC response correlates with better ICF rates" (no threshold, not actionable)

FORBIDDEN OBVIOUS INSIGHTS (these provide NO value):
❌ "More contact attempts lead to higher ICF rates" (obvious, no threshold)
❌ "Faster PC response correlates with success" (no specific tipping point)
❌ "Better PC performance drives better outcomes" (circular logic)
❌ Any correlation without actionable threshold or multi-dimensional HOW

CRITICAL DATA AVAILABILITY RULE:
NEVER speculate about missing data or mention what you DON'T have.
❌ FORBIDDEN: "While the data doesn't provide granular detail on attempts..."
❌ FORBIDDEN: "Further data would likely reveal..."
❌ FORBIDDEN: "The provided data doesn't explicitly detail..."
❌ FORBIDDEN: "Although we lack information on..."
✅ REQUIRED: ONLY analyze metrics that ARE provided in the data
✅ REQUIRED: If a metric isn't in the data, don't mention it at all
✅ REQUIRED: Work with what you HAVE, not what you wish you had

CRITICAL: When recommending PC team improvements, provide SPECIFIC, NUMBERED ACTION PLANS with achievable goals
❌ BAD: "PC team would benefit from response time training"
❌ BAD: "Improving PC performance presents a significant opportunity"
✅ GOOD: "The key areas with most opportunity to drive ICF improvement are: response time and contact attempt frequency. The 65% of referrals (340/month) in the >2 day response bucket are achieving only 24% ICF rate vs the 35% in <1 day bucket achieving 58%. Recommended actions:
1. Reduce response time from current median of 2 days 14 hours to study top-quartile of <22 hours for the 340 referrals/month currently in 3+ day bucket (tipping point data shows <1 day = 58% ICF vs >2 days = 24%)
2. Increase contact attempts from current median of 2.3 to study top-quartile of 4 attempts per referral (4+ attempts show 2.9x higher ICF rates)
3. Shift 50% of afternoon calls (2-5pm, currently 45% of daily volume) to morning window (9-11am) per heatmap showing 2.2x higher contact success
4. An achievable goal would be to get these 340 referrals to study top-quartile metrics, which would result in an additional 95 ICFs annually"

REQUIRED ELEMENTS FOR ACTION PLANS:
- Identify specific underperforming behavioral segments with volume counts
- Reference study median/quartiles as achievable benchmarks (NOT invented targets)
- Provide numbered, specific PC behavior changes with metrics
- Quantify expected impact: "would result in X additional ICFs per month/year"
- Multi-dimensional PC improvements (timing + attempts + call windows)

EXAMPLE OF EXCELLENT ICF INSIGHT WITH TIPPING POINTS:
"PC response <1 day achieves 58% ICF rate vs >2 days: 24% ICF (<1 day is critical tipping point, 2.4x difference). Currently 35% of referrals (280/month) fall in 3+ day response bucket. Effectiveness data shows 3-4 contact attempts yield 67% ICF vs 1-2 attempts: 23% ICF (2.9x difference, 3+ is threshold). Heatmap shows morning calls (9-11am) achieve 67% contact rate vs afternoon (2-5pm): 31% (2.2x difference).

PRESCRIPTION for PC team:
(a) Reduce response time for 280 referrals/month from 3+ days to <22 hours (below 1-day tipping point) - shift from 24% to 58% ICF rate
(b) Increase contact attempts from team average of 2.3 to 4 per referral - leverage 3+ attempt threshold
(c) Shift 50% of afternoon calls (currently 45% of daily volume) to 9-11am window per heatmap

Expected impact:
- Response improvement: 280 × (58%-24%) = 95 additional ICFs annually
- Attempt improvement: 15-20 additional ICFs monthly from increased persistence
- Timing optimization: 8-12 additional ICFs monthly from optimal call window
Total: ~130 additional ICFs annually (+43% improvement)"

EXAMPLE OF BAD ICF INSIGHT:
❌ "PC team would benefit from response time training" (vague, no tipping point, no HOW)
❌ "More contact attempts lead to higher ICF rates" (obvious, no threshold)

IMPORTANT: Start your response DIRECTLY with section 1 below. NO introductory sentences.

Provide data-driven insights focused EXCLUSIVELY on ICF success:

1. EXECUTIVE OVERVIEW (2-3 sentences)
   - Quantify ICF conversion rates by PC behavior segments
   - Identify key behavioral patterns driving ICF
   - NO mention of enrollment

2. ICF SUCCESS DRIVERS (Top 3 PC behaviors driving ICF)
   - Analyze ICF rates by segment: "Referrals with <1 day response achieve 58% ICF vs 24% for 3+ days"
   - Identify ICF-driving behaviors: "3-4 attempts yield 67% ICF vs 1 attempt at 23%"
   - ICF timing patterns: "Morning calls achieve 67% ICF vs afternoon 31%"

3. ICF OPPORTUNITIES (Top 3 quantified ICF improvement areas)
   - Compare ICF rates across segments with SPECIFIC numbers
   - Calculate ICF impact: "Moving X% of referrals to optimal behavior would yield N additional ICFs"
   - Quantify referral volume in low-ICF segments that could improve

4. FORWARD OUTLOOK (2-3 sentences)
   - Quantify which behavioral changes offer highest ICF ROI
   - Calculate ICF impact if team adopted best practices
   - Prioritize specific IC F-driving process changes

Remember: ACT AS A DATA SCIENTIST. Focus ONLY on driving ICF conversions."""

        else:  # focus == "Enrollment"
            prompt = f"""You are a Chief Data Scientist analyzing PC behavior patterns to maximize Enrollment conversions.

FOCUS: DRIVING ENROLLMENT CONVERSIONS
Identify PC behaviors that drive complete enrollment success.

EFFECTIVENESS ANALYSIS BY {segment_label.upper()}:
{json.dumps(sanitized_data.get('all_segments', []), indent=2)}

{time_metrics_text}

{heatmap_text}

CRITICAL TERMINOLOGY:
StS = "Sent to Site" (NOT "screened to site" or "screen to site")
ALWAYS use "Sent to Site" or "StS" - NEVER use incorrect terminology.

TIME FORMAT RULE:
Convert all decimal days/times to human-readable format in your analysis.
- Example: "2.91 days" → "2 days 22 hours"
- Example: "0.5 days" → "12 hours"

FORMATTING RULE:
Use ### for all section headers (NOT # or ##)
✅ GOOD: "### 1. EXECUTIVE OVERVIEW"
❌ BAD: "# 1. EXECUTIVE OVERVIEW" or "## 1. EXECUTIVE OVERVIEW"

DATA SOURCES TO CROSS-REFERENCE:
You have access to multiple PC team performance data sources - analyze ALL of them together:

1. Time to First Contact Effectiveness (PC TEAM behavior):
   - Enrollment rates by PC response time segments
   - Identifies PC response speed impact on enrollment

2. Contact Attempt Effectiveness (PC TEAM behavior):
   - Enrollment rates by PC contact frequency
   - Identifies optimal PC persistence for enrollment

3. Call Timing Heatmap Data (if available):
   - Best/worst hours for enrollment success
   - Peak activity patterns

4. Any other PC operational metrics visible on the page

REQUIRED: Cross-reference ALL sources for multi-dimensional PC team enrollment recommendations

KEY METRICS FOR ENROLLMENT SUCCESS:
- Enrollment Rate by response speed segments
- Enrollment Rate by contact attempt frequency
- Complete funnel efficiency (New → Enrollment)
- Enrollment rates by call timing patterns
- Quality of referrals sent to site (impact on enrollment)

IGNORE FOR THIS ANALYSIS:
- StS rates alone (intermediate metric, focus on enrollment)
- Appointment scheduling (pre-enrollment step)

DATA-DRIVEN ANALYSIS RULES FOR ENROLLMENT:
1. USE DISTRIBUTION-BASED COMPARISONS (calculate ALL statistics from actual data):
   - ✅ GOOD: "The average post-ICF response time is [calculate from actual data], with [calculate %] of post-ICF referrals receiving first contact after this average"
   - ✅ GOOD: "Post-ICF referrals contacted within 24 hours represent the top [calculate %] of response times"
   - ✅ GOOD: "The median (50th percentile) post-ICF response time is [calculate], meaning half faster and half slower"
   - ❌ BAD: "Average post-ICF response time is 3.2 days, indicating room for improvement" (no baseline context)
   - ❌ BAD: "The median is 3.2 days, with 70% of referrals after this threshold" (WRONG - median = 50th percentile by definition!)
   - CRITICAL: CALCULATE all statistics from the actual data provided - NEVER echo example numbers from these instructions
   - STATISTICAL ACCURACY: Median = 50th percentile (exactly 50% above, 50% below). Average/mean can have any distribution.
   - REQUIRED: Use average/mean with distribution context ("X% of referrals fall below/above the average")
   - FORBIDDEN: Saying "room for improvement" without baseline comparison

2. IDENTIFY CONCENTRATION & BEHAVIORAL PATTERNS:
   - ✅ GOOD: "The 30% of post-ICF referrals (240/month) receiving <24hr response are generating 75% of the study's enrollments"
   - ✅ GOOD: "Post-ICF referrals with 3-4 contact attempts (22% of volume) account for 71% of enrollment outcomes"
   - ❌ BAD: "More post-ICF contact attempts lead to higher enrollment" (obvious, no concentration analysis)
   - REQUIRED: Identify which PC behavioral segments drive majority of enrollment results
   - REQUIRED: Multi-factor causation for why certain post-ICF behaviors succeed

3. MULTI-FACTOR CAUSATION (NOT single correlations):
   - ✅ GOOD: "The 75% of enrollments from fast-response post-ICF referrals correlate with: (1) response time <24hr vs study median of 3.2 days, (2) 3-4 contact attempts vs median of 2.1, (3) morning call timing (9-11am) vs median afternoon timing"
   - ❌ BAD: "More PC follow-up correlates with higher enrollment" (single factor, obvious)
   - REQUIRED: Link multiple PC post-ICF behaviors (timing + attempts + call windows) to explain enrollment performance

4. Cite specific enrollment rates from segments: "Response within 1 day: 47% enrollment vs 3+ days: 19%"
5. Show enrollment causality: "More contact attempts drive 2.5x higher enrollment because [data shows...]"
6. Quantify enrollment behavioral differences: "3-4 attempts yield X% enrollment vs 1-2 attempts Y%"
7. Calculate enrollment impact: "Increasing follow-up for these referrals would yield N additional enrollments"
8. NO generic training - specify WHAT behaviors drive enrollment
9. Show enrollment-specific math
10. FIND TIPPING POINTS: Identify PC behavior thresholds where enrollment outcomes shift dramatically
   - ✅ GOOD: "PC response <24 hours: 52% enrollment vs >48 hours: 28% enrollment (<24hr is critical threshold)"
   - ✅ GOOD: "3-4 contact attempts: 47% enrollment vs 1-2 attempts: 19% enrollment (3+ attempts is threshold)"
   - ✅ GOOD: "Morning calls (9-11am): 48% enrollment vs afternoon (2-5pm): 22% enrollment (morning is optimal)"
   - ❌ BAD: "More PC follow-up correlates with higher enrollment" (no threshold, not actionable)

FORBIDDEN OBVIOUS INSIGHTS (these provide NO value):
❌ "More contact attempts lead to higher enrollment" (obvious, no threshold)
❌ "Faster PC response correlates with enrollment success" (no specific tipping point)
❌ "Better PC follow-up drives better outcomes" (circular logic)
❌ Any correlation without actionable threshold or multi-dimensional HOW

CRITICAL DATA AVAILABILITY RULE:
NEVER speculate about missing data or mention what you DON'T have.
❌ FORBIDDEN: "While the data doesn't provide granular detail on attempts..."
❌ FORBIDDEN: "Further data would likely reveal..."
❌ FORBIDDEN: "The provided data doesn't explicitly detail..."
❌ FORBIDDEN: "Although we lack information on..."
✅ REQUIRED: ONLY analyze metrics that ARE provided in the data
✅ REQUIRED: If a metric isn't in the data, don't mention it at all
✅ REQUIRED: Work with what you HAVE, not what you wish you had

CRITICAL: When recommending PC team improvements, provide SPECIFIC, NUMBERED ACTION PLANS with achievable goals
❌ BAD: "PC team would benefit from better follow-up"
❌ BAD: "Improving post-ICF PC performance presents a significant opportunity"
✅ GOOD: "The key areas with most opportunity to drive enrollment improvement are: post-ICF response time and contact attempt frequency. The 70% of post-ICF referrals (420/month) in the >48hr response bucket are achieving only 28% enrollment rate vs the 30% in <24hr bucket achieving 52%. Recommended actions:
1. Reduce post-ICF response time from current median of 3 days 2 hours to study top-quartile of <22 hours for the 420 referrals/month currently in 48hr+ bucket (tipping point data shows <24hr = 52% enrollment vs >48hr = 28%)
2. Increase post-ICF contact attempts from current median of 2.1 to study top-quartile of 4 per referral (4+ attempts show 2.5x higher enrollment rates)
3. Shift 60% of afternoon post-ICF calls (2-5pm, currently 55% of daily volume) to morning window (9-11am) per heatmap showing 2.2x higher enrollment success
4. An achievable goal would be to get these 420 post-ICF referrals to study top-quartile metrics, which would result in an additional 140 enrollments annually"

REQUIRED ELEMENTS FOR ACTION PLANS:
- Identify specific underperforming post-ICF behavioral segments with volume counts
- Reference study median/quartiles as achievable benchmarks (NOT invented targets)
- Provide numbered, specific PC behavior changes with metrics
- Quantify expected impact: "would result in X additional enrollments per month/year"
- Multi-dimensional PC improvements (post-ICF timing + attempts + call windows)

EXAMPLE OF EXCELLENT ENROLLMENT INSIGHT WITH TIPPING POINTS:
"PC response <24 hours achieves 52% enrollment rate vs >48 hours: 28% enrollment (<24hr is critical tipping point, 1.9x difference). Currently 45% of post-ICF referrals (340/month) receive first contact after 48 hours. Effectiveness data shows 3-4 contact attempts yield 47% enrollment vs 1-2 attempts: 19% enrollment (2.5x difference, 3+ is threshold). Heatmap shows morning post-ICF calls (9-11am) achieve 48% enrollment vs afternoon (2-5pm): 22% (2.2x difference).

PRESCRIPTION for PC team:
(a) Reduce post-ICF response time for 340 referrals/month from >48hr to <22 hours (below 24hr tipping point) - shift from 28% to 52% enrollment rate
(b) Increase contact attempts from team average of 2.1 to 4 per post-ICF referral - leverage 3+ attempt threshold
(c) Shift 60% of afternoon post-ICF calls (currently 55% of daily volume) to 9-11am window per heatmap

Expected impact:
- Response improvement: 340 × (52%-28%) = 82 additional enrollments annually
- Attempt improvement: 18-25 additional enrollments monthly from increased persistence
- Timing optimization: 12-15 additional enrollments monthly from optimal call window
Total: ~140 additional enrollments annually (+52% improvement)"

EXAMPLE OF BAD ENROLLMENT INSIGHT:
❌ "PC team would benefit from better follow-up" (vague, no tipping point, no HOW)
❌ "More contact attempts lead to higher enrollment" (obvious, no threshold)

IMPORTANT: Start your response DIRECTLY with section 1 below. NO introductory sentences.

Provide data-driven insights focused EXCLUSIVELY on Enrollment success:

1. EXECUTIVE OVERVIEW (2-3 sentences)
   - Quantify enrollment rates by PC behavior segments
   - Identify key behavioral patterns driving enrollment
   - NO mention of pre-enrollment metrics

2. ENROLLMENT SUCCESS DRIVERS (Top 3 PC behaviors driving enrollment)
   - Analyze enrollment rates by segment: "3-4 attempts achieve 47% enrollment vs 19% for 1-2"
   - Identify enrollment-driving behaviors: "Rapid response yields 52% enrollment vs 28% for delayed"
   - Enrollment timing patterns: "Morning calls achieve 48% enrollment vs afternoon 22%"

3. ENROLLMENT OPPORTUNITIES (Top 3 quantified enrollment improvement areas)
   - Compare enrollment rates across segments with SPECIFIC numbers
   - Calculate enrollment impact: "Improving X behavior would yield N additional enrollments"
   - Quantify referral volume in low-enrollment segments

4. FORWARD OUTLOOK (2-3 sentences)
   - Quantify which behavioral changes offer highest enrollment ROI
   - Calculate enrollment impact if team adopted best practices
   - Prioritize specific enrollment-driving process changes

Remember: ACT AS A DATA SCIENTIST. Focus ONLY on driving Enrollment conversions."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating insights: {str(e)}\n\nPlease check your GEMINI_API_KEY in .streamlit/secrets.toml"


def generate_funnel_insights(sanitized_data: dict, focus: str = "ICF") -> str:
    """
    Generate AI insights for funnel analysis.

    Args:
        sanitized_data: Aggregated funnel metrics
        focus: Analysis focus - "ICF" or "Enrollment"

    Returns:
        AI-generated insights text
    """
    if 'error' in sanitized_data:
        return f"Unable to generate insights: {sanitized_data['error']}"

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=genai.GenerationConfig(temperature=0.2)
        )

        if focus == "ICF":
            prompt = f"""You are a Chief Data Scientist analyzing pipeline projections to assess ICF capacity.

FOCUS: ICF CAPACITY ANALYSIS
Assess current pipeline ICF capacity and project potential outcomes.

CURRENT PIPELINE PROJECTIONS:
- Total Expected ICF Yield from Pipeline: {sanitized_data['total_icf_yield']:.1f}
- Peak Monthly ICF Expected: {sanitized_data['peak_monthly_icf']:.1f}
- Average Monthly ICF: {sanitized_data['avg_monthly_icf']:.1f}
- Final Cumulative ICF: {sanitized_data['final_cumulative_icf']:.1f}
- Projection Timeframe: {sanitized_data['num_months_projected']} months

CRITICAL CONTEXT: This represents expected ICFs from leads ALREADY in pipeline - what would happen if new recruitment stopped today.

CRITICAL TERMINOLOGY:
StS = "Sent to Site" (NOT "screened to site" or "screen to site")
ALWAYS use "Sent to Site" or "StS" - NEVER use incorrect terminology.

TIME FORMAT RULE:
Convert all decimal months to human-readable format (e.g., "6.5 months" → "6 months 2 weeks")

FORMATTING RULE:
Use ### for all section headers (NOT # or ##)
✅ GOOD: "### 1. EXECUTIVE OVERVIEW"
❌ BAD: "# 1. EXECUTIVE OVERVIEW" or "## 1. EXECUTIVE OVERVIEW"

DATA SOURCES:
You have access to pipeline projection metrics showing ICF capacity over time.

KEY METRICS FOR ICF CAPACITY:
- Total ICF yield projection
- Monthly ICF run rate
- Peak ICF capacity months
- ICF timeline and velocity

IGNORE FOR THIS ANALYSIS:
- Enrollment-specific gaps (focus on ICF capacity first)
- Post-ICF conversion rates (separate analysis)

DATA-DRIVEN ANALYSIS RULES FOR ICF:
1. USE DISTRIBUTION-BASED COMPARISONS for scenarios (calculate ALL statistics from actual data):
   - ✅ GOOD: "Pipeline projects [calculate from data] ICFs over [data] months ([calculate]/month average). Scenario A yields [calculate] ICFs. Scenario B yields [calculate] ICFs."
   - ✅ GOOD: "Current [calculate]/month ICFs places pipeline performance at [calculate context from data]"
   - ❌ BAD: "Pipeline needs improvement" (no baseline context)
   - ❌ BAD: Using example numbers from instructions (127, 145, 180, etc.) - these are EXAMPLES ONLY, not real data
   - CRITICAL: CALCULATE all statistics from the actual data provided - NEVER echo example numbers from these instructions
   - REQUIRED: Reference percentile/benchmark context for scenarios
   - FORBIDDEN: Saying outcomes are "good" or "bad" without baseline comparison

2. IDENTIFY CONCENTRATION & LEVERAGE POINTS:
   - ✅ GOOD: "3 specific conversion points drive 75% of ICF yield variance: (1) Qualified→StS rate, (2) StS→Appt rate, (3) Appt→ICF rate. Improving these 3 vs spreading effort across all 5 funnel stages delivers 2.3x better ROI"
   - ✅ GOOD: "The top 30% of projected months (months 2-4) account for 65% of total ICF yield"
   - ❌ BAD: "Improve conversion rates" (no specific leverage point identification)
   - REQUIRED: Identify which specific conversion points or time periods drive majority of ICF capacity
   - REQUIRED: Multi-factor scenario analysis

3. MULTI-DIMENSIONAL SCENARIO ANALYSIS (NOT single-variable):
   - ✅ GOOD: "Scenario analysis comparing: (A) conversion improvement alone (+10pp Qualified→ICF = +53 ICFs), (B) volume increase alone (+30% qualified = +24 ICFs), (C) combined approach (+7pp conversion + 15% volume = +48 ICFs with lower cost). Recommendation: Prioritize (A) for 2.2x better ICF yield per unit effort."
   - ❌ BAD: "Improve conversion rates to increase ICF capacity" (single dimension, no comparison)
   - REQUIRED: Compare multiple improvement pathways with specific ICF yield calculations

4. Assess ICF capacity: "Pipeline projects X ICFs over Y months (Z ICFs/month average)"
5. Project ICF scenarios: "To reach various ICF levels (e.g., 150 ICFs), would need N more qualified referrals"
6. Calculate ICF timelines: "At current X ICFs/month rate, pipeline completion spans Y months"
7. Show ICF math clearly
8. Provide ICF-specific numbers
9. FIND TIPPING POINTS: Identify conversion rate or volume thresholds that dramatically change ICF capacity
   - ✅ GOOD: "Improving Qualified→ICF from 38% to 45% (+7pp) would yield 45 additional ICFs vs improving to 42% (+4pp) yields only 18 ICFs (45% is efficiency threshold)"
   - ✅ GOOD: "Increasing recruitment by 25% yields +15 ICFs but 50% increase yields +42 ICFs (non-linear returns, 50% is capacity threshold)"
   - ❌ BAD: "Improve conversion rates to increase ICF capacity" (no threshold, not actionable)

FORBIDDEN OBVIOUS INSIGHTS (these provide NO value):
❌ "Higher conversion rates lead to more ICFs" (circular logic)
❌ "More recruitment yields more ICFs" (obvious volume correlation)
❌ "Pipeline needs improvement" (vague, no specific threshold)
❌ Any statement without specific conversion threshold or recruitment scenario

CRITICAL DATA AVAILABILITY RULE:
NEVER speculate about missing data or mention what you DON'T have.
❌ FORBIDDEN: "While the data doesn't provide current conversion rates..."
❌ FORBIDDEN: "Further data would likely reveal..."
❌ FORBIDDEN: "The provided data doesn't explicitly detail..."
❌ FORBIDDEN: "Although we lack information on..."
✅ REQUIRED: ONLY analyze pipeline metrics that ARE provided
✅ REQUIRED: If a metric isn't in the data, don't mention it at all
✅ REQUIRED: Work with what you HAVE, not what you wish you had

CRITICAL: When recommending capacity improvements, provide SPECIFIC, NUMBERED SCENARIO COMPARISONS
❌ BAD: "Continue recruitment efforts for ICF"
❌ BAD: "Pipeline needs improvement to meet goals"
✅ GOOD: "Current pipeline: 127 ICFs over 6 months (21/month average). The key leverage points for ICF improvement are: conversion optimization vs volume scaling. Scenario comparison:

SCENARIO A - Modest conversion improvement:
1. Improve Qualified→ICF from 38% to 42% (+4pp)
2. Result: 145 total ICFs (+18 ICFs, +14% improvement)
3. Effort: moderate site training

SCENARIO B - Threshold conversion improvement:
1. Improve Qualified→ICF from 38% to 48% (+10pp)
2. Result: 180 total ICFs (+53 ICFs, +42% improvement)
3. Effort: comprehensive site optimization (2.5x more effort than Scenario A)
4. ROI: 2.9x better return (48% is efficiency threshold - non-linear gains)

SCENARIO C - Volume increase:
1. Increase qualified recruitment by 30% at current 38% conversion
2. Result: 151 total ICFs (+24 ICFs, +19% improvement)
3. Cost: high (more ad spend, more PC resources)

RECOMMENDATION: Prioritize Scenario B (conversion to 48%) over Scenario C (volume increase). Conversion approach delivers 2.2x more ICFs than volume approach with lower ongoing cost."

REQUIRED ELEMENTS FOR SCENARIOS:
- Compare 3+ distinct improvement pathways with full calculations
- Show non-linear returns and identify efficiency thresholds
- Quantify effort/cost differences between scenarios
- Provide clear recommendation with specific ICF yield comparison
- NO invented targets - scenarios should be based on achievable benchmarks

EXAMPLE OF EXCELLENT ICF CAPACITY INSIGHT WITH TIPPING POINTS:
"Pipeline projects 127 ICFs over 6 months (21/month average, 63.5 cumulative). Scenario analysis reveals conversion improvement threshold:

Scenario A - Modest conversion improvement:
- Qualified→ICF: 38% → 42% (+4pp improvement)
- Result: +18 ICFs (145 total)
- Effort: moderate site training

Scenario B - Threshold conversion improvement:
- Qualified→ICF: 38% → 48% (+10pp improvement)
- Result: +53 ICFs (180 total)
- Effort: comprehensive site optimization (2.5x more effort than Scenario A)
- ROI: 2.9x better return (48% is efficiency threshold - non-linear gains)

Scenario C - Volume increase:
- Increase qualified recruitment by 30% at current 38% conversion
- Result: +24 ICFs (151 total)
- Cost: high (more ad spend, more PC resources)

PRESCRIPTION:
Prioritize Scenario B (conversion to 48%) over Scenario C (volume increase). Conversion approach delivers 2.2x more ICFs than volume approach with lower ongoing cost. Focus on: (a) site response time <1 day (data shows 2.4x ICF rate), (b) 3-4 contact attempts per referral, (c) appointment rate optimization."

EXAMPLE OF BAD ICF INSIGHT:
❌ "Continue recruitment efforts for ICF" (vague, no math, no scenarios)
❌ "Improve conversion rates to increase capacity" (no threshold, not actionable)

IMPORTANT: Start your response DIRECTLY with section 1 below. NO introductory sentences.

Provide data-driven insights focused EXCLUSIVELY on ICF capacity:

1. EXECUTIVE OVERVIEW (2-3 sentences)
   - Quantify ICF pipeline capacity with specific PROJECTION NUMBERS (total ICFs, monthly ICF rates)
   - Assess pipeline ICF capacity and velocity
   - NO mention of enrollment

2. ICF CAPACITY INDICATORS (Top 3 measurable pipeline metrics)
   - Monthly ICF run rate: "Averaging X ICFs/month over Y-month projection"
   - Peak ICF capacity: "Peak month projects X ICFs, indicating maximum ICF capacity"
   - ICF timeline: "Current projection spans X months, achieving Y total ICFs"

3. ICF OPPORTUNITIES (Top 3 quantified ICF improvement scenarios)
   - Project ICF capacity scenarios: "Pipeline: X ICFs. To reach Y ICFs (hypothetical), need Z additional ICFs"
   - Quantify recruitment impact: "At W% conversion, adding X qualified referrals/month would yield Y additional ICFs"
   - Calculate conversion improvements: "Improving Qualified→ICF from X% to Y% would yield Z additional ICFs"
   - Timeline projections: "To reach various ICF levels faster, could increase recruitment by X% or improve conversion by Y%"

4. FORWARD OUTLOOK (2-3 sentences with ICF MATH)
   - Project ICF completion timeline: "At X ICFs/month, pipeline yields Y total ICFs over Z months"
   - Quantify ICF acceleration scenarios: "Increasing recruitment by X% would yield Y additional ICFs; improving conversion by Z% would yield W additional ICFs"
   - Specific ICF actions: "Add Z qualified referrals/month OR improve Qualified→ICF conversion by W% to maximize ICF capacity"

Remember: ACT AS A DATA SCIENTIST. Focus ONLY on ICF capacity. Show all ICF calculations."""

        else:  # focus == "Enrollment"
            prompt = f"""You are a Chief Data Scientist analyzing pipeline projections to assess Enrollment capacity.

FOCUS: ENROLLMENT CAPACITY ANALYSIS
Assess current pipeline enrollment capacity and project potential outcomes.

CURRENT PIPELINE PROJECTIONS:
- Total Expected Enrollment Yield: {sanitized_data['total_enroll_yield']:.1f}
- Peak Monthly Enrollment Expected: {sanitized_data['peak_monthly_enrollment']:.1f}
- Average Monthly Enrollment: {sanitized_data['avg_monthly_enrollment']:.1f}
- Final Cumulative Enrollment: {sanitized_data['final_cumulative_enrollment']:.1f}
- Projection Timeframe: {sanitized_data['num_months_projected']} months

CRITICAL CONTEXT: This represents expected enrollments from leads ALREADY in pipeline - what would happen if new recruitment stopped today.

CRITICAL TERMINOLOGY:
StS = "Sent to Site" (NOT "screened to site" or "screen to site")
ALWAYS use "Sent to Site" or "StS" - NEVER use incorrect terminology.

TIME FORMAT RULE:
Convert all decimal months to human-readable format (e.g., "6.5 months" → "6 months 2 weeks")

FORMATTING RULE:
Use ### for all section headers (NOT # or ##)
✅ GOOD: "### 1. EXECUTIVE OVERVIEW"
❌ BAD: "# 1. EXECUTIVE OVERVIEW" or "## 1. EXECUTIVE OVERVIEW"

DATA SOURCES:
You have access to pipeline projection metrics showing enrollment capacity over time.

KEY METRICS FOR ENROLLMENT CAPACITY:
- Total enrollment yield projection
- Monthly enrollment run rate
- Peak enrollment capacity months
- Enrollment timeline and velocity

IGNORE FOR THIS ANALYSIS:
- ICF-specific metrics (intermediate step, focus on final enrollments)
- Pre-ICF conversion rates (earlier funnel stage)

DATA-DRIVEN ANALYSIS RULES FOR ENROLLMENT:
1. USE DISTRIBUTION-BASED COMPARISONS for scenarios (calculate ALL statistics from actual data):
   - ✅ GOOD: "Pipeline projects [calculate from data] enrollments over [data] months ([calculate]/month average). Scenario A yields [calculate] enrollments. Scenario B yields [calculate] enrollments."
   - ✅ GOOD: "Current [calculate]/month enrollments places pipeline performance at [calculate context from data]"
   - ❌ BAD: "Pipeline enrollment capacity needs improvement" (no baseline context)
   - ❌ BAD: Using example numbers from instructions (87, 102, 129, etc.) - these are EXAMPLES ONLY, not real data
   - CRITICAL: CALCULATE all statistics from the actual data provided - NEVER echo example numbers from these instructions
   - REQUIRED: Reference percentile/benchmark context for scenarios
   - FORBIDDEN: Saying outcomes are "good" or "bad" without baseline comparison

2. IDENTIFY CONCENTRATION & LEVERAGE POINTS:
   - ✅ GOOD: "2 specific conversion points drive 80% of enrollment yield variance: (1) ICF→Enrollment rate, (2) Screen fail rate. Improving these 2 vs spreading effort across all 6 funnel stages delivers 3.1x better ROI"
   - ✅ GOOD: "The top 25% of projected months (months 3-5) account for 68% of total enrollment yield"
   - ❌ BAD: "Improve enrollment conversion" (no specific leverage point identification)
   - REQUIRED: Identify which specific conversion points or time periods drive majority of enrollment capacity
   - REQUIRED: Multi-factor scenario analysis

3. MULTI-DIMENSIONAL SCENARIO ANALYSIS (NOT single-variable):
   - ✅ GOOD: "Scenario analysis comparing: (A) screen fail reduction alone (22%→12% = +42 enrollments via better ICF→Enrollment), (B) ICF volume increase alone (+30% ICFs = +18 enrollments), (C) combined approach (screen fail to 15% + 15% more ICFs = +35 enrollments with moderate cost). Recommendation: Prioritize (A) for 2.3x better enrollment yield per unit effort."
   - ❌ BAD: "Improve ICF to Enrollment conversion" (single dimension, no comparison)
   - REQUIRED: Compare multiple improvement pathways with specific enrollment yield calculations

4. Assess enrollment capacity: "Pipeline projects X enrollments over Y months (Z enrollments/month average)"
5. Project enrollment scenarios: "To reach various enrollment levels (e.g., 120 enrollments), would need N additional ICFs"
6. Calculate enrollment timelines: "At current X enrollments/month rate, pipeline completion spans Y months"
7. Show enrollment math clearly
8. Provide enrollment-specific numbers
9. FIND TIPPING POINTS: Identify conversion rate or ICF volume thresholds that dramatically change enrollment capacity
   - ✅ GOOD: "Improving ICF→Enrollment from 68% to 75% (+7pp) would yield 38 additional enrollments vs improving to 72% (+4pp) yields only 15 enrollments (75% is efficiency threshold)"
   - ✅ GOOD: "Reducing screen fail from 22% to 15% yields +25 enrollments but reducing to 10% yields +48 enrollments (non-linear returns, 10% is quality threshold)"
   - ❌ BAD: "Improve ICF to Enrollment conversion" (no threshold, not actionable)

FORBIDDEN OBVIOUS INSIGHTS (these provide NO value):
❌ "Higher ICF→Enrollment rates lead to more enrollments" (circular logic)
❌ "More ICFs yield more enrollments" (obvious volume correlation)
❌ "Pipeline needs improvement" (vague, no specific threshold)
❌ Any statement without specific conversion threshold or ICF volume scenario

CRITICAL DATA AVAILABILITY RULE:
NEVER speculate about missing data or mention what you DON'T have.
❌ FORBIDDEN: "While the data doesn't provide current conversion rates..."
❌ FORBIDDEN: "Further data would likely reveal..."
❌ FORBIDDEN: "The provided data doesn't explicitly detail..."
❌ FORBIDDEN: "Although we lack information on..."
✅ REQUIRED: ONLY analyze pipeline metrics that ARE provided
✅ REQUIRED: If a metric isn't in the data, don't mention it at all
✅ REQUIRED: Work with what you HAVE, not what you wish you had

CRITICAL: When recommending capacity improvements, provide SPECIFIC, NUMBERED SCENARIO COMPARISONS
❌ BAD: "Continue recruitment efforts"
❌ BAD: "Pipeline enrollment capacity needs improvement"
✅ GOOD: "Current pipeline: 87 enrollments over 6 months (14.5/month average). The key leverage points for enrollment improvement are: screen fail reduction vs ICF volume scaling. Scenario comparison:

SCENARIO A - Modest screen fail reduction:
1. Reduce screen fail from 22% to 18% (-4pp)
2. Improve ICF→Enrollment from 68% to 72% (+4pp)
3. Result: 102 total enrollments (+15 enrollments, +17% improvement)
4. Effort: moderate pre-screening improvements

SCENARIO B - Threshold screen fail reduction:
1. Reduce screen fail from 22% to 12% (-10pp)
2. Improve ICF→Enrollment from 68% to 78% (+10pp)
3. Result: 129 total enrollments (+42 enrollments, +48% improvement)
4. Effort: comprehensive screening protocol overhaul (2.5x more effort than Scenario A)
5. ROI: 2.8x better return (12% screen fail is quality threshold - non-linear gains)

SCENARIO C - ICF volume increase:
1. Increase ICF volume by 30% at current 68% conversion rate
2. Result: 105 total enrollments (+18 enrollments, +21% improvement)
3. Cost: high (more site resources, more recruitment)

RECOMMENDATION: Prioritize Scenario B (screen fail to 12%) over Scenario C (volume increase). Quality approach delivers 2.3x more enrollments than volume approach with better long-term retention."

REQUIRED ELEMENTS FOR SCENARIOS:
- Compare 3+ distinct improvement pathways with full calculations
- Show non-linear returns and identify quality/efficiency thresholds
- Quantify effort/cost differences between scenarios
- Provide clear recommendation with specific enrollment yield comparison
- NO invented targets - scenarios should be based on achievable benchmarks

EXAMPLE OF EXCELLENT ENROLLMENT CAPACITY INSIGHT WITH TIPPING POINTS:
"Pipeline projects 87 enrollments over 6 months (14.5/month average, 43.5 cumulative). Scenario analysis reveals screen fail reduction as non-linear tipping point:

Scenario A - Modest screen fail reduction:
- Screen fail: 22% → 18% (-4pp improvement)
- ICF→Enrollment: 68% → 72% (+4pp)
- Result: +15 enrollments (102 total)
- Effort: moderate pre-screening improvements

Scenario B - Threshold screen fail reduction:
- Screen fail: 22% → 12% (-10pp improvement)
- ICF→Enrollment: 68% → 78% (+10pp)
- Result: +42 enrollments (129 total)
- Effort: comprehensive screening protocol overhaul (2.5x more effort than Scenario A)
- ROI: 2.8x better return (12% screen fail is quality threshold - non-linear gains)

Scenario C - ICF volume increase:
- Increase ICF volume by 30% at current 68% conversion rate
- Result: +18 enrollments (105 total)
- Cost: high (more site resources, more recruitment)

PRESCRIPTION:
Prioritize Scenario B (screen fail to 12%) over Scenario C (volume increase). Quality approach delivers 2.3x more enrollments than volume approach with better long-term retention. Focus on: (a) better pre-ICF screening criteria alignment, (b) site training on eligibility assessment, (c) early identification of disqualifying factors."

EXAMPLE OF BAD ENROLLMENT INSIGHT:
❌ "Continue recruitment efforts" (vague, no math, no scenarios)
❌ "Improve ICF to Enrollment conversion" (no threshold, not actionable)

IMPORTANT: Start your response DIRECTLY with section 1 below. NO introductory sentences.

Provide data-driven insights focused EXCLUSIVELY on Enrollment capacity:

1. EXECUTIVE OVERVIEW (2-3 sentences)
   - Quantify enrollment pipeline capacity with specific PROJECTION NUMBERS (total enrollments, monthly enrollment rates)
   - Assess pipeline enrollment capacity and velocity
   - NO mention of pre-enrollment metrics

2. ENROLLMENT CAPACITY INDICATORS (Top 3 measurable pipeline metrics)
   - Monthly enrollment run rate: "Averaging X enrollments/month over Y-month projection"
   - Peak enrollment capacity: "Peak month projects X enrollments, indicating maximum capacity"
   - Enrollment timeline: "Current projection spans X months, achieving Y total enrollments"

3. ENROLLMENT OPPORTUNITIES (Top 3 quantified enrollment improvement scenarios)
   - Project enrollment capacity scenarios: "Pipeline: X enrollments. To reach Y enrollments (hypothetical), need Z additional enrollments"
   - Quantify recruitment impact: "At W% ICF→Enrollment rate, adding X ICFs would yield Y additional enrollments, requiring Z qualified referrals"
   - Calculate conversion improvements: "Improving ICF→Enrollment from X% to Y% would yield Z additional enrollments"
   - Timeline projections: "To reach various enrollment levels faster, could increase recruitment by X% or improve conversion by Y%"

4. FORWARD OUTLOOK (2-3 sentences with ENROLLMENT MATH)
   - Project enrollment completion timeline: "At X enrollments/month, pipeline yields Y total enrollments over Z months"
   - Quantify enrollment acceleration scenarios: "Increasing recruitment by X% would yield Y additional enrollments; improving conversion by Z% would yield W additional enrollments"
   - Specific enrollment actions: "Add Z qualified referrals/month OR improve ICF→Enrollment conversion by W% to maximize enrollment capacity"

Remember: ACT AS A DATA SCIENTIST. Focus ONLY on Enrollment capacity. Show all enrollment calculations."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating insights: {str(e)}\n\nPlease check your GEMINI_API_KEY in .streamlit/secrets.toml"
