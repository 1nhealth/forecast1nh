# pages/4_Comparison_Analysis.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from constants import *
from comparison_calculations import (
    calculate_comparison_for_site_performance,
    calculate_comparison_for_site_outreach,
    calculate_comparison_for_ad_performance,
    calculate_comparison_for_pc_performance,
    calculate_comparison_for_funnel,
    validate_date_ranges
)
from comparison_helpers import (
    format_delta_indicator,
    create_comparison_display_df,
    create_metric_card_comparison,
    create_side_by_side_bar_chart,
    create_comparison_line_chart,
    create_comparison_pie_charts,
    display_validation_messages,
    create_summary_stats_table
)
from helpers import format_performance_df, load_css

st.set_page_config(page_title="Comparison Analysis", page_icon="⚖️", layout="wide")

# Load custom CSS
load_css("custom_theme.css")

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("⚖️ Comparison Analysis")
st.info("Compare metrics across two custom date ranges to understand how performance has changed over time.")

# Page Guard
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# Initialize session state keys for comparison
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'comparison_ai_insights' not in st.session_state:
    st.session_state.comparison_ai_insights = None

# --- STEP 1: Category Selection ---
st.subheader("1️⃣ Select Analysis Category")

category = st.radio(
    "Choose the type of analysis to compare:",
    options=["Site Performance", "PC Performance", "Ad Performance", "Funnel Analysis"],
    horizontal=True,
    key="comparison_category"
)

st.divider()

# --- STEP 2: Date Ranges & Labels ---
st.subheader("2️⃣ Configure Time Periods")

# Get data date range
df = st.session_state.referral_data_processed
min_date = df['Submitted On_DT'].min().date()
max_date = df['Submitted On_DT'].max().date()

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 📅 Period A (Baseline)")
        label_a = st.text_input(
            "Label for Period A",
            value="Before Training",
            key="comparison_label_a",
            help="Give this period a meaningful name (e.g., 'Before Training', 'Q1 2024')"
        )

        # Default to first half of data
        mid_date = min_date + (max_date - min_date) / 2
        default_start_a = min_date
        default_end_a = mid_date

        date_range_a = st.date_input(
            "Date Range for Period A",
            value=[default_start_a, default_end_a],
            min_value=min_date,
            max_value=max_date,
            key="comparison_date_range_a"
        )

with col2:
    with st.container(border=True):
        st.markdown("### 📅 Period B (Comparison)")
        label_b = st.text_input(
            "Label for Period B",
            value="After Training",
            key="comparison_label_b",
            help="Give this period a meaningful name (e.g., 'After Training', 'Q2 2024')"
        )

        # Default to second half of data
        default_start_b = mid_date + timedelta(days=1)
        default_end_b = max_date

        date_range_b = st.date_input(
            "Date Range for Period B",
            value=[default_start_b, default_end_b],
            min_value=min_date,
            max_value=max_date,
            key="comparison_date_range_b"
        )

# Ensure we have tuples
if isinstance(date_range_a, list) and len(date_range_a) == 2:
    start_a, end_a = date_range_a
elif hasattr(date_range_a, '__iter__') and not isinstance(date_range_a, str):
    dates_list = list(date_range_a)
    start_a = dates_list[0] if len(dates_list) > 0 else min_date
    end_a = dates_list[1] if len(dates_list) > 1 else start_a
else:
    start_a = end_a = date_range_a

if isinstance(date_range_b, list) and len(date_range_b) == 2:
    start_b, end_b = date_range_b
elif hasattr(date_range_b, '__iter__') and not isinstance(date_range_b, str):
    dates_list = list(date_range_b)
    start_b = dates_list[0] if len(dates_list) > 0 else mid_date
    end_b = dates_list[1] if len(dates_list) > 1 else start_b
else:
    start_b = end_b = date_range_b

st.divider()

# --- STEP 3: Category-Specific Options ---
st.subheader("3️⃣ Comparison Options")

# Clear comparison results if category changes
if 'last_comparison_category' not in st.session_state:
    st.session_state.last_comparison_category = category
elif st.session_state.last_comparison_category != category:
    # Category changed - clear results
    st.session_state.comparison_results = None
    st.session_state.comparison_ai_insights = None
    st.session_state.last_comparison_category = category

# Business hours toggle (for categories that support it)
if category in ["Site Performance", "PC Performance"]:
    business_hours_only = st.toggle(
        "Business Hours Only (Mon-Fri 9am-5pm)",
        value=False,
        key="comparison_business_hours",
        help="Filter calculations to only include actions during business hours"
    )

    # Clear results if business hours setting changes
    if 'last_business_hours_setting' not in st.session_state:
        st.session_state.last_business_hours_setting = business_hours_only
    elif st.session_state.last_business_hours_setting != business_hours_only:
        st.session_state.comparison_results = None
        st.session_state.comparison_ai_insights = None
        st.session_state.last_business_hours_setting = business_hours_only
else:
    business_hours_only = False

# Category-specific options
if category == "Site Performance":
    with st.container(border=True):
        st.markdown("**Site Performance Options:**")
        site_comparison_type = st.radio(
            "Select comparison type:",
            options=["Site KPI Analysis", "Site Outreach Effectiveness"],
            key="comparison_site_type",
            horizontal=True,
            help="Site KPI Analysis: Compare overall performance scores and metrics\nSite Outreach Effectiveness: Compare time to first action effectiveness"
        )

        if site_comparison_type == "Site KPI Analysis":
            st.info("Comparing full site performance ranking table with all metrics.")
        else:
            st.info("Comparing how quickly sites take first action after receiving referrals and the impact on conversion rates.")

elif category == "Ad Performance":
    with st.container(border=True):
        st.markdown("**Ad Performance Options:**")
        table_type = st.radio(
            "Select comparison type:",
            options=["UTM Source", "UTM Source/Medium Combo"],
            key="comparison_ad_table_type",
            horizontal=True
        )

elif category == "PC Performance":
    with st.container(border=True):
        st.markdown("**PC Performance Options:**")
        pc_comparison_type = st.radio(
            "Select comparison type:",
            options=["Time Metrics Overview", "Contact Effectiveness", "Time to First Contact Effectiveness"],
            key="comparison_pc_type",
            horizontal=True
        )

elif category == "Funnel Analysis":
    with st.container(border=True):
        st.markdown("**Funnel Analysis Options:**")

        # Funnel rate method
        funnel_rate_method = st.radio(
            "Conversion Rate Method:",
            options=["Rolling Historical Average", "Manual Input"],
            index=0,
            key="comparison_funnel_rate_method",
            horizontal=True
        )

        if funnel_rate_method == "Rolling Historical Average":
            funnel_rolling_window = st.selectbox(
                "Rolling Window:",
                options=[1, 3, 6, 999],
                index=1,
                format_func=lambda x: "Overall Average" if x == 999 else f"{x}-Month",
                key="comparison_funnel_rolling_window"
            )
            funnel_manual_rates = {}
        else:
            funnel_rolling_window = 3
            st.markdown("**Manual Conversion Rates:**")
            cols_rate = st.columns(5)
            funnel_manual_rates = {
                f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}": cols_rate[0].slider("POF → PreScreen %", 0.0, 100.0, 95.0, key='comp_fa_cr_qps', format="%.1f%%") / 100.0,
                f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}": cols_rate[1].slider("PreScreen → StS %", 0.0, 100.0, 20.0, key='comp_fa_cr_pssts', format="%.1f%%") / 100.0,
                f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}": cols_rate[2].slider("StS → Appt %", 0.0, 100.0, 45.0, key='comp_fa_cr_sa', format="%.1f%%") / 100.0,
                f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}": cols_rate[3].slider("Appt → ICF %", 0.0, 100.0, 55.0, key='comp_fa_cr_ai', format="%.1f%%") / 100.0,
                f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}": cols_rate[4].slider("ICF → Enrolled %", 0.0, 100.0, 85.0, key='comp_fa_cr_ie', format="%.1f%%") / 100.0
            }

st.divider()

# --- STEP 4: Generate Comparison ---
st.subheader("4️⃣ Generate Comparison")

# Warn if scoring weights haven't been configured (only for KPI Analysis)
if category == "Site Performance" and site_comparison_type == "Site KPI Analysis":
    if 'ranked_sites_df' not in st.session_state or st.session_state.get('ranked_sites_df') is None or st.session_state.ranked_sites_df.empty:
        st.warning("⚠️ **Scoring Weights Not Configured:** This comparison will use default scoring weights. For accurate results, please visit the Site Performance page and click 'Apply & Recalculate Score' first.")
    st.info(f"💡 **Note:** Scores are calculated based on weights configured on the **{category}** page. Make sure you've configured and applied your preferred scoring weights before running this comparison for meaningful results.")
elif category == "Ad Performance":
    if 'ranked_ad_source_df' not in st.session_state or st.session_state.get('ranked_ad_source_df') is None or st.session_state.ranked_ad_source_df.empty:
        st.warning("⚠️ **Scoring Weights Not Configured:** This comparison will use default scoring weights. For accurate results, please visit the Ad Performance page and click 'Apply & Recalculate Score' first.")
    st.info(f"💡 **Note:** Scores are calculated based on weights configured on the **{category}** page. Make sure you've configured and applied your preferred scoring weights before running this comparison for meaningful results.")

if st.button("🔬 Generate Comparison", type="primary", use_container_width=True):
    with st.spinner("Calculating comparison..."):
        try:
            # Validate date ranges first
            validation = validate_date_ranges(start_a, end_a, start_b, end_b, df)

            if not validation['valid']:
                display_validation_messages(validation)
                st.stop()

            # Display warnings if any
            if validation.get('warnings'):
                display_validation_messages(validation)

            # Call appropriate comparison function based on category
            if category == "Site Performance":
                if site_comparison_type == "Site KPI Analysis":
                    # Get current weights from session state
                    weights = {
                        "StS to Enrollment %": st.session_state.get('w_site_sts_to_enr', 10),
                        "ICF to Enrollment %": st.session_state.get('w_site_icf_to_enroll', 10),
                        "StS to ICF %": st.session_state.get('w_site_sts_to_icf', 10),
                        "StS to Appt %": st.session_state.get('w_site_sts_appt', 10),
                        "StS Contact Rate %": st.session_state.get('w_site_contact_rate', 10),
                        "Average time to first site action": st.session_state.get('w_site_avg_time_to_first_action', 10),
                        "Avg time from StS to Appt Sched.": st.session_state.get('w_site_lag_sts_appt', 10),
                        "Avg. Time Between Site Contacts": st.session_state.get('w_site_avg_time_between_contacts', 10),
                        "Avg time from StS to ICF": st.session_state.get('w_site_lag_sts_icf', 10),
                        "Total Referrals Awaiting First Site Action": st.session_state.get('w_site_awaiting_action', 10),
                        "SF or Lost After ICF %": st.session_state.get('w_site_icf_to_lost', 10),
                        "StS to Lost %": st.session_state.get('w_site_sts_to_lost', 10),
                    }

                    # Normalize weights
                    total_weight = sum(abs(w) for w in weights.values())
                    weights_normalized = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {}

                    comparison_results = calculate_comparison_for_site_performance(
                        df=df,
                        date_range_a=(start_a, end_a),
                        date_range_b=(start_b, end_b),
                        ordered_stages=st.session_state.ordered_stages,
                        ts_col_map=st.session_state.ts_col_map,
                        weights=weights_normalized,
                        business_hours_only=business_hours_only
                    )
                else:  # Site Outreach Effectiveness
                    comparison_results = calculate_comparison_for_site_outreach(
                        df=df,
                        date_range_a=(start_a, end_a),
                        date_range_b=(start_b, end_b),
                        ts_col_map=st.session_state.ts_col_map,
                        ordered_stages=st.session_state.ordered_stages,
                        business_hours_only=business_hours_only
                    )

            elif category == "Ad Performance":
                # Get ad weights
                weights = {
                    "Qualified to Enrollment %": st.session_state.get('w_ad_qual_to_enroll', 10),
                    "ICF to Enrollment %": st.session_state.get('w_ad_icf_to_enroll', 10),
                    "Qualified to ICF %": st.session_state.get('w_ad_qual_to_icf', 10),
                    "StS to Appt %": st.session_state.get('w_ad_sts_to_appt', 10),
                    "Average time to first site action": st.session_state.get('w_ad_avg_time_to_first_action', 10),
                    "Avg time from StS to Appt Sched.": st.session_state.get('w_ad_lag_sts_appt', 10),
                    "Screen Fail % (from Qualified)": st.session_state.get('w_ad_generic_sf', 10),
                }

                total_weight = sum(abs(w) for w in weights.values())
                weights_normalized = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {}

                table_type_param = 'source' if table_type == "UTM Source" else 'combo'

                comparison_results = calculate_comparison_for_ad_performance(
                    df=df,
                    date_range_a=(start_a, end_a),
                    date_range_b=(start_b, end_b),
                    ordered_stages=st.session_state.ordered_stages,
                    ts_col_map=st.session_state.ts_col_map,
                    weights=weights_normalized,
                    table_type=table_type_param
                )

            elif category == "PC Performance":
                pc_type_map = {
                    "Time Metrics Overview": "time_metrics",
                    "Contact Effectiveness": "contact_effectiveness",
                    "Time to First Contact Effectiveness": "time_to_contact_effectiveness"
                }

                comparison_results = calculate_comparison_for_pc_performance(
                    df=df,
                    date_range_a=(start_a, end_a),
                    date_range_b=(start_b, end_b),
                    ts_col_map=st.session_state.ts_col_map,
                    comparison_type=pc_type_map[pc_comparison_type],
                    business_hours_only=business_hours_only
                )

            elif category == "Funnel Analysis":
                comparison_results = calculate_comparison_for_funnel(
                    df=df,
                    date_range_a=(start_a, end_a),
                    date_range_b=(start_b, end_b),
                    ordered_stages=st.session_state.ordered_stages,
                    ts_col_map=st.session_state.ts_col_map,
                    inter_stage_lags=st.session_state.inter_stage_lags,
                    comparison_type='full_projection',
                    rate_method=funnel_rate_method,
                    rolling_window=funnel_rolling_window,
                    manual_rates=funnel_manual_rates
                )

            # Store results in session state
            st.session_state.comparison_results = comparison_results
            st.session_state.stored_comparison_category = category  # Use different key to avoid widget conflict
            st.session_state.comparison_labels = (label_a, label_b)

            st.success("✅ Comparison generated successfully!")

        except Exception as e:
            st.error(f"Error generating comparison: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

st.divider()

# --- STEP 5: Display Results ---
if st.session_state.comparison_results and not st.session_state.comparison_results.get('error'):
    results = st.session_state.comparison_results
    stored_category = st.session_state.get('stored_comparison_category', category)
    stored_labels = st.session_state.get('comparison_labels', (label_a, label_b))
    label_a_stored, label_b_stored = stored_labels

    st.subheader(f"📊 Comparison Results: {stored_category}")
    st.caption(f"Comparing **{label_a_stored}** vs **{label_b_stored}**")

    # Display based on category and type
    if stored_category == "Site Performance" and results['type'] == 'full_table':
        st.markdown("### Site Performance Ranking Comparison")

        # Display summary stats first
        with st.container(border=True):
            st.markdown("#### Summary Statistics")
            key_metrics = ['Score', 'StS to Appt %', 'StS to ICF %', 'StS to Enrollment %', 'Average time to first site action']
            summary_df = create_summary_stats_table(results['comparison'], key_metrics, label_a_stored, label_b_stored)

            if not summary_df.empty:
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Full Comparison Table")

            # Create simplified comparison table
            comp_df = results['comparison'].copy()
            display_df = pd.DataFrame()

            # Key column
            display_df['Site'] = comp_df['Site']

            # Score columns
            if 'Score_A' in comp_df.columns:
                display_df[f'Score ({label_a_stored})'] = comp_df['Score_A'].round(1)
                display_df[f'Score ({label_b_stored})'] = comp_df['Score_B'].round(1)
                display_df['Δ Score'] = comp_df['Score_Delta'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "—")

            # Grade columns
            if 'Grade_A' in comp_df.columns:
                display_df[f'Grade ({label_a_stored})'] = comp_df['Grade_A']
                display_df[f'Grade ({label_b_stored})'] = comp_df['Grade_B']

            # Count metrics - show Period B with + sign, no separate change column
            count_metrics = ['Total Qualified', 'StS Count', 'Appt Count', 'ICF Count', 'Enrollment Count']
            for metric in count_metrics:
                if f'{metric}_A' in comp_df.columns:
                    display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].fillna(0).astype(int)
                    # Add + sign to Period B value
                    display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"+{x:,.0f}" if pd.notna(x) and x > 0 else f"{x:,.0f}" if pd.notna(x) else "—")

            # Percentage metrics - show percentage change (not percentage point)
            pct_metrics = ['StS to Appt %', 'StS to ICF %', 'StS to Enrollment %', 'ICF to Enrollment %']
            for metric in pct_metrics:
                if f'{metric}_A' in comp_df.columns:
                    display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                    display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                    # For percentages, show percentage change: (new-old)/old * 100
                    # Example: 25% to 30% = (30-25)/25 * 100 = 20% change
                    def calc_pct_change(row):
                        delta = row[f'{metric}_Delta']
                        base = row[f'{metric}_A']
                        if pd.notna(delta) and pd.notna(base) and base != 0:
                            pct_change = (delta / base) * 100
                            return f"{pct_change:+.1f}%"
                        return "—"
                    display_df[f'Δ {metric}'] = comp_df.apply(calc_pct_change, axis=1)

            # Time metrics - show in days with change
            time_metrics = ['Average time to first site action', 'Avg time from StS to Appt Sched.']
            for metric in time_metrics:
                if f'{metric}_A' in comp_df.columns:
                    display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].apply(lambda x: f"{x:.1f}d" if pd.notna(x) else "—")
                    display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"{x:.1f}d" if pd.notna(x) else "—")
                    display_df[f'Δ {metric}'] = comp_df[f'{metric}_Delta'].apply(lambda x: f"{x:+.1f}d" if pd.notna(x) else "—")

            # Display with formatting
            st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Top movers
        with st.container(border=True):
            st.markdown("#### Top Performance Changes")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📈 Most Improved Sites (by Score)**")
                top_improved = results['comparison'].nlargest(5, 'Score_Delta')[
                    ['Site', 'Score_A', 'Score_B', 'Score_Delta', 'Grade_A', 'Grade_B']
                ]
                st.dataframe(top_improved, hide_index=True, use_container_width=True)

            with col2:
                st.markdown("**📉 Most Declined Sites (by Score)**")
                top_declined = results['comparison'].nsmallest(5, 'Score_Delta')[
                    ['Site', 'Score_A', 'Score_B', 'Score_Delta', 'Grade_A', 'Grade_B']
                ]
                st.dataframe(top_declined, hide_index=True, use_container_width=True)

    elif stored_category == "Site Performance" and results['type'] == 'time_to_first_action':
        st.markdown("### Site Outreach Effectiveness Comparison")
        st.markdown("_Comparing how quickly sites take first action after receiving referrals_")

        # Display Outreach KPIs first
        st.markdown("### Outreach KPIs")

        # Get operational KPIs from results
        operational_kpis = results.get('operational_kpis', {})
        kpis_a = operational_kpis.get('period_a', {})
        kpis_b = operational_kpis.get('period_b', {})

        if kpis_a and kpis_b:
            # KPI 1: Avg. Time to First Site Action
            create_metric_card_comparison(
                metric_name="Avg. Time to First Site Action",
                value_a=kpis_a.get('avg_time_to_first_action', 0) or 0,
                value_b=kpis_b.get('avg_time_to_first_action', 0) or 0,
                label_a=label_a_stored,
                label_b=label_b_stored,
                format_type='days',
                is_inverse=True
            )

            # KPI 2: Avg. Time Between Site Contacts
            create_metric_card_comparison(
                metric_name="Avg. Time Between Site Contacts",
                value_a=kpis_a.get('avg_time_between_contacts', 0) or 0,
                value_b=kpis_b.get('avg_time_between_contacts', 0) or 0,
                label_a=label_a_stored,
                label_b=label_b_stored,
                format_type='days',
                is_inverse=True
            )

            # KPI 3: Avg. Time StS to Appt. Sched.
            create_metric_card_comparison(
                metric_name="Avg. Time StS to Appt. Sched.",
                value_a=kpis_a.get('avg_time_sts_to_appt', 0) or 0,
                value_b=kpis_b.get('avg_time_sts_to_appt', 0) or 0,
                label_a=label_a_stored,
                label_b=label_b_stored,
                format_type='days',
                is_inverse=True
            )

            # KPI 4: Referrals Awaiting Action > 7 Days
            create_metric_card_comparison(
                metric_name="Referrals Awaiting Action > 7 Days",
                value_a=kpis_a.get('referrals_awaiting_action_7d', 0) or 0,
                value_b=kpis_b.get('referrals_awaiting_action_7d', 0) or 0,
                label_a=label_a_stored,
                label_b=label_b_stored,
                format_type='number',
                is_inverse=True
            )

        # Check if comparison data exists
        if 'comparison' in results and not results['comparison'].empty:
            with st.container(border=True):
                st.markdown("#### Time to First Action Effectiveness")

                # Format the display table
                comp_df = results['comparison'].copy()

                # Define the correct order for time buckets (quickest to longest)
                time_bucket_order = ['< 4 Hours', '4 - 8 Hours', '8 - 24 Hours', '1 - 2 Days', '2 - 3 Days',
                                    '3 - 5 Days', '5 - 7 Days', '7 - 14 Days', '> 14 Days']

                # Sort by the correct time bucket order
                comp_df['Time to First Site Action'] = pd.Categorical(
                    comp_df['Time to First Site Action'],
                    categories=time_bucket_order,
                    ordered=True
                )
                comp_df = comp_df.sort_values('Time to First Site Action')

                display_df = pd.DataFrame()

                # Key column (time bucket)
                display_df['Time to First Site Action'] = comp_df['Time to First Site Action'].astype(str)

                # Count metrics - show Period A, Period B with + sign for change
                count_metrics = ['Attempts', 'Total_Appts', 'Total_ICF', 'Total_Enrolled']
                metric_labels = {
                    'Attempts': 'Total Referrals',
                    'Total_Appts': 'Total Appointments',
                    'Total_ICF': 'Total ICFs',
                    'Total_Enrolled': 'Total Enrollments'
                }

                for metric in count_metrics:
                    label = metric_labels.get(metric, metric)
                    if f'{metric}_A' in comp_df.columns:
                        display_df[f'{label} ({label_a_stored})'] = comp_df[f'{metric}_A'].fillna(0).astype(int)
                        # Add + sign to Period B value
                        display_df[f'{label} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(
                            lambda x: f"+{x:,.0f}" if pd.notna(x) and x > 0 else f"{x:,.0f}" if pd.notna(x) else "—"
                        )

                # Rate metrics - show as percentages
                rate_metrics = ['Appt_Rate', 'ICF_Rate', 'Enrollment_Rate']
                rate_labels = {
                    'Appt_Rate': 'Appt. Rate',
                    'ICF_Rate': 'ICF Rate',
                    'Enrollment_Rate': 'Enrollment Rate'
                }

                for metric in rate_metrics:
                    label = rate_labels.get(metric, metric)
                    if f'{metric}_A' in comp_df.columns:
                        # Convert decimal to percentage
                        display_df[f'{label} ({label_a_stored})'] = comp_df[f'{metric}_A'].apply(
                            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
                        )
                        display_df[f'{label} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(
                            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
                        )
                        # Show percentage change
                        def calc_pct_change(row, metric=metric):
                            delta = row[f'{metric}_Delta']
                            base = row[f'{metric}_A']
                            if pd.notna(delta) and pd.notna(base) and base != 0:
                                pct_change = (delta / base) * 100
                                return f"{pct_change:+.1f}%"
                            return "—"
                        display_df[f'Δ {label}'] = comp_df.apply(lambda row: calc_pct_change(row, metric), axis=1)

                st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No comparison data available. This may be due to insufficient data in the selected date ranges.")

    elif stored_category == "Ad Performance":
        st.markdown(f"### {results.get('key_column', 'Ad')} Performance Comparison")

        # Check if comparison data exists
        if 'comparison' in results and not results['comparison'].empty:
            # Display summary stats
            with st.container(border=True):
                st.markdown("#### Summary Statistics")
                key_metrics = ['Score', 'Qualified to StS %', 'Qualified to Appt %', 'Qualified to ICF %', 'Qualified to Enrollment %']
                summary_df = create_summary_stats_table(results['comparison'], key_metrics, label_a_stored, label_b_stored)

                if not summary_df.empty:
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)

            with st.container(border=True):
                st.markdown("#### Full Comparison Table")

                # Format the display table
                comp_df = results['comparison'].copy()
                display_df = pd.DataFrame()

                # Key column
                key_col = results.get('key_column', 'UTM Source')
                display_df[key_col] = comp_df[key_col]

                # Score columns
                if 'Score_A' in comp_df.columns:
                    display_df[f'Score ({label_a_stored})'] = comp_df['Score_A'].round(1)
                    display_df[f'Score ({label_b_stored})'] = comp_df['Score_B'].round(1)
                    display_df['Δ Score'] = comp_df['Score_Delta'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "—")

                # Grade columns
                if 'Grade_A' in comp_df.columns:
                    display_df[f'Grade ({label_a_stored})'] = comp_df['Grade_A']
                    display_df[f'Grade ({label_b_stored})'] = comp_df['Grade_B']

                # Count metrics - show Period B with + sign, no separate change column
                count_metrics = ['Total Qualified', 'StS Count', 'Appt Count', 'ICF Count', 'Enrollment Count', 'Screen Fail Count']
                for metric in count_metrics:
                    if f'{metric}_A' in comp_df.columns:
                        display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].fillna(0).astype(int)
                        # Add + sign to Period B value
                        display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"+{x:,.0f}" if pd.notna(x) and x > 0 else f"{x:,.0f}" if pd.notna(x) else "—")

                # Percentage metrics - show percentage change
                pct_metrics = ['Qualified to StS %', 'StS to Appt %', 'Qualified to Appt %', 'Qualified to ICF %', 'Qualified to Enrollment %', 'ICF to Enrollment %', 'Screen Fail % (from Qualified)']
                for metric in pct_metrics:
                    if f'{metric}_A' in comp_df.columns:
                        display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                        display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                        # Show percentage change
                        def calc_pct_change(row):
                            delta = row[f'{metric}_Delta']
                            base = row[f'{metric}_A']
                            if pd.notna(delta) and pd.notna(base) and base != 0:
                                pct_change = (delta / base) * 100
                                return f"{pct_change:+.1f}%"
                            return "—"
                        display_df[f'Δ {metric}'] = comp_df.apply(calc_pct_change, axis=1)

                # Time metrics - show in days with change
                time_metrics = ['Average time to first site action', 'Avg time from StS to Appt Sched.']
                for metric in time_metrics:
                    if f'{metric}_A' in comp_df.columns:
                        display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].apply(lambda x: f"{x:.1f}d" if pd.notna(x) else "—")
                        display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"{x:.1f}d" if pd.notna(x) else "—")
                        display_df[f'Δ {metric}'] = comp_df[f'{metric}_Delta'].apply(lambda x: f"{x:+.1f}d" if pd.notna(x) else "—")

                st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No comparison data available. This may be due to insufficient data in the selected date ranges.")

    elif stored_category == "PC Performance":
        if results['type'] == 'time_metrics':
            st.markdown("### Time Metrics Comparison")

            # Display each metric as comparison cards
            metrics_a = results['period_a']
            metrics_b = results['period_b']

            for metric_key, value_a in metrics_a.items():
                value_b = metrics_b.get(metric_key, 0)
                create_metric_card_comparison(
                    metric_name=metric_key,
                    value_a=value_a,
                    value_b=value_b,
                    label_a=label_a_stored,
                    label_b=label_b_stored,
                    format_type='days',
                    is_inverse=True
                )

        elif results['type'] == 'time_to_contact_effectiveness':
            st.markdown("### Time to First Contact Effectiveness Comparison")

            # Check if comparison data exists
            if 'comparison' in results and not results['comparison'].empty:
                with st.container(border=True):
                    st.markdown("#### Comparison Table")

                    # Format the display table with proper column ordering
                    comp_df = results['comparison'].copy()

                    # Define the correct order for time buckets (fastest to slowest)
                    time_bucket_order = ['<= 5 min', '5-15 min', '15-30 min', '30-60 min',
                                        '1-3 hours', '3-6 hours', '6-12 hours', '12-24 hours', '> 24 hours']

                    # Sort by the correct time bucket order
                    comp_df['Time to First Contact'] = pd.Categorical(
                        comp_df['Time to First Contact'],
                        categories=time_bucket_order,
                        ordered=True
                    )
                    comp_df = comp_df.sort_values('Time to First Contact')

                    display_df = pd.DataFrame()

                    # Key column (time bucket)
                    display_df['Time to First Contact'] = comp_df['Time to First Contact'].astype(str)

                    # Count metrics - show Period B with + sign, no separate change column
                    count_metrics = ['Attempts', 'Total_StS', 'Total_ICF', 'Total_Enrolled']
                    for metric in count_metrics:
                        if f'{metric}_A' in comp_df.columns:
                            display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].fillna(0).astype(int)
                            # Add + sign to Period B value
                            display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"+{x:,.0f}" if pd.notna(x) and x > 0 else f"{x:,.0f}" if pd.notna(x) else "—")

                    # Rate metrics - show as percentages
                    rate_metrics = ['StS_Rate', 'ICF_Rate', 'Enrollment_Rate']
                    for metric in rate_metrics:
                        if f'{metric}_A' in comp_df.columns:
                            # Convert decimal to percentage
                            display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                            display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                            # Show percentage change
                            def calc_pct_change(row):
                                delta = row[f'{metric}_Delta']
                                base = row[f'{metric}_A']
                                if pd.notna(delta) and pd.notna(base) and base != 0:
                                    pct_change = (delta / base) * 100
                                    return f"{pct_change:+.1f}%"
                                return "—"
                            display_df[f'Δ {metric}'] = comp_df.apply(calc_pct_change, axis=1)

                    st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No comparison data available. This may be due to insufficient data in the selected date ranges.")

        elif results['type'] == 'contact_effectiveness':
            st.markdown("### Contact Effectiveness Comparison")

            # Check if comparison data exists
            if 'comparison' in results and not results['comparison'].empty:
                with st.container(border=True):
                    st.markdown("#### Comparison Table")

                    # Format the display table with proper column ordering and percentage formatting
                    comp_df = results['comparison'].copy()
                    display_df = pd.DataFrame()

                    # Key column
                    display_df['Number of Attempts'] = comp_df['Number of Attempts']

                    # Count metrics - show Period B with + sign, no separate change column
                    count_metrics = ['Total Referrals', 'Total_StS', 'Total_ICF', 'Total_Enrolled']
                    for metric in count_metrics:
                        if f'{metric}_A' in comp_df.columns:
                            display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].fillna(0).astype(int)
                            # Add + sign to Period B value
                            display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"+{x:,.0f}" if pd.notna(x) and x > 0 else f"{x:,.0f}" if pd.notna(x) else "—")

                    # Rate metrics - show as percentages in order: A, B, Delta for each metric
                    rate_metrics = ['StS_Rate', 'ICF_Rate', 'Enrollment_Rate']
                    for metric in rate_metrics:
                        if f'{metric}_A' in comp_df.columns:
                            # Convert decimal to percentage
                            display_df[f'{metric} ({label_a_stored})'] = comp_df[f'{metric}_A'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                            display_df[f'{metric} ({label_b_stored})'] = comp_df[f'{metric}_B'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                            # Show percentage change
                            def calc_pct_change(row):
                                delta = row[f'{metric}_Delta']
                                base = row[f'{metric}_A']
                                if pd.notna(delta) and pd.notna(base) and base != 0:
                                    pct_change = (delta / base) * 100
                                    return f"{pct_change:+.1f}%"
                                return "—"
                            display_df[f'Δ {metric}'] = comp_df.apply(calc_pct_change, axis=1)

                    st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No comparison data available. This may be due to insufficient data in the selected date ranges.")

    elif stored_category == "Funnel Analysis":
        st.markdown("### Funnel Analysis Comparison")

        # Show validation warnings if rolling average method failed
        rates_desc_a = results['period_a'].get('rates_desc', '')
        rates_desc_b = results['period_b'].get('rates_desc', '')

        # Check if rolling average fallback occurred
        if 'fallback' in rates_desc_a.lower() or 'fallback' in rates_desc_b.lower() or \
           'manual' in rates_desc_a.lower() or 'manual' in rates_desc_b.lower():
            st.warning(f"⚠️ **Rolling Average Fallback:** Insufficient historical data for rolling average calculation. Using manual input rates instead.\n\n- Period A: {rates_desc_a}\n- Period B: {rates_desc_b}")

        # Display yield metrics
        col1, col2 = st.columns(2)

        results_a = results['period_a']['results']
        results_b = results['period_b']['results']

        with col1:
            with st.container(border=True):
                st.metric(
                    f"Expected ICF Yield - {label_a_stored}",
                    f"{results_a['total_icf_yield']:,.1f}"
                )
                st.metric(
                    f"Expected Enrollment Yield - {label_a_stored}",
                    f"{results_a['total_enroll_yield']:,.1f}"
                )

        with col2:
            with st.container(border=True):
                icf_delta = results_b['total_icf_yield'] - results_a['total_icf_yield']
                enroll_delta = results_b['total_enroll_yield'] - results_a['total_enroll_yield']

                st.metric(
                    f"Expected ICF Yield - {label_b_stored}",
                    f"{results_b['total_icf_yield']:,.1f}",
                    delta=f"{icf_delta:+,.1f}"
                )
                st.metric(
                    f"Expected Enrollment Yield - {label_b_stored}",
                    f"{results_b['total_enroll_yield']:,.1f}",
                    delta=f"{enroll_delta:+,.1f}"
                )

    st.divider()

    # --- STEP 6: AI Insights ---
    st.subheader("🤖 AI-Powered Insights")

    # Button to generate AI insights
    if st.button("Generate AI Insights", type="secondary", use_container_width=False):
        with st.spinner("Analyzing comparison data with AI..."):
            try:
                from comparison_ai import (
                    generate_site_performance_insights,
                    generate_ad_performance_insights,
                    generate_pc_performance_insights,
                    generate_funnel_insights,
                    generate_site_outreach_insights,
                    display_ai_insights
                )

                if stored_category == "Site Performance" and results['type'] == 'full_table':
                    insights = generate_site_performance_insights(
                        results['comparison'],
                        label_a_stored,
                        label_b_stored
                    )

                elif stored_category == "Site Performance" and results['type'] == 'time_to_first_action':
                    # Site Outreach Effectiveness - pass operational KPIs
                    insights = generate_site_outreach_insights(
                        results.get('operational_kpis', {}).get('period_a', {}),
                        results.get('operational_kpis', {}).get('period_b', {}),
                        label_a_stored,
                        label_b_stored
                    )

                elif stored_category == "Ad Performance":
                    table_type_param = 'source' if results.get('key_column', '') == 'UTM Source' else 'combo'
                    insights = generate_ad_performance_insights(
                        results['comparison'],
                        label_a_stored,
                        label_b_stored,
                        table_type_param
                    )

                elif stored_category == "PC Performance":
                    insights = generate_pc_performance_insights(
                        results['period_a'],
                        results['period_b'],
                        label_a_stored,
                        label_b_stored,
                        results['type']
                    )

                elif stored_category == "Funnel Analysis":
                    insights = generate_funnel_insights(
                        results['period_a']['results'],
                        results['period_b']['results'],
                        label_a_stored,
                        label_b_stored
                    )

                else:
                    insights = {
                        'key_changes': ["AI insights not yet supported for this comparison type"],
                        'management_summary': "Please try a different comparison type."
                    }

                # Store insights in session state
                st.session_state.comparison_ai_insights = insights

                # Display insights
                display_ai_insights(insights)

            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

    # Display cached insights if available
    elif st.session_state.comparison_ai_insights:
        from comparison_ai import display_ai_insights
        display_ai_insights(st.session_state.comparison_ai_insights)

elif st.session_state.comparison_results and st.session_state.comparison_results.get('error'):
    st.error("Error in comparison calculation. Please check your inputs and try again.")
    if 'validation' in st.session_state.comparison_results:
        display_validation_messages(st.session_state.comparison_results['validation'])
