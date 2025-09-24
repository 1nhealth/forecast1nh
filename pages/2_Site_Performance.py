# pages/2_Site_Performance.py
import streamlit as st
import pandas as pd
from scoring import score_sites
from helpers import format_performance_df, format_days_to_dhm
from calculations import (
    calculate_site_operational_kpis, 
    calculate_site_ttfc_effectiveness, 
    calculate_site_contact_attempt_effectiveness,
    calculate_site_performance_over_time
)
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Site Performance", page_icon="🏆", layout="wide")

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("🏆 Site Performance Dashboard")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Site Operational KPIs & Effectiveness Section ---
with st.container(border=True):
    st.subheader("Site Operational Analysis")
    st.markdown("Analyze site-level efficiency. Select a specific site or view the overall average for all metrics in this section.")

    site_list = st.session_state.site_metrics_calculated['Site'].unique().tolist()
    site_list = [site for site in site_list if site != "Unassigned Site"]
    options = ["Overall"] + sorted(site_list)
    
    selected_site = st.selectbox(
        "Select a Site to Analyze (or Overall)", 
        options=options,
        key="site_kpi_selector"
    )

    site_kpis = calculate_site_operational_kpis(
        st.session_state.referral_data_processed,
        st.session_state.ts_col_map,
        "Parsed_Lead_Status_History",
        selected_site
    )

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        value = site_kpis.get('avg_sts_to_first_action')
        st.metric(
            label="Avg. Time to First Site Action",
            value=format_days_to_dhm(value),
            help="Time from when a lead is 'Sent To Site' until the site takes any follow-up action (e.g., status change, appointment scheduled)."
        )
    with kpi_cols[1]:
        value = site_kpis.get('avg_time_between_site_contacts')
        st.metric(
            label="Avg. Time Between Site Contacts",
            value=format_days_to_dhm(value),
            help="The average time between explicit 'Contact Attempts' made by a site before an appointment is scheduled."
        )
    with kpi_cols[2]:
        value = site_kpis.get('avg_sts_to_appt')
        st.metric(
            label="Avg. Time StS to Appt. Sched.",
            value=format_days_to_dhm(value),
            help="The average total time from when a lead is 'Sent to Site' until an appointment is successfully scheduled."
        )

    st.divider()
    st.subheader(f"Time to First Action Effectiveness: {selected_site}")
    
    site_effectiveness_df = calculate_site_ttfc_effectiveness(
        st.session_state.referral_data_processed,
        st.session_state.ts_col_map,
        selected_site
    )

    if site_effectiveness_df.empty or site_effectiveness_df['Attempts'].sum() == 0:
        st.info(f"Not enough data for '{selected_site}' to analyze first action effectiveness.")
    else:
        display_df = site_effectiveness_df.copy()
        display_df['Appt. Rate'] = display_df['Appt_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df['ICF Rate'] = display_df['ICF_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df['Enrollment Rate'] = display_df['Enrollment_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        
        display_df.rename(columns={
            'Attempts': 'Total Referrals',
            'Total_Appts': 'Total Appointments',
            'Total_ICF': 'Total ICFs',
            'Total_Enrolled': 'Total Enrollments'
        }, inplace=True)
        
        final_cols = [
            'Time to First Site Action', 'Total Referrals',
            'Total Appointments', 'Appt. Rate',
            'Total ICFs', 'ICF Rate',
            'Total Enrollments', 'Enrollment Rate'
        ]
        
        st.dataframe(
            display_df[final_cols],
            hide_index=True,
            use_container_width=True
        )

    st.divider()
    st.subheader(f"Contact Attempt Effectiveness: {selected_site}")
    st.markdown("Analyzes how the number of site contact attempts (post-'Sent to Site') impacts downstream funnel conversions.")

    site_contact_effectiveness_df = calculate_site_contact_attempt_effectiveness(
        st.session_state.referral_data_processed,
        st.session_state.ts_col_map,
        "Parsed_Lead_Status_History",
        selected_site
    )

    if site_contact_effectiveness_df.empty or site_contact_effectiveness_df['Total Referrals'].sum() == 0:
        st.info(f"Not enough data for '{selected_site}' to analyze contact attempt effectiveness.")
    else:
        display_df_contact = site_contact_effectiveness_df.copy()
        display_df_contact['Appt. Rate'] = display_df_contact['Appt_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df_contact['ICF Rate'] = display_df_contact['ICF_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df_contact['Enrollment Rate'] = display_df_contact['Enrollment_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        
        display_df_contact.rename(columns={
            'Total_Appts': 'Total Appointments',
            'Total_ICF': 'Total ICFs',
            'Total_Enrolled': 'Total Enrollments'
        }, inplace=True)
        
        final_cols_contact = [
            'Number of Site Attempts', 'Total Referrals',
            'Total Appointments', 'Appt. Rate',
            'Total ICFs', 'ICF Rate',
            'Total Enrollments', 'Enrollment Rate'
        ]
        
        st.dataframe(
            display_df_contact[final_cols_contact],
            hide_index=True,
            use_container_width=True
        )
    
    st.divider()
    st.subheader(f"Performance Over Time (Weekly): {selected_site}")
    st.markdown("""
    Track key metrics on a weekly basis. For conversion rates, the solid line represents weeks with mature data, 
    while the **dotted line projects the recent trend** forward for weeks that are not yet mature.
    """)

    over_time_df = calculate_site_performance_over_time(
        st.session_state.referral_data_processed,
        st.session_state.ts_col_map,
        "Parsed_Lead_Status_History",
        selected_site
    )

    if over_time_df.empty:
        st.info(f"Not enough data for '{selected_site}' to generate a performance trend graph.")
    else:
        secondary_metric = 'Total Sent to Site per Week'
        
        primary_metric_options = {
            'Sent to Site -> Appointment %': ('Sent to Site -> Appointment % (Actual)', 'Sent to Site -> Appointment % (Projected)'),
            'Sent to Site -> ICF %': ('Sent to Site -> ICF % (Actual)', 'Sent to Site -> ICF % (Projected)'),
            'Sent to Site -> Enrollment %': ('Sent to Site -> Enrollment % (Actual)', 'Sent to Site -> Enrollment % (Projected)'),
            'Total Appointments per Week': ('Total Appointments per Week', None),
            'Average Time to First Site Action (Days)': ('Average Time to First Site Action (Days)', None),
        }

        available_options = [opt for opt, cols in primary_metric_options.items() if cols[0] in over_time_df.columns]
        
        if not available_options:
            st.warning("No performance metrics could be calculated for the trend chart.")
        else:
            primary_metric_display_name = st.selectbox(
                "Select a primary metric to display on the chart:",
                options=available_options,
                key=f"site_perf_time_selector_{selected_site.replace(' ', '_')}"
            )
            
            compare_with_volume = st.toggle(f"Compare with {secondary_metric}", value=True, key=f"site_perf_time_toggle_{selected_site.replace(' ', '_')}")
            
            actual_col, projected_col = primary_metric_options[primary_metric_display_name]

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # --- CHANGE: Plot Actuals Trace ---
            fig.add_trace(
                go.Scatter(x=over_time_df.index, y=over_time_df[actual_col], name=primary_metric_display_name,
                           mode='lines+markers', line=dict(color='#53CA97')),
                secondary_y=False,
            )
            # --- CHANGE: Plot Projected Trace (if it exists) ---
            if projected_col and projected_col in over_time_df.columns:
                fig.add_trace(
                    go.Scatter(x=over_time_df.index, y=over_time_df[projected_col], name="Projected Trend",
                               mode='lines', line=dict(color='#53CA97', dash='dot'), showlegend=False),
                    secondary_y=False,
                )

            if compare_with_volume and secondary_metric in over_time_df.columns:
                fig.add_trace(
                    go.Scatter(x=over_time_df.index, y=over_time_df[secondary_metric], name=secondary_metric, line=dict(dash='dot', color='gray')),
                    secondary_y=True,
                )

            fig.update_yaxes(title_text=f"<b>{primary_metric_display_name}</b>", secondary_y=False)
            if compare_with_volume and secondary_metric in over_time_df.columns:
                fig.update_yaxes(title_text=f"<b>{secondary_metric}</b>", secondary_y=True, showgrid=False)

            fig.update_layout(
                title_text=f"Weekly Trend for {selected_site}: {primary_metric_display_name}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Synced Assumption Controls ---
with st.expander("Adjust Performance Scoring Weights"):
    st.session_state.w_qual_to_enroll = st.slider("Qual (POF) -> Enrollment %", 0, 100, st.session_state.w_qual_to_enroll, key="w_q_enr_site")
    st.session_state.w_icf_to_enroll = st.slider("ICF -> Enrollment %", 0, 100, st.session_state.w_icf_to_enroll, key="w_icf_enr_site")
    st.session_state.w_qual_to_icf = st.slider("Qual (POF) -> ICF %", 0, 100, st.session_state.w_qual_to_icf, key="w_q_icf_site")
    st.session_state.w_avg_ttc = st.slider("Avg Time to Contact (Sites)", 0, 100, st.session_state.w_avg_ttc, help="Lower is better.", key="w_ttc_site")
    st.session_state.w_site_sf = st.slider("Site Screen Fail %", 0, 100, st.session_state.w_site_sf, help="Lower is better.", key="w_ssf_site")
    st.session_state.w_sts_appt = st.slider("StS -> Appt Sched %", 0, 100, st.session_state.w_sts_appt, key="w_sts_appt_site")
    st.session_state.w_appt_icf = st.slider("Appt Sched -> ICF %", 0, 100, st.session_state.w_appt_icf, key="w_appt_icf_site")
    st.session_state.w_lag_q_icf = st.slider("Lag Qual -> ICF (Days)", 0, 100, st.session_state.w_lag_q_icf, help="Lower is better.", key="w_lag_site")
    st.session_state.w_generic_sf = st.slider("Generic Screen Fail % (Ads)", 0, 100, st.session_state.w_generic_sf, help="Lower is better.", key="w_gsf_site")
    st.session_state.w_proj_lag = st.slider("Generic Projection Lag (Ads)", 0, 100, st.session_state.w_proj_lag, help="Lower is better.", key="w_gpl_site")
    st.caption("Changes will apply automatically and be reflected on the Ad Performance page.")

# --- Calculation Logic ---
weights = {
    "Qual to Enrollment %": st.session_state.w_qual_to_enroll, "ICF to Enrollment %": st.session_state.w_icf_to_enroll,
    "Qual -> ICF %": st.session_state.w_qual_to_icf, "Avg TTC (Days)": st.session_state.w_avg_ttc,
    "Site Screen Fail %": st.session_state.w_site_sf, "StS -> Appt %": st.session_state.w_sts_appt,
    "Appt -> ICF %": st.session_state.w_appt_icf, "Lag Qual -> ICF (Days)": st.session_state.w_lag_q_icf,
    "Screen Fail % (from ICF)": st.session_state.w_generic_sf, "Projection Lag (Days)": st.session_state.w_proj_lag,
}
total_weight = sum(abs(w) for w in weights.values())
weights_normalized = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {}

site_metrics = st.session_state.site_metrics_calculated

if site_metrics is not None and not site_metrics.empty and weights_normalized:
    ranked_sites_df = score_sites(site_metrics, weights_normalized)

    st.subheader("Key Performance Indicators (Overall)")
    
    total_qualified = site_metrics['Total Qualified'].sum() if 'Total Qualified' in site_metrics.columns else 0
    total_enrollments = site_metrics['Enrollment Count'].sum() if 'Enrollment Count' in site_metrics.columns else 0
    total_icfs = site_metrics['Signed ICF Count'].sum() if 'Signed ICF Count' in site_metrics.columns else 0
    
    overall_qual_to_icf_rate = (total_icfs / total_qualified) * 100 if total_qualified > 0 else 0

    kpi_cols = st.columns(3)
    with kpi_cols[0], st.container(border=True):
        st.metric(label="Total Qualified Leads", value=f"{total_qualified:,}")
    with kpi_cols[1], st.container(border=True):
        st.metric(label="Total Enrollments", value=f"{total_enrollments:,}")
    with kpi_cols[2], st.container(border=True):
        st.metric(label="Overall Qualified to ICF Rate", value=f"{overall_qual_to_icf_rate:.1f}%")
            
    st.divider()

    with st.container(border=True):
        st.subheader("Site Performance Ranking")
        
        display_cols = [
            'Site', 'Score', 'Grade', 
            'Total Qualified', 
            'Pre-Screening Activities Count', 'Sent To Site Count',
            'Appointment Scheduled Count', 'Signed ICF Count', 'Enrollment Count', 
            'Qual to Enrollment %', 'ICF to Enrollment %', 'Qual -> ICF %', 
            'POF -> PSA %', 'PSA -> StS %', 'StS -> Appt %', 'Appt -> ICF %', 
            'Avg TTC (Days)', 'Site Screen Fail %', 'Lag Qual -> ICF (Days)', 
            'Site Projection Lag (Days)'
        ]
        
        display_cols_exist = [col for col in display_cols if col in ranked_sites_df.columns]
        
        if display_cols_exist:
            final_display_df = ranked_sites_df[display_cols_exist]
            if not final_display_df.empty:
                rename_map = {
                    'Pre-Screening Activities Count': 'PSA Count',
                    'Sent To Site Count': 'StS Count',
                    'Appointment Scheduled Count': 'Appt Count',
                    'Signed ICF Count': 'ICF Count',
                    'Enrollment Count': 'Enrollments'
                }
                final_display_df = final_display_df.rename(columns=rename_map)
                
                formatted_df = format_performance_df(final_display_df)
                st.dataframe(formatted_df, hide_index=True, use_container_width=True)
            else:
                st.warning("Could not generate the site ranking table.")
        else:
            st.warning("None of the standard display columns were found.")
else:
    st.warning("Site metrics have not been calculated.")