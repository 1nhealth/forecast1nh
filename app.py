# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import io
import os

# Utility and Constant Imports
from parsing import parse_funnel_definition
from processing import preprocess_referral_data
from calculations import calculate_overall_inter_stage_lags, calculate_site_metrics
from constants import *
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(
    page_title="Recruitment Forecasting Tool",
    page_icon="assets/favicon.png", 
    layout="wide"
)

# --- Session State Initialization ---
required_keys = [
    'data_processed_successfully', 'referral_data_processed', 'funnel_definition',
    'ordered_stages', 'ts_col_map', 'site_metrics_calculated', 'inter_stage_lags',
    'historical_spend_df', 'ad_spend_input_dict',
    # --- OLD Scoring Weights ---
    'w_qual_to_enroll', 'w_icf_to_enroll', 'w_qual_to_icf', 'w_avg_ttc',
    'w_site_sf', 'w_sts_appt', 'w_appt_icf', 'w_lag_q_icf',
    'w_generic_sf', 'w_proj_lag',
    # --- NEW SITE-SPECIFIC Scoring Weights ---
    'w_site_awaiting_action', 'w_site_avg_time_between_contacts', 'w_site_contact_rate',
    'w_site_sts_to_icf', 'w_site_sts_to_enr', 'w_site_sts_to_lost',
    'w_site_icf_to_lost', 'w_site_lag_sts_appt', 'w_site_lag_sts_icf',
    'w_site_lag_sts_enr', 'w_site_qual_to_sts', 'w_site_qual_to_appt',
    'w_site_avg_time_to_first_action' # <-- NEW KEY
]
default_values = {
    'data_processed_successfully': False,
    'historical_spend_df': pd.DataFrame([
        {'Month (YYYY-MM)': (datetime.now() - pd.DateOffset(months=2)).strftime('%Y-%m'), 'Historical Spend': 45000.0},
        {'Month (YYYY-MM)': (datetime.now() - pd.DateOffset(months=1)).strftime('%Y-%m'), 'Historical Spend': 60000.0}
    ]),
    # --- OLD Default values ---
    'w_qual_to_enroll': 10, 'w_icf_to_enroll': 10, 'w_qual_to_icf': 20, 'w_avg_ttc': 10,
    'w_site_sf': 5, 'w_sts_appt': 15, 'w_appt_icf': 15, 'w_lag_q_icf': 5,
    'w_generic_sf': 5, 'w_proj_lag': 0,
    # --- NEW SITE-SPECIFIC Default values ---
    'w_site_awaiting_action': 5, 'w_site_avg_time_between_contacts': 10, 'w_site_contact_rate': 10,
    'w_site_sts_to_icf': 15, 'w_site_sts_to_enr': 20, 'w_site_sts_to_lost': 5,
    'w_site_icf_to_lost': 5, 'w_site_lag_sts_appt': 10, 'w_site_lag_sts_icf': 5,
    'w_site_lag_sts_enr': 0, 'w_site_qual_to_sts': 0, 'w_site_qual_to_appt': 0,
    'w_site_avg_time_to_first_action': 10 # <-- NEW DEFAULT
}
for key in required_keys:
    if key not in st.session_state:
        st.session_state[key] = default_values.get(key, None)

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")
    
    st.header("⚙️ Setup")
    st.info("Start here by uploading your data files.")
    st.warning("🔒 **Privacy Notice:** Do not upload files containing PII.", icon="⚠️")
    pii_checkbox = st.checkbox("I confirm my files do not contain PII.")

    if pii_checkbox:
        uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"])
        uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (CSV/TSV)", type=["csv", "tsv"])
    else:
        uploaded_referral_file = None
        uploaded_funnel_def_file = None

    st.divider()

    with st.expander("Historical Ad Spend"):
        edited_df = st.data_editor(st.session_state.historical_spend_df, num_rows="dynamic", key="hist_spend_editor")
        temp_spend_dict = {}
        valid_entries = True
        for _, row in edited_df.iterrows():
            try:
                if row['Month (YYYY-MM)'] and pd.notna(row['Historical Spend']):
                    month_period = pd.Period(row['Month (YYYY-MM)'], freq='M')
                    temp_spend_dict[month_period] = float(row['Historical Spend'])
            except Exception:
                st.error(f"Invalid month format: {row['Month (YYYY-MM)']}. Please use YYYY-MM.")
                valid_entries = False
                break
        if valid_entries:
            st.session_state.ad_spend_input_dict = temp_spend_dict
            st.session_state.historical_spend_df = edited_df

# --- Main Page Content ---
st.title("📊 Recruitment Forecasting Tool")
st.header("Home & Data Setup")

if uploaded_referral_file and uploaded_funnel_def_file:
    if st.button("Process Uploaded Data", type="primary", use_container_width=True):
        with st.spinner("Parsing files and processing data..."):
            try:
                referral_bytes_data = uploaded_referral_file.getvalue()
                header_df = pd.read_csv(io.BytesIO(referral_bytes_data), nrows=0, low_memory=False)
                pii_cols = [c for c in ["notes", "first name", "last name", "name", "phone", "email"] if c in [str(h).lower().strip() for h in header_df.columns]]

                if pii_cols:
                    original_col_names = [col for col in header_df.columns if str(col).lower().strip() in pii_cols]
                    st.error(f"PII Detected in columns: {', '.join(original_col_names)}. Please remove them and re-upload.", icon="🚫")
                    st.stop()
                
                funnel_def, ordered_st, ts_map = parse_funnel_definition(uploaded_funnel_def_file)
                
                if funnel_def and ordered_st and ts_map:
                    raw_df = pd.read_csv(io.BytesIO(referral_bytes_data))
                    processed_data = preprocess_referral_data(raw_df, funnel_def, ordered_st, ts_map)

                    if processed_data is not None and not processed_data.empty:
                        st.session_state.funnel_definition = funnel_def
                        st.session_state.ordered_stages = ordered_st
                        st.session_state.ts_col_map = ts_map
                        st.session_state.referral_data_processed = processed_data
                        st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(processed_data, ordered_st, ts_map)
                        st.session_state.site_metrics_calculated = calculate_site_metrics(processed_data, ordered_st, ts_map)
                        st.session_state.data_processed_successfully = True
                        st.success("Data processed successfully!")
                        st.rerun()
                    else:
                        st.error("Data processing failed after preprocessing.")
                else:
                    st.error("Funnel definition parsing failed.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)

if st.session_state.data_processed_successfully:
    st.success("Data is loaded and ready.")
    st.info("👈 Please select an analysis page from the sidebar to view the results.")
else:
    with st.container(border=True):
        st.info("👋 **Welcome to the Recruitment Forecasting Tool!**")
        st.markdown("""
            This application helps you analyze historical recruitment data to forecast future performance. To get started:

            1.  **Confirm No PII**: Check the box in the sidebar to confirm your files are free of Personally Identifiable Information.
            2.  **Upload Your Data**: Use the file uploaders in the sidebar to provide your referral data and funnel definition files.
            3.  **Process Data**: Click the "Process Uploaded Data" button that will appear above.
            4.  **Explore**: Once processing is complete, the analysis pages will become available in the sidebar.
        """)