# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import io
import os

from parsing import parse_funnel_definition
from processing import preprocess_referral_data
from calculations import calculate_overall_inter_stage_lags, calculate_enhanced_site_metrics, calculate_enhanced_ad_metrics
from constants import *
from helpers import format_performance_df, load_css

st.set_page_config(
    page_title="Recruitment Forecasting Tool",
    page_icon="assets/favicon.png", 
    layout="wide"
)

# Load custom CSS for branded theme
load_css("custom_theme.css")

# --- Session State Initialization ---

### FIX: Create a comprehensive list of ALL keys that depend on uploaded data.
# This now includes both the processed data (ingredients) and the calculated results (cooked meals).

# 1. Keys for raw and processed data
APP_DATA_KEYS = [
    'data_processed_successfully', 'referral_data_processed', 'funnel_definition',
    'ordered_stages', 'ts_col_map', 'inter_stage_lags',
    'historical_spend_df', 'ad_spend_input_dict',
    'enhanced_site_metrics_df', 'enhanced_ad_source_metrics_df', 'enhanced_ad_combo_metrics_df',
    'uploaded_file_name'
]

# 2. Keys for results calculated on other pages
APP_RESULT_KEYS = [
    'ranked_sites_df',
    'ranked_ad_source_df',
    'ranked_ad_combo_df',
    'funnel_analysis_results',
    'funnel_narrative_data',
    'funnel_analysis_rates_desc'
]

# 3. Keys for scoring weights that should be reset to default
SCORING_WEIGHT_KEYS = [
    'w_site_qual_to_enroll', 'w_site_icf_to_enroll', 'w_site_qual_to_icf',
    'w_site_awaiting_action', 'w_site_avg_time_between_contacts', 'w_site_contact_rate',
    'w_site_sts_to_icf', 'w_site_sts_to_enr', 'w_site_sts_to_lost', 'w_site_sts_appt',
    'w_site_icf_to_lost', 'w_site_lag_sts_appt', 'w_site_lag_sts_icf',
    'w_site_lag_sts_enr', 'w_site_qual_to_sts', 'w_site_qual_to_appt',
    'w_site_avg_time_to_first_action', 'w_ad_qual_to_enroll', 'w_ad_icf_to_enroll', 
    'w_ad_qual_to_icf', 'w_ad_sts_to_appt', 'w_ad_awaiting_action', 
    'w_ad_avg_time_between_contacts', 'w_ad_contact_rate', 'w_ad_sts_to_icf', 
    'w_ad_sts_to_enr', 'w_ad_sts_to_lost', 'w_ad_icf_to_lost', 'w_ad_lag_sts_appt', 
    'w_ad_lag_sts_icf', 'w_ad_lag_sts_enr', 'w_ad_qual_to_sts', 'w_ad_qual_to_appt',
    'w_ad_avg_time_to_first_action', 'w_ad_generic_sf'
]

# This state is only used for the reset button. It will NOT be initialized globally.
PAGE_UI_STATE_KEYS = [
    'chat',
    'messages',
    'gemini_model'
]

# Combine all keys that need to be initialized
ALL_APP_KEYS = APP_DATA_KEYS + APP_RESULT_KEYS + SCORING_WEIGHT_KEYS

default_values = {
    'data_processed_successfully': False,
    'historical_spend_df': pd.DataFrame([
        {'Month (YYYY-MM)': (datetime.now() - pd.DateOffset(months=2)).strftime('%Y-%m'), 'Historical Spend': 45000.0},
        {'Month (YYYY-MM)': (datetime.now() - pd.DateOffset(months=1)).strftime('%Y-%m'), 'Historical Spend': 60000.0}
    ]),
    'ranked_sites_df': pd.DataFrame(), # Initialize result DFs as empty
    'ranked_ad_source_df': pd.DataFrame(),
    'ranked_ad_combo_df': pd.DataFrame(),
    'w_site_qual_to_enroll': 10, 'w_site_icf_to_enroll': 10, 'w_site_qual_to_icf': 20,
    'w_site_awaiting_action': 5, 'w_site_avg_time_between_contacts': 10, 'w_site_contact_rate': 10,
    'w_site_sts_to_icf': 15, 'w_site_sts_to_enr': 20, 'w_site_sts_to_lost': 5, 'w_site_sts_appt': 15,
    'w_site_icf_to_lost': 5, 'w_site_lag_sts_appt': 10, 'w_site_lag_sts_icf': 5,
    'w_site_lag_sts_enr': 0, 'w_site_qual_to_sts': 0, 'w_site_qual_to_appt': 0,
    'w_site_avg_time_to_first_action': 10,
    'w_ad_qual_to_enroll': 10, 'w_ad_icf_to_enroll': 10, 'w_ad_qual_to_icf': 20,
    'w_ad_sts_to_appt': 15, 'w_ad_avg_time_to_first_action': 10, 'w_ad_lag_sts_appt': 10,
    'w_ad_generic_sf': 10,
}

# Initialize session state for all keys if they don't exist
for key in ALL_APP_KEYS:
    if key not in st.session_state:
        st.session_state[key] = default_values.get(key)


# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")
    
    st.header("⚙️ Setup")
    
    ### FIX: The UI in the sidebar will now depend on whether data is already loaded
    if not st.session_state.data_processed_successfully:
        st.info("Start here by uploading your data files.")
        st.warning("🔒 **Privacy Notice:** Do not upload files containing PII.", icon="⚠️")
        pii_checkbox = st.checkbox("I confirm my files do not contain PII.")

        if pii_checkbox:
            uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"])
            uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (CSV/TSV)", type=["csv", "tsv"])
        else:
            uploaded_referral_file = None
            uploaded_funnel_def_file = None
    else:
        st.success(f"Data from **{st.session_state.uploaded_file_name}** is loaded.")
        st.info("To analyze a new dataset, please use the 'Clear Data & Reset' button on the main page.")
        uploaded_referral_file = None
        uploaded_funnel_def_file = None

# --- Main Page Content ---
st.title("📊 Recruitment Forecasting Tool")
st.header("Home & Data Setup")

### FIX: Update the reset button to clear ALL data and result keys.
if st.button("🗑️ Clear Data & Start New Analysis", type="secondary"):
    # Clear all data, result, AND page-specific UI keys
    for key in APP_DATA_KEYS + APP_RESULT_KEYS + PAGE_UI_STATE_KEYS:
        if key in st.session_state:
            del st.session_state[key]
            
    # Reset all scoring weights to their defaults
    for key in SCORING_WEIGHT_KEYS:
        st.session_state[key] = default_values.get(key)
        
    st.toast("Data and settings have been reset!", icon="✅")
    import time
    time.sleep(1) 
    st.rerun()

st.divider()

### FIX: The main part of the page is now controlled by the 'data_processed_successfully' flag.
if st.session_state.data_processed_successfully:
    st.success(f"Data from **{st.session_state.uploaded_file_name}** is loaded and ready.")
    st.info("👈 Please select an analysis page from the sidebar to view the results.")
    st.toast("🚀 Data Ready!")

else:
    # This block runs only if no data has been loaded in the current session.
    if uploaded_referral_file and uploaded_funnel_def_file:
        if st.button("Process Uploaded Data", type="primary", use_container_width=True):
            with st.spinner("Parsing files and processing data... This may take a moment."):
                try:
                    # Store the filename for display later
                    st.session_state.uploaded_file_name = uploaded_referral_file.name

                    referral_bytes_data = uploaded_referral_file.getvalue()
                    header_df = pd.read_csv(io.BytesIO(referral_bytes_data), nrows=0, low_memory=False)
                    pii_cols = [c for c in ["notes", "first name", "last name", "name", "phone", "email"] if c in [str(h).lower().strip() for h in header_df.columns]]

                    if pii_cols:
                        original_col_names = [col for col in header_df.columns if str(col).lower().strip() in pii_cols]
                        st.error(f"PII Detected in columns: {', '.join(original_col_names)}. Please remove them and re-upload.", icon="🚫")
                        st.stop()
                    
                    funnel_def, ordered_st, ts_map = parse_funnel_definition(uploaded_funnel_def_file)
                    
                    if funnel_def and ordered_st and ts_map:
                        raw_df = pd.read_csv(io.BytesIO(referral_bytes_data), low_memory=False)
                        
                        # ALL processing now saves directly to session_state
                        st.session_state.funnel_definition = funnel_def
                        st.session_state.ordered_stages = ordered_st
                        st.session_state.ts_col_map = ts_map
                        
                        processed_data = preprocess_referral_data(raw_df, funnel_def, ordered_st, ts_map)

                        if processed_data is not None and not processed_data.empty:
                            st.session_state.referral_data_processed = processed_data
                            st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(processed_data, ordered_st, ts_map)
                            st.session_state.enhanced_site_metrics_df = calculate_enhanced_site_metrics(processed_data, ordered_st, ts_map, "Parsed_Lead_Status_History")
                            st.session_state.enhanced_ad_source_metrics_df = calculate_enhanced_ad_metrics(processed_data, ordered_st, ts_map, "UTM Source", "Unclassified Source")
                            
                            df_for_combo = processed_data.copy()
                            df_for_combo['UTM Source/Medium'] = df_for_combo['UTM Source'].astype(str).fillna("Unclassified") + ' / ' + df_for_combo['UTM Medium'].astype(str).fillna("Unclassified")
                            st.session_state.enhanced_ad_combo_metrics_df = calculate_enhanced_ad_metrics(df_for_combo, ordered_st, ts_map, "UTM Source/Medium", "Unclassified Combo")

                            # Set the master flag to True
                            st.session_state.data_processed_successfully = True
                            st.rerun() # Rerun to show the success message and hide the uploaders
                        else:
                            st.error("Data processing failed after preprocessing.")
                    else:
                        st.error("Funnel definition parsing failed.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.exception(e)
    
    # Show the welcome message if the button and files aren't ready yet
    with st.container(border=True):
        st.info("👋 **Welcome to the Recruitment Forecasting Tool!**")
        st.markdown("""
            This application helps you analyze historical recruitment data to forecast future performance. To get started:

            1.  **Confirm No PII**: Check the box in the sidebar to confirm your files are free of Personally Identifiable Information.
            2.  **Upload Your Data**: Use the file uploaders in the sidebar to provide your referral data and funnel definition files.
            3.  **Process Data**: Click the "Process Uploaded Data" button that will appear above.
            4.  **Explore**: Once processing is complete, the analysis pages will become available in the sidebar.
        """)