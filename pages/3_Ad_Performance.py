# pages/3_Ad_Performance.py
import streamlit as st
import pandas as pd

from calculations import calculate_enhanced_ad_metrics
from scoring import score_performance_groups
from helpers import format_performance_df

st.set_page_config(page_title="Ad Performance", page_icon="📢", layout="wide")

if 'ranked_ad_source_df' not in st.session_state:
    st.session_state.ranked_ad_source_df = pd.DataFrame()
if 'ranked_ad_combo_df' not in st.session_state:
    st.session_state.ranked_ad_combo_df = pd.DataFrame()

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📢 Ad Channel Performance")
st.info("Performance metrics grouped by UTM parameters. Adjust weights and click 'Apply' to recalculate scores.")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

with st.expander("Adjust Ad Performance Scoring Weights"):
    st.markdown("Adjust the importance of different metrics in the overall ad channel score. Changes here do not affect the Site Performance page.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Conversion Weights")
        st.slider("Qualified to Enrollment %", 0, 100, key="w_ad_qual_to_enroll")
        st.slider("Qualified to ICF %", 0, 100, key="w_ad_qual_to_icf")
        st.slider("ICF to Enrollment %", 0, 100, key="w_ad_icf_to_enroll")
    with c2:
        st.subheader("Negative Outcome & Lag Weights")
        st.markdown("_Lower is better for these metrics._")
        st.slider("Screen Fail % (from Qualified)", 0, 100, key="w_ad_generic_sf")
        st.slider("Projection Lag (Days)", 0, 100, key="w_ad_proj_lag")
    
    if st.button("Apply & Recalculate Score", type="primary", use_container_width=True, key="apply_ad_weights"):
        weights = {
            "Qualified to Enrollment %": st.session_state.w_ad_qual_to_enroll,
            "ICF to Enrollment %": st.session_state.w_ad_icf_to_enroll,
            "Qualified to ICF %": st.session_state.w_ad_qual_to_icf,
            "Screen Fail % (from Qualified)": st.session_state.w_ad_generic_sf,
            "Projection Lag (Days)": st.session_state.w_ad_proj_lag,
        }
        total_weight = sum(abs(w) for w in weights.values())
        weights_normalized = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {}
        
        # Recalculate for UTM Source
        utm_source_metrics_df = calculate_enhanced_ad_metrics(st.session_state.referral_data_processed, st.session_state.ordered_stages, st.session_state.ts_col_map, "UTM Source", "Unclassified Source")
        if not utm_source_metrics_df.empty:
            st.session_state.ranked_ad_source_df = score_performance_groups(utm_source_metrics_df, weights_normalized, "UTM Source")
        
        # Recalculate for UTM Source/Medium
        df_for_combo = st.session_state.referral_data_processed.copy()
        df_for_combo['UTM Source/Medium'] = df_for_combo['UTM Source'].astype(str).fillna("Unclassified") + ' / ' + df_for_combo['UTM Medium'].astype(str).fillna("Unclassified")
        utm_combo_metrics_df = calculate_enhanced_ad_metrics(df_for_combo, st.session_state.ordered_stages, st.session_state.ts_col_map, "UTM Source/Medium", "Unclassified Combo")
        if not utm_combo_metrics_df.empty:
            st.session_state.ranked_ad_combo_df = score_performance_groups(utm_combo_metrics_df, weights_normalized, "UTM Source/Medium")

# --- Performance by UTM Source ---
with st.container(border=True):
    st.subheader("Performance by UTM Source")
    if not st.session_state.ranked_ad_source_df.empty:
        display_cols_ad = ['UTM Source', 'Score', 'Grade', 'Total Qualified', 'ICF Count', 'Enrollment Count', 'Screen Fail Count', 'Qualified to ICF %', 'Qualified to Enrollment %', 'ICF to Enrollment %', 'Screen Fail % (from Qualified)', 'Projection Lag (Days)']
        display_cols_exist = [col for col in display_cols_ad if col in st.session_state.ranked_ad_source_df.columns]
        
        final_ad_display = st.session_state.ranked_ad_source_df[display_cols_exist]
        formatted_df = format_performance_df(final_ad_display)
        st.dataframe(formatted_df, hide_index=True, use_container_width=True)
    else:
        st.info("Adjust weights and click 'Apply & Recalculate Score' to generate the ranking table.")

st.write("") 

# --- Performance by UTM Source & Medium ---
if "UTM Medium" in st.session_state.referral_data_processed.columns:
    with st.container(border=True):
        st.subheader("Performance by UTM Source & Medium")
        if not st.session_state.ranked_ad_combo_df.empty:
            ranked_utm_combo_df = st.session_state.ranked_ad_combo_df.copy()
            if 'UTM Source/Medium' in ranked_utm_combo_df.columns:
                split_cols = ranked_utm_combo_df['UTM Source/Medium'].str.split(' / ', n=1, expand=True)
                ranked_utm_combo_df['UTM Source'] = split_cols[0]
                ranked_utm_combo_df['UTM Medium'] = split_cols[1]

            display_cols_combo = ['UTM Source', 'UTM Medium', 'Score', 'Grade', 'Total Qualified', 'ICF Count', 'Enrollment Count', 'Screen Fail Count', 'Qualified to ICF %', 'Qualified to Enrollment %', 'ICF to Enrollment %', 'Screen Fail % (from Qualified)', 'Projection Lag (Days)']
            display_cols_combo_exist = [col for col in display_cols_combo if col in ranked_utm_combo_df.columns]
            
            final_combo_display = ranked_utm_combo_df[display_cols_combo_exist]
            formatted_df_combo = format_performance_df(final_combo_display)
            st.dataframe(formatted_df_combo, hide_index=True, use_container_width=True)
        else:
            st.info("Adjust weights and click 'Apply & Recalculate Score' to generate the ranking table.")