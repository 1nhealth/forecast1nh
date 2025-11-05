# pages/1_Monthly_ProForma.py
import streamlit as st
import pandas as pd
from calculations import calculate_proforma_metrics
from helpers import format_performance_df, load_css

# --- Page Configuration ---
st.set_page_config(page_title="Monthly ProForma", page_icon="📅", layout="wide")

# Load custom CSS for branded theme
load_css("custom_theme.css")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📅 Monthly ProForma")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Historical Ad Spend Controls (Moved from app.py sidebar) ---
with st.container(border=True):
    st.subheader("Historical Ad Spend")
    st.caption("Edit the monthly ad spend data below. This data is used to calculate cost-per metrics in the ProForma table.")
    
    edited_df = st.data_editor(
        st.session_state.historical_spend_df, 
        num_rows="dynamic", 
        key="hist_spend_editor",
        use_container_width=True,
        column_config={
            "Month (YYYY-MM)": st.column_config.TextColumn(
                help="Enter month in YYYY-MM format (e.g., 2024-01)",
                required=True
            ),
            "Historical Spend": st.column_config.NumberColumn(
                help="Total ad spend for this month in dollars",
                format="$%.2f",
                min_value=0.0,
                required=True
            )
        }
    )
    
    # Process the edited data
    temp_spend_dict = {}
    valid_entries = True
    error_messages = []
    
    for idx, row in edited_df.iterrows():
        try:
            if row['Month (YYYY-MM)'] and pd.notna(row['Historical Spend']):
                # Validate month format
                month_period = pd.Period(row['Month (YYYY-MM)'], freq='M')
                temp_spend_dict[month_period] = float(row['Historical Spend'])
            elif row['Month (YYYY-MM)'] and pd.isna(row['Historical Spend']):
                error_messages.append(f"Row {idx + 1}: Missing spend value for {row['Month (YYYY-MM)']}")
                valid_entries = False
        except Exception as e:
            error_messages.append(f"Row {idx + 1}: Invalid month format '{row['Month (YYYY-MM)']}'. Please use YYYY-MM format.")
            valid_entries = False
    
    # Show errors if any
    if error_messages:
        for msg in error_messages:
            st.error(msg)
    
    # Update session state if valid
    if valid_entries and temp_spend_dict:
        st.session_state.ad_spend_input_dict = temp_spend_dict
        st.session_state.historical_spend_df = edited_df
    elif valid_entries and not temp_spend_dict:
        st.info("ℹ️ Add ad spend data to calculate cost metrics")

st.divider()

# --- Load Data from Session State ---
processed_data = st.session_state.referral_data_processed
ordered_stages = st.session_state.ordered_stages
ts_col_map = st.session_state.ts_col_map
ad_spend_dict = st.session_state.ad_spend_input_dict

# --- Main Page Logic ---
if processed_data is not None and not processed_data.empty and ad_spend_dict:
    proforma_df = calculate_proforma_metrics(
        processed_data,
        ordered_stages,
        ts_col_map,
        ad_spend_dict
    )

    if not proforma_df.empty:
        with st.container(border=True):
            st.subheader("Monthly Performance Data")
            st.caption("Historical performance metrics calculated from your referral data and ad spend")
            
            proforma_display = proforma_df.transpose()
            proforma_display.columns = [str(col) for col in proforma_display.columns]

            format_dict = {
                idx: ("${:,.2f}" if 'Cost' in idx or 'Spend' in idx else
                      ("{:.1%}" if '%' in idx else "{:,.0f}"))
                for idx in proforma_display.index
            }

            st.dataframe(
                proforma_display.style.format(format_dict, na_rep='-'), 
                use_container_width=True,
                height=600
            )

            try:
                csv_data = proforma_df.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download ProForma Data (CSV)",
                    data=csv_data,
                    file_name='monthly_proforma.csv',
                    mime='text/csv',
                    key='download_proforma',
                    type="primary"
                )
            except Exception as e:
                st.warning(f"Could not prepare data for download: {e}")

    else:
        st.warning("Could not generate the ProForma table. This may be due to a mismatch between the months in your data and the historical ad spend entered.")
        st.info("💡 **Tip**: Make sure your ad spend months match the months in your referral data.")
else:
    with st.container(border=True):
        st.info("👆 Please enter historical ad spend data above to generate the monthly ProForma analysis.")