# pages/7_AI_Analyst.py
import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from io import StringIO
import traceback
import matplotlib.pyplot as plt
import altair as alt
import matplotlib.dates as mdates
import re
import plotly.graph_objects as go
import plotly.express as px
import sys

from constants import *
from helpers import format_performance_df, load_css

st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

# Load custom CSS for branded theme
load_css("custom_theme.css")

# PAGE-SPECIFIC CSS: Chat input styling - OPTIMIZED VERSION
st.markdown("""
<style>
/* AI ANALYST PAGE ONLY - CHAT INPUT STYLING */

/* Nuclear option - catches ALL elements (makes first 3 blocks redundant) */
[data-testid="stChatInput"] *:not(button):not(svg):not(path):not(circle):not(rect),
[data-testid="stChatInputContainer"] *:not(button):not(svg):not(path):not(circle):not(rect) {
    background-color: #FFFFFF !important;
    border-color: #dfe2e6;
}

/* Textarea specific styling */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    border: none !important;
    color: #1b2222 !important;
}

/* Focus state - green border */
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInputContainer"]:focus-within {
    border-color: #53CA97 !important;
}

/* Send button - white background */
[data-testid="stChatInput"] button,
[data-testid="stChatInputContainer"] button {
    background-color: #FFFFFF !important;
    border: none !important;
}

/* Button icon - grey default */
[data-testid="stChatInput"] button svg,
[data-testid="stChatInput"] button svg path,
[data-testid="stChatInputContainer"] button svg,
[data-testid="stChatInputContainer"] button svg path {
    color: #6B7280 !important;
    fill: #6B7280 !important;
}

/* Button icon - green on hover and click (combined) */
[data-testid="stChatInput"] button:hover svg,
[data-testid="stChatInput"] button:hover svg path,
[data-testid="stChatInput"] button:active svg,
[data-testid="stChatInput"] button:active svg path,
[data-testid="stChatInputContainer"] button:hover svg,
[data-testid="stChatInputContainer"] button:hover svg path,
[data-testid="stChatInputContainer"] button:active svg,
[data-testid="stChatInputContainer"] button:active svg path {
    color: #53CA97 !important;
    fill: #53CA97 !important;
}

/* Placeholder text */
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: rgba(27, 34, 34, 0.5) !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("🤖 Strategic AI Analyst")
st.info("""
This AI Analyst is now a conversational partner. It remembers your previous questions and can use its Python tool to analyze data, find insights, and even correct its own mistakes. Start by asking a question!
""")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data and Config from Session State ---
df = st.session_state.referral_data_processed
ts_col_map = st.session_state.ts_col_map
ordered_stages = st.session_state.ordered_stages
status_history_col = "Parsed_Lead_Status_History"
site_performance_df = st.session_state.enhanced_site_metrics_df
utm_performance_df = st.session_state.enhanced_ad_source_metrics_df

# --- Configure the Gemini API (Session-Specific for User Isolation) ---
if "gemini_model" not in st.session_state:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Create model in session state - each user gets their own instance
        generation_config = genai.GenerationConfig(temperature=0.2)
        st.session_state.gemini_model = genai.GenerativeModel(
            'gemini-flash-latest',
            generation_config=generation_config
        )
    except Exception as e:
        st.error("Error configuring the AI model. Have you set your GEMINI_API_KEY in Streamlit's secrets?")
        st.exception(e)
        st.stop()

# Use the session-specific model
model = st.session_state.gemini_model

# --- System Prompts for the Advanced Agent ---
def get_df_info(df):
    if df is None: # Add a check for None to be safe
        return "DataFrame is not available."
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def get_system_prompt():
    # Use .get() for safety in case a dataframe doesn't exist yet
    site_perf_info = get_df_info(st.session_state.get('enhanced_site_metrics_df'))
    utm_perf_info = get_df_info(st.session_state.get('enhanced_ad_source_metrics_df'))
    raw_df_info = get_df_info(st.session_state.get('referral_data_processed'))

    return f"""You are an expert-level Python data analyst specializing in clinical trial patient recruitment analytics. Your goal is to be a full partner in data analysis for the user.

====================
DOMAIN CONTEXT - Clinical Trial Recruitment
====================

You are analyzing patient recruitment data for clinical trials. The funnel follows this progression:
1. **Passed Online Form (POF)** - Initial qualified referrals who passed screening
2. **Pre-Screening Activities (PSA)** - Additional screening/qualification steps
3. **Sent To Site (StS)** - Referral assigned to a study site (critical handoff point)
4. **Appointment Scheduled** - Patient has scheduled a site visit
5. **Signed ICF (Informed Consent Form)** - Patient officially enrolled in study
6. **Enrolled** - Final enrollment confirmed
7. **Terminal stages:** Lost or Screen Failed (exit points from the funnel)

**Key Concepts:**
- "Qualified referrals" = leads that passed the online form (POF)
- "Conversion rate" = percentage of referrals advancing from one stage to the next
- "StS" is the critical handoff from central patient coordinators to study sites
- Sites are measured on their ability to convert StS → Appointment → ICF → Enrollment
- Each referral has ONE timestamp per stage; non-null timestamp = referral reached that stage

====================
AVAILABLE DATAFRAMES
====================

You have access to three pandas DataFrames:

1. **`df`** - Raw, event-level referral data with all timestamp columns
2. **`site_performance_df`** - Pre-computed site-level metrics (USE THIS for site comparisons!)
3. **`utm_performance_df`** - Pre-computed ad source metrics (USE THIS for marketing analysis!)

**Important:** site_performance_df and utm_performance_df already contain calculated conversion rates, lag times, and operational KPIs. Always check if metrics already exist there before recalculating!

**Key Variables Available:**
- `ts_col_map` - Dictionary mapping stage names to timestamp column names
  Example: ts_col_map['Sent To Site'] → 'TS_Sent_To_Site'
- `ordered_stages` - List of funnel stages in sequential order

====================
CRITICAL: MATURITY ADJUSTMENTS FOR ACCURATE METRICS
====================

**Why Maturity Matters:**
Recent cohorts haven't had time to progress through the funnel. Including them will artificially lower conversion rates and produce misleading trends.

**Best Practice for Time-Based Analysis:**
1. Calculate average lag time for the stage transition
2. Set maturity period = 1.5 × average lag time (minimum 20 days)
3. Only calculate conversion rates for periods where: period_end + maturity_days < today
4. For immature periods, return np.nan or mark as "Pending"

**Example - Weekly Funnel with Maturity:**
```python
from helpers import calculate_avg_lag_generic

# Calculate maturity period
sts_ts_col = ts_col_map['Sent To Site']
icf_ts_col = ts_col_map['Signed ICF']
avg_sts_to_icf_lag = calculate_avg_lag_generic(df, sts_ts_col, icf_ts_col)
maturity_days = (avg_sts_to_icf_lag * 1.5) if pd.notna(avg_sts_to_icf_lag) else 45

# Index by stage timestamp you're measuring FROM
time_df = df[df[sts_ts_col].notna()].set_index(sts_ts_col)

def get_weekly_metrics(week_df):
    # Check if week is mature enough
    week_end = week_df.index.max()
    is_mature = (week_end + pd.Timedelta(days=maturity_days)) < pd.Timestamp.now()

    # Calculate rates only for mature weeks
    if is_mature and len(week_df) > 0:
        sts_to_icf_rate = week_df[icf_ts_col].notna().sum() / len(week_df)
    else:
        sts_to_icf_rate = np.nan

    return pd.Series({{
        'StS Count': len(week_df),
        'StS → ICF %': sts_to_icf_rate
    }})

weekly_summary = time_df.resample('W').apply(get_weekly_metrics)
st.dataframe(weekly_summary)
```

**When Maturity is NOT Needed:**
- Counting current volume (e.g., "How many referrals this month?")
- Analyzing completed transitions only (e.g., "Average lag between StS and Appointment")
- Looking at pre-computed dataframes (maturity already applied)

====================
AVAILABLE HELPER FUNCTIONS - USE THESE!
====================

**From helpers module (already imported):**
- `calculate_avg_lag_generic(df, col_from, col_to, business_hours_only=False)`
  - Calculates average time between two timestamp columns
  - Use business_hours_only=True for site operational metrics (Mon-Fri 9am-5pm)
  - Returns float (days) or np.nan if no valid transitions

- `format_performance_df(df)`
  - Formats metrics for display (percentages, counts, time lags)
  - Use this before displaying site_performance_df or utm_performance_df

**From calculations module (available in codebase):**
- Site/UTM metrics are already calculated and available in pre-computed dataframes
- Don't recalculate what already exists!

====================
COMMON QUERY PATTERNS - RECOGNIZE THESE
====================

**Pattern 1: "Show me weekly/monthly funnel data"**
→ User wants: Time-series view of funnel stage counts and conversion rates
→ Implementation: Use .resample('W') or .groupby('Submission_Month') WITH maturity filtering
→ Group by timestamp of the STARTING stage for each conversion rate

**Pattern 2: "How are sites performing?" or "Compare sites"**
→ User wants: Site comparison on conversion rates and operational efficiency
→ Implementation: Use pre-computed site_performance_df, sort by key metrics, consider formatting

**Pattern 3: "What's our conversion rate from X to Y?"**
→ User wants: Overall or time-based conversion calculation
→ Implementation:
  - Overall: count(reached_Y) / count(reached_X) using .notna().sum()
  - Time-based: Apply maturity filtering based on avg lag(X→Y)

**Pattern 4: "How long does it take to get from X to Y?"**
→ User wants: Average lag time calculation
→ Implementation: Use calculate_avg_lag_generic(df, ts_col_X, ts_col_Y)

**Pattern 5: "Show me ad source performance" or "UTM analysis"**
→ User wants: Marketing channel effectiveness
→ Implementation: Use pre-computed utm_performance_df, already has conversion rates by source

**Pattern 6: "Business hours only" or "Site operational metrics"**
→ User wants: Metrics calculated only during working hours (Mon-Fri 9am-5pm)
→ Implementation: Pass business_hours_only=True to calculate_avg_lag_generic

====================
DATA STRUCTURE GUIDE
====================

**Timestamp Columns (access via ts_col_map):**
- ts_col_map['Passed Online Form'] → 'TS_Passed_Online_Form'
- ts_col_map['Sent To Site'] → 'TS_Sent_To_Site'
- ts_col_map['Appointment Scheduled'] → 'TS_Appointment_Scheduled'
- ts_col_map['Signed ICF'] → 'TS_Signed_ICF'
- ts_col_map['Enrolled'] → 'TS_Enrolled'

**Key df Columns:**
- Submission_Month: Period - for cohort grouping
- Site: str - study site assignment
- UTM Source, UTM Medium, UTM Campaign: str - marketing attribution
- Submitted On_DT: datetime - original submission timestamp

**Pre-computed DataFrame Columns:**
- Conversion rate columns are decimals (0-1 range), not percentages!
- Lag time columns are in days (float)
- Count columns are integers

====================
CODE EXAMPLES FOR COMMON QUERIES
====================

**Example 1: Weekly Funnel Volume (No Maturity Needed)**
```python
# Count referrals reaching each stage per week
pof_ts_col = ts_col_map['Passed Online Form']
sts_ts_col = ts_col_map['Sent To Site']
appt_ts_col = ts_col_map['Appointment Scheduled']
icf_ts_col = ts_col_map['Signed ICF']

weekly_volume = df.groupby(pd.Grouper(key='Submitted On_DT', freq='W')).agg({{
    'Qualified (POF)': (pof_ts_col, lambda x: x.notna().sum()),
    'Sent to Site': (sts_ts_col, lambda x: x.notna().sum()),
    'Appointments': (appt_ts_col, lambda x: x.notna().sum()),
    'ICF Signed': (icf_ts_col, lambda x: x.notna().sum())
}})

st.dataframe(weekly_volume)
```

**Example 2: Site Comparison (Use Pre-computed!)**
```python
# Use the pre-computed site_performance_df
from helpers import format_performance_df

# Select key metrics
site_comparison = site_performance_df[[
    'Site',
    'StS Count',
    'StS to Appt %',
    'StS to ICF %',
    'Avg time from StS to Appt Sched.'
]].copy()

# Sort by ICF conversion rate
site_comparison = site_comparison.sort_values('StS to ICF %', ascending=False)

# Format and display
formatted_df = format_performance_df(site_comparison)
st.dataframe(formatted_df)
```

**Example 3: Weekly Conversion Rates with Maturity**
```python
from helpers import calculate_avg_lag_generic

# Setup
sts_ts_col = ts_col_map['Sent To Site']
appt_ts_col = ts_col_map['Appointment Scheduled']

# Calculate maturity
avg_lag = calculate_avg_lag_generic(df, sts_ts_col, appt_ts_col)
maturity_days = (avg_lag * 1.5) if pd.notna(avg_lag) else 30

# Create weekly cohorts based on when they reached StS
sts_df = df[df[sts_ts_col].notna()].set_index(sts_ts_col)

def calc_weekly(week_df):
    week_end = week_df.index.max()
    is_mature = (week_end + pd.Timedelta(days=maturity_days)) < pd.Timestamp.now()

    sts_count = len(week_df)
    appt_count = week_df[appt_ts_col].notna().sum()

    return pd.Series({{
        'StS Count': sts_count,
        'Appointments': appt_count,
        'StS → Appt %': (appt_count / sts_count) if (is_mature and sts_count > 0) else np.nan
    }})

weekly = sts_df.resample('W').apply(calc_weekly)
st.dataframe(weekly)
```

====================
YOUR WORKFLOW
====================

1. **Reason:** Understand the user's intent. What metrics do they want? What time frame? What grouping?
2. **Check Pre-computed Data:** Before writing custom code, check if site_performance_df or utm_performance_df already has what you need!
3. **Act:** If you need custom analysis, write Python code in ```python ... ``` block. Your code MUST end with a display command (st.dataframe(), st.plotly_chart(), print()).
4. **Apply Best Practices:**
   - Use maturity filtering for time-based conversion rates
   - Access timestamp columns via ts_col_map
   - Use helper functions instead of reinventing calculations
   - Format output for readability
5. **Observe & Summarize:** After code execution, provide a comprehensive, user-friendly summary with insights and recommendations.
6. **Clarify if Ambiguous:** If the request is vague, ask specific questions rather than making assumptions.

====================
DATAFRAME SCHEMAS
====================

**site_performance_df Schema:**
{site_perf_info}

**utm_performance_df Schema:**
{utm_perf_info}

**Raw df Schema:**
{raw_df_info}

====================
BEGIN
====================

Introduce yourself as the AI Analyst and ask what the user would like to analyze.
"""

# --- Conversational Chat Logic ---

### FIX: This is the corrected, robust initialization block. ###
# It handles all cases: the very first visit, or a visit after a full data reset.
if "messages" not in st.session_state or "chat" not in st.session_state:
    # Initialize the Gemini model chat
    st.session_state.chat = model.start_chat(history=[])
    
    # Send the initial system prompt to get the welcome message
    initial_response = st.session_state.chat.send_message(get_system_prompt())

    # Initialize the messages list with the welcome message
    st.session_state.messages = [{"role": "assistant", "content": initial_response.text}]

# Display existing messages (This line is now safe to run)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Send user prompt to the model and get the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(user_prompt)
            
            # Check if the response contains code to be executed
            code_match = re.search(r"```python\s*([\s\S]+?)```", response.text)
            
            if code_match:
                code_to_execute = code_match.group(1).strip()
                with st.expander("View AI's Generated Code", expanded=True):
                    st.code(code_to_execute, language="python")

                # Execute the code and capture the output
                st.markdown("**Execution Result:**")
                result_display_area = st.container()
                output_buffer = StringIO()
                sys.stdout = output_buffer
                
                # Create a storage for captured display data
                captured_data = []
                # Store actual DataFrames for download functionality
                captured_dataframes = []
                
                def get_dataframe_summary(df):
                    """Extract comprehensive summary statistics from a DataFrame"""
                    summary_parts = []
                    
                    # Basic info
                    summary_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                    
                    # Show the actual data (first 20 rows, all columns)
                    summary_parts.append("\nData Preview (first 20 rows):")
                    summary_parts.append(df.head(20).to_string())
                    
                    # If there are more rows, show the last few too
                    if len(df) > 20:
                        summary_parts.append(f"\n... ({len(df) - 20} more rows)")
                        summary_parts.append("\nLast 5 rows:")
                        summary_parts.append(df.tail(5).to_string())
                    
                    # Statistical summary for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        summary_parts.append("\nStatistical Summary for Numeric Columns:")
                        for col in numeric_cols:
                            stats = df[col].describe()
                            summary_parts.append(f"\n{col}:")
                            summary_parts.append(f"  Mean: {stats['mean']:.2f}")
                            summary_parts.append(f"  Median (50%): {stats['50%']:.2f}")
                            summary_parts.append(f"  Std Dev: {stats['std']:.2f}")
                            summary_parts.append(f"  Min: {stats['min']:.2f}")
                            summary_parts.append(f"  Max: {stats['max']:.2f}")
                            summary_parts.append(f"  Range: {stats['max'] - stats['min']:.2f}")
                    
                    return "\n".join(summary_parts)
                
                # Create wrapper functions that capture AND display
                def capture_dataframe(data, *args, **kwargs):
                    """Wrapper for st.dataframe that captures the data with statistics"""
                    if isinstance(data, pd.DataFrame):
                        captured_data.append(("dataframe", get_dataframe_summary(data)))
                        captured_dataframes.append(("dataframe", data.copy()))
                    elif isinstance(data, pd.Series):
                        df_from_series = data.to_frame()
                        captured_data.append(("dataframe", get_dataframe_summary(df_from_series)))
                        captured_dataframes.append(("dataframe", df_from_series.copy()))
                    elif isinstance(data, (list, dict)):
                        captured_data.append(("dataframe", str(data)))
                    return st.dataframe(data, *args, **kwargs)
                
                def capture_table(data, *args, **kwargs):
                    """Wrapper for st.table that captures the data with statistics"""
                    if isinstance(data, pd.DataFrame):
                        captured_data.append(("table", get_dataframe_summary(data)))
                        captured_dataframes.append(("table", data.copy()))
                    elif isinstance(data, pd.Series):
                        df_from_series = data.to_frame()
                        captured_data.append(("table", get_dataframe_summary(df_from_series)))
                        captured_dataframes.append(("table", df_from_series.copy()))
                    elif isinstance(data, (list, dict)):
                        captured_data.append(("table", str(data)))
                    return st.table(data, *args, **kwargs)
                
                def capture_write(*args, **kwargs):
                    """Wrapper for st.write that captures the data with statistics"""
                    for arg in args:
                        if isinstance(arg, pd.DataFrame):
                            captured_data.append(("write", get_dataframe_summary(arg)))
                            captured_dataframes.append(("write", arg.copy()))
                        elif isinstance(arg, pd.Series):
                            df_from_series = arg.to_frame()
                            captured_data.append(("write", get_dataframe_summary(df_from_series)))
                            captured_dataframes.append(("write", df_from_series.copy()))
                        elif isinstance(arg, (list, dict)):
                            captured_data.append(("write", str(arg)))
                        else:
                            captured_data.append(("write", str(arg)))
                    return st.write(*args, **kwargs)
                
                def capture_metric(label, value, *args, **kwargs):
                    """Wrapper for st.metric that captures the data"""
                    captured_data.append(("metric", f"{label}: {value}"))
                    return st.metric(label, value, *args, **kwargs)
                
                def capture_plotly_chart(fig, *args, **kwargs):
                    """Wrapper for st.plotly_chart that extracts actual data values"""
                    try:
                        chart_info_parts = []
                        
                        if hasattr(fig, 'data') and fig.data:
                            chart_info_parts.append(f"Chart Type: Plotly with {len(fig.data)} trace(s)")
                            
                            # Extract layout info
                            if hasattr(fig, 'layout'):
                                layout = fig.layout
                                if hasattr(layout, 'title') and layout.title:
                                    chart_info_parts.append(f"Title: {layout.title.text if hasattr(layout.title, 'text') else layout.title}")
                                if hasattr(layout, 'xaxis') and hasattr(layout.xaxis, 'title'):
                                    chart_info_parts.append(f"X-axis: {layout.xaxis.title.text if hasattr(layout.xaxis.title, 'text') else layout.xaxis.title}")
                                if hasattr(layout, 'yaxis') and hasattr(layout.yaxis, 'title'):
                                    chart_info_parts.append(f"Y-axis: {layout.yaxis.title.text if hasattr(layout.yaxis.title, 'text') else layout.yaxis.title}")
                            
                            # Extract data from each trace
                            for i, trace in enumerate(fig.data):
                                trace_info = [f"\nTrace {i+1}:"]
                                
                                if hasattr(trace, 'type'):
                                    trace_info.append(f"  Type: {trace.type}")
                                if hasattr(trace, 'name') and trace.name:
                                    trace_info.append(f"  Name: {trace.name}")
                                
                                # Extract actual x and y data
                                if hasattr(trace, 'x') and trace.x is not None:
                                    x_data = list(trace.x)
                                    if len(x_data) <= 50:
                                        trace_info.append(f"  X values: {x_data}")
                                    else:
                                        trace_info.append(f"  X values (first 10): {x_data[:10]}")
                                        trace_info.append(f"  X values (last 10): {x_data[-10:]}")
                                        trace_info.append(f"  Total X points: {len(x_data)}")
                                
                                if hasattr(trace, 'y') and trace.y is not None:
                                    y_data = list(trace.y)
                                    if len(y_data) <= 50:
                                        trace_info.append(f"  Y values: {y_data}")
                                    else:
                                        trace_info.append(f"  Y values (first 10): {y_data[:10]}")
                                        trace_info.append(f"  Y values (last 10): {y_data[-10:]}")
                                        trace_info.append(f"  Total Y points: {len(y_data)}")
                                    
                                    # Add statistics for Y values
                                    y_array = np.array(y_data)
                                    if np.issubdtype(y_array.dtype, np.number):
                                        trace_info.append(f"  Y Statistics:")
                                        trace_info.append(f"    Mean: {np.mean(y_array):.2f}")
                                        trace_info.append(f"    Median: {np.median(y_array):.2f}")
                                        trace_info.append(f"    Min: {np.min(y_array):.2f}")
                                        trace_info.append(f"    Max: {np.max(y_array):.2f}")
                                        trace_info.append(f"    Std Dev: {np.std(y_array):.2f}")
                                
                                chart_info_parts.append("\n".join(trace_info))
                        
                        captured_data.append(("plotly_chart", "\n".join(chart_info_parts)))
                    except Exception as e:
                        captured_data.append(("plotly_chart", f"Plotly chart generated (error extracting data: {str(e)})"))
                    return st.plotly_chart(fig, *args, **kwargs)
                
                # Create a custom streamlit object with wrapped functions
                class StreamlitWrapper:
                    def __init__(self, st_module):
                        self._st = st_module
                    
                    def __getattr__(self, name):
                        # Return wrapped versions of display functions
                        if name == 'dataframe':
                            return capture_dataframe
                        elif name == 'table':
                            return capture_table
                        elif name == 'write':
                            return capture_write
                        elif name == 'metric':
                            return capture_metric
                        elif name == 'plotly_chart':
                            return capture_plotly_chart
                        else:
                            # Pass through all other streamlit functions unchanged
                            return getattr(self._st, name)
                
                st_wrapped = StreamlitWrapper(st)
                
                try:
                    with result_display_area:
                        exec(code_to_execute, {
                            "st": st_wrapped, "pd": pd, "np": np, "px": px, "go": go, "plt": plt, "alt": alt,
                            "df": df, "site_performance_df": site_performance_df, "utm_performance_df": utm_performance_df,
                            "ts_col_map": ts_col_map, "ordered_stages": ordered_stages
                        })
                    
                    # Combine stdout and captured display data
                    execution_output = output_buffer.getvalue()
                    if execution_output:
                        st.text(execution_output)

                    # Add download buttons for any captured DataFrames
                    if captured_dataframes:
                        st.markdown("---")
                        st.markdown("### 📥 Download Options")

                        from datetime import datetime
                        import io

                        for idx, (display_type, df_data) in enumerate(captured_dataframes):
                            # Create a unique identifier for this table
                            table_num = idx + 1
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            # Create columns for the download section
                            col1, col2, col3 = st.columns([2, 1, 1])

                            with col1:
                                st.caption(f"**Table {table_num}** ({df_data.shape[0]} rows × {df_data.shape[1]} columns)")

                            with col2:
                                # CSV Download
                                try:
                                    csv_data = df_data.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="CSV",
                                        data=csv_data,
                                        file_name=f'ai_analysis_table{table_num}_{timestamp}.csv',
                                        mime='text/csv',
                                        key=f'download_csv_{idx}_{timestamp}',
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.warning(f"CSV export unavailable: {str(e)}")

                            with col3:
                                # Excel Download
                                try:
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        df_data.to_excel(writer, sheet_name=f'Table_{table_num}', index=False)
                                    excel_data = output.getvalue()

                                    st.download_button(
                                        label="Excel",
                                        data=excel_data,
                                        file_name=f'ai_analysis_table{table_num}_{timestamp}.xlsx',
                                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                        key=f'download_excel_{idx}_{timestamp}',
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.warning(f"Excel export unavailable: {str(e)}")

                        st.markdown("---")

                    # Build comprehensive output for AI summarization
                    comprehensive_output = []
                    if execution_output.strip():
                        comprehensive_output.append(f"Printed output:\n{execution_output}")
                    
                    if captured_data:
                        comprehensive_output.append("\nDisplayed data:")
                        for display_type, content in captured_data:
                            comprehensive_output.append(f"\n[{display_type.upper()}]:\n{content}\n")
                    
                    final_output = "\n".join(comprehensive_output) if comprehensive_output else "Code executed successfully with visual output."
                    
                    # Send the execution result back to the model for summarization
                    with st.spinner("Summarizing results..."):
                        summary_prompt = f"""Based on the ACTUAL DATA VALUES shown below, provide a specific, data-driven analysis. 

CRITICAL INSTRUCTIONS:
- Reference the ACTUAL NUMBERS from the data (e.g., "Week of 2024-01-15 had 45.2%")
- Identify the SPECIFIC highest and lowest values with their exact dates/categories
- Calculate and mention SPECIFIC trends (e.g., "increased from 35% to 48% over 8 weeks")
- DO NOT use placeholder language like "(Describe the general movement)" or "(Identify the week(s))"
- DO NOT provide generic templated analysis
- Every statement must reference specific data points from the output below

Data Output:
{final_output}

Provide a concise, specific analysis based on these exact values."""
                        summary_response = st.session_state.chat.send_message(summary_prompt)
                        st.markdown(summary_response.text)
                        st.session_state.messages.append({"role": "assistant", "content": summary_response.text})

                except Exception:
                    error_traceback = traceback.format_exc()
                    st.error("An error occurred during code execution:")
                    st.code(error_traceback, language="bash")
                    # Send error back to the model for potential self-correction
                    with st.spinner("Attempting to self-correct..."):
                        correction_prompt = f"The code you provided failed to execute with the following error. Please analyze the error and provide a corrected version of the code. \n\nError:\n{error_traceback}"
                        correction_response = st.session_state.chat.send_message(correction_prompt)
                        st.markdown("**AI Self-Correction Attempt:**\n" + correction_response.text)
                        st.session_state.messages.append({"role": "assistant", "content": "**AI Self-Correction Attempt:**\n" + correction_response.text})
                
                finally:
                    sys.stdout = sys.__stdout__ # Restore stdout
            
            else:
                # If no code, just display the text response
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})