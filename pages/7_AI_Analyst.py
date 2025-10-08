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
from helpers import format_performance_df

st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

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

# --- Configure the Gemini API ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    
    generation_config = genai.GenerationConfig(temperature=0.2)
    model = genai.GenerativeModel(
        'gemini-flash-latest',
        generation_config=generation_config
    )

except Exception as e:
    st.error("Error configuring the AI model. Have you set your GEMINI_API_KEY in Streamlit's secrets?")
    st.exception(e)
    st.stop()

# --- System Prompts for the Advanced Agent ---
@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

@st.cache_data
def get_coder_prompt(_df_info, _ts_col_map_str, _site_perf_info, _utm_perf_info):
    prompt_parts = [
        "You are an expert-level Python data analyst and a helpful AI assistant. Your goal is to be a full partner in data analysis for the user.",
        "You have been provided with three pandas DataFrames...",
        # The rest of the coder prompt remains the same
    ]
    return "\n".join(prompt_parts)

# --- THIS IS THE UPDATED PROMPT ---
@st.cache_data
def get_synthesizer_prompt():
    return """You are an expert business analyst and senior strategist.
Your goal is to provide a single, cohesive, and insightful executive summary based on the data provided.

You will be given the user's question, your own thought process, the code you executed, and the raw data result from that code.

**CRITICAL INSTRUCTION: You MUST extract the actual data (names, numbers, percentages) from the 'Raw Result' section and embed it directly into your summary. DO NOT use placeholders like '[Insert Data]' or '[Site Name]'. Your response must be ready for a final report.**

- Start with a bolded headline that directly answers the user's core question.
- Weave the specific results from the data into a clear and easy-to-understand narrative.
- Connect the data to business goals like speed, efficiency, or performance.
- Conclude with a clear recommendation or key takeaway based on the data.
"""

# --- Main Chat Logic ---
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    
    # Get the full system prompt only once
    site_perf_info = get_df_info(st.session_state.enhanced_site_metrics_df)
    utm_perf_info = get_df_info(st.session_state.enhanced_ad_source_metrics_df)
    raw_df_info = get_df_info(st.session_state.referral_data_processed)
    system_prompt = get_coder_prompt(raw_df_info, str(ts_col_map), site_perf_info, utm_perf_info)

    initial_response = st.session_state.chat.send_message(system_prompt)
    st.session_state.messages = [{"role": "assistant", "content": initial_response.text}]

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(user_prompt)
            
            code_match = re.search(r"```python\s*([\s\S]+?)```", response.text)
            
            if code_match:
                code_to_execute = code_match.group(1).strip()
                with st.expander("View AI's Generated Code", expanded=True):
                    st.code(code_to_execute, language="python")

                st.markdown("**Execution Result:**")
                result_display_area = st.container()
                output_buffer = StringIO()
                sys.stdout = output_buffer
                
                try:
                    with result_display_area:
                        exec(code_to_execute, {
                            "st": st, "pd": pd, "np": np, "px": px, "go": go, "plt": plt, "alt": alt,
                            "df": df, "site_performance_df": site_performance_df, "utm_performance_df": utm_performance_df,
                            "ts_col_map": ts_col_map, "ordered_stages": ordered_stages
                        })
                    execution_output = output_buffer.getvalue()
                    if execution_output:
                        st.text(execution_output)
                    
                    with st.spinner("Summarizing results..."):
                        # Get the synthesizer prompt with the new strict instructions
                        synthesizer_prompt = get_synthesizer_prompt()
                        
                        synthesis_context_parts = [
                            "**User's Question:**", user_prompt, "\n\n",
                            "**My Thought Process:**", st.session_state.chat.history[-2].parts[0].text, "\n\n", # Get the AI's previous thought
                            "**My Executed Code:**", f"```python\n{code_to_execute}\n```\n\n",
                            "**Raw Result:**", execution_output if execution_output else "A plot was generated successfully."
                        ]
                        synthesis_context = "".join(synthesis_context_parts)
                        
                        full_synthesizer_prompt = f"{synthesizer_prompt}\n\n--- ANALYSIS DETAILS ---\n{synthesis_context}\n\n--- EXECUTIVE SUMMARY ---"

                        summary_response = model.generate_content(full_synthesizer_prompt) # Use a one-off call for this
                        summary_text = summary_response.text
                        
                        st.markdown(summary_text)
                        st.session_state.messages.append({"role": "assistant", "content": summary_text})
                        # Also add the executed code and results to the visible history for context
                        st.session_state.chat.history.append(response.parts[0]) # The AI's code response
                        st.session_state.chat.history.append(genai.types.Content(role="user", parts=[genai.types.Part(text=summary_prompt)])) # The result we sent back

                except Exception:
                    error_traceback = traceback.format_exc()
                    st.error("An error occurred during code execution:")
                    st.code(error_traceback, language="bash")
                    with st.spinner("Attempting to self-correct..."):
                        correction_prompt = f"The code you provided failed to execute with the following error. Please analyze the error and provide a corrected version of the code. \n\nError:\n{error_traceback}"
                        correction_response = st.session_state.chat.send_message(correction_prompt)
                        st.markdown("**AI Self-Correction Attempt:**\n" + correction_response.text)
                        st.session_state.messages.append({"role": "assistant", "content": "**AI Self-Correction Attempt:**\n" + correction_response.text})
                
                finally:
                    sys.stdout = sys.__stdout__
            
            else:
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})