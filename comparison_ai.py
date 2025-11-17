# comparison_ai.py
"""
AI-powered insights generation for comparison analysis using Google Gemini.
Generates statistical analysis, key changes, and management summaries.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import google.generativeai as genai


def initialize_gemini_model(temperature: float = 0.2):
    """
    Initialize Gemini model for comparison insights.

    Args:
        temperature: Model temperature (0.0-1.0), lower = more focused/deterministic

    Returns:
        Generative model instance
    """
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=genai.GenerationConfig(temperature=temperature)
        )
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {str(e)}")
        return None


def format_comparison_context_for_ai(
    comparison_df: pd.DataFrame,
    label_a: str,
    label_b: str,
    category: str,
    key_column: str,
    significance_results: Optional[Dict] = None
) -> str:
    """
    Format comparison data into a context string for AI analysis.

    Args:
        comparison_df: Merged comparison dataframe with deltas
        label_a: Label for period A
        label_b: Label for period B
        category: Category name (Site Performance, etc.)
        key_column: Key column name (Site, UTM Source, etc.)
        significance_results: Optional statistical significance results

    Returns:
        Formatted context string
    """
    context_parts = []

    # Header
    context_parts.append(f"=== {category} Comparison Analysis ===")
    context_parts.append(f"Period A: {label_a}")
    context_parts.append(f"Period B: {label_b}")
    context_parts.append(f"Total entities analyzed: {len(comparison_df)}")
    context_parts.append("")

    # Overall statistics
    if 'Score_A' in comparison_df.columns and 'Score_B' in comparison_df.columns:
        avg_score_a = comparison_df['Score_A'].mean()
        avg_score_b = comparison_df['Score_B'].mean()
        avg_score_delta = avg_score_b - avg_score_a

        context_parts.append("=== Overall Performance Scores ===")
        context_parts.append(f"Average Score ({label_a}): {avg_score_a:.2f}")
        context_parts.append(f"Average Score ({label_b}): {avg_score_b:.2f}")
        context_parts.append(f"Average Change: {avg_score_delta:+.2f} ({(avg_score_delta/avg_score_a*100):+.1f}%)")
        context_parts.append("")

        # Grade distribution
        if 'Grade_A' in comparison_df.columns and 'Grade_B' in comparison_df.columns:
            # Convert to string to handle mixed types, then sort
            grade_dist_a = comparison_df['Grade_A'].astype(str).value_counts().sort_index()
            grade_dist_b = comparison_df['Grade_B'].astype(str).value_counts().sort_index()

            context_parts.append("=== Grade Distribution ===")
            context_parts.append(f"{label_a}: {dict(grade_dist_a)}")
            context_parts.append(f"{label_b}: {dict(grade_dist_b)}")
            context_parts.append("")

    # Top improvers and decliners
    if 'Score_Delta' in comparison_df.columns:
        context_parts.append("=== Top 5 Improvers (by Score) ===")
        top_improvers = comparison_df.nlargest(5, 'Score_Delta')[
            [key_column, 'Score_A', 'Score_B', 'Score_Delta', 'Grade_A', 'Grade_B']
        ]
        context_parts.append(top_improvers.to_string(index=False))
        context_parts.append("")

        context_parts.append("=== Top 5 Decliners (by Score) ===")
        top_decliners = comparison_df.nsmallest(5, 'Score_Delta')[
            [key_column, 'Score_A', 'Score_B', 'Score_Delta', 'Grade_A', 'Grade_B']
        ]
        context_parts.append(top_decliners.to_string(index=False))
        context_parts.append("")

    # Key metric changes (averages across all entities)
    context_parts.append("=== Average Changes in Key Metrics ===")

    # Find all delta columns
    delta_cols = [col for col in comparison_df.columns if col.endswith('_Delta') and not col.endswith('_Delta_Pct')]

    for delta_col in delta_cols[:10]:  # Limit to top 10 metrics
        metric_name = delta_col.replace('_Delta', '')
        if f'{metric_name}_A' in comparison_df.columns and f'{metric_name}_B' in comparison_df.columns:
            avg_a = comparison_df[f'{metric_name}_A'].mean()
            avg_b = comparison_df[f'{metric_name}_B'].mean()
            avg_delta = comparison_df[delta_col].mean()
            avg_delta_pct = (avg_delta / avg_a * 100) if avg_a != 0 else 0

            context_parts.append(
                f"{metric_name}: {avg_a:.2f} → {avg_b:.2f} (Δ {avg_delta:+.2f}, {avg_delta_pct:+.1f}%)"
            )

    context_parts.append("")

    # Statistical significance (if provided)
    if significance_results:
        context_parts.append("=== Statistical Significance ===")
        for metric, result in significance_results.items():
            if result.get('is_significant'):
                context_parts.append(
                    f"{metric}: p={result['p_value']:.4f} {result['significance_level']} "
                    f"(test: {result['test_used']})"
                )
        context_parts.append("")

    return "\n".join(context_parts)


def generate_site_performance_insights(
    comparison_df: pd.DataFrame,
    label_a: str,
    label_b: str,
    significance_results: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate AI insights for site performance comparison.

    Returns dict with 'key_changes' (list) and 'management_summary' (str)
    """
    model = initialize_gemini_model()

    if model is None:
        return {
            'key_changes': ["AI insights unavailable - Gemini API not configured"],
            'management_summary': "Unable to generate AI insights at this time."
        }

    # Format context
    context = format_comparison_context_for_ai(
        comparison_df, label_a, label_b,
        "Site Performance", "Site", significance_results
    )

    # Create prompt
    prompt = f"""
You are analyzing site performance data for a clinical trial recruitment platform. Your audience is management who needs clear, actionable insights.

IMPORTANT TERMINOLOGY:
- "StS" means "Sent to Site" (the stage when a qualified referral is sent to a clinical trial site)
- "ICF" means "Informed Consent Form"
- "Appt" means "Appointment"

{context}

Based on this comparison, provide:

1. **Key Changes** (3-5 bullet points):
   - Focus on the most statistically significant and practically important changes
   - Be specific with numbers and percentages
   - Highlight both improvements and areas of concern
   - Note statistical significance where relevant (p < 0.05)
   - Make insights actionable

2. **Management Summary** (3-4 sentences):
   - Overall trend: Are sites generally improving, declining, or mixed?
   - Most critical areas requiring immediate attention or celebration
   - Recommended next actions based on the data
   - Written for non-technical executive audience

Format your response EXACTLY as:
KEY_CHANGES:
• [First bullet point]
• [Second bullet point]
• [Third bullet point]
• [etc.]

MANAGEMENT_SUMMARY:
[Your 3-4 sentence paragraph here]
"""

    try:
        response = model.generate_content(prompt)
        return parse_ai_response(response.text)
    except Exception as e:
        return {
            'key_changes': [f"Error generating insights: {str(e)}"],
            'management_summary': "Unable to generate AI insights due to an error."
        }


def generate_ad_performance_insights(
    comparison_df: pd.DataFrame,
    label_a: str,
    label_b: str,
    table_type: str = 'source',
    significance_results: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate AI insights for ad performance comparison.
    """
    model = initialize_gemini_model()

    if model is None:
        return {
            'key_changes': ["AI insights unavailable - Gemini API not configured"],
            'management_summary': "Unable to generate AI insights at this time."
        }

    key_col = 'UTM Source' if table_type == 'source' else 'UTM Source/Medium'

    context = format_comparison_context_for_ai(
        comparison_df, label_a, label_b,
        f"Ad Performance ({table_type})", key_col, significance_results
    )

    prompt = f"""
You are analyzing advertising channel performance data for a clinical trial recruitment platform. Your audience is marketing management making budget allocation decisions.

IMPORTANT TERMINOLOGY:
- "StS" means "Sent to Site" (the stage when a qualified referral is sent to a clinical trial site)
- "ICF" means "Informed Consent Form"
- "Qualified" means a referral that met initial screening criteria

{context}

Based on this comparison, provide:

1. **Key Changes** (3-5 bullet points):
   - Which UTM sources/mediums showed significant ROI improvement or decline
   - Conversion rate changes (statistical significance)
   - Screen fail rate trends and implications
   - Budget reallocation recommendations
   - Be specific with numbers

2. **Management Summary** (3-4 sentences):
   - Overall marketing performance trend
   - Recommended budget shifts based on data
   - Critical channels requiring attention
   - Next actions for marketing team

Format your response EXACTLY as:
KEY_CHANGES:
• [First bullet point]
• [Second bullet point]
• [Third bullet point]
• [etc.]

MANAGEMENT_SUMMARY:
[Your 3-4 sentence paragraph here]
"""

    try:
        response = model.generate_content(prompt)
        return parse_ai_response(response.text)
    except Exception as e:
        return {
            'key_changes': [f"Error generating insights: {str(e)}"],
            'management_summary': "Unable to generate AI insights due to an error."
        }


def generate_pc_performance_insights(
    period_a_data: Dict,
    period_b_data: Dict,
    label_a: str,
    label_b: str,
    comparison_type: str
) -> Dict[str, Any]:
    """
    Generate AI insights for PC performance comparison.
    """
    model = initialize_gemini_model()

    if model is None:
        return {
            'key_changes': ["AI insights unavailable - Gemini API not configured"],
            'management_summary': "Unable to generate AI insights at this time."
        }

    # Format context based on comparison type
    if comparison_type == 'time_metrics':
        context = f"""
=== PC Performance Time Metrics Comparison ===
Period A ({label_a}):
{pd.Series(period_a_data).to_string()}

Period B ({label_b}):
{pd.Series(period_b_data).to_string()}

Changes:
"""
        for key in period_a_data.keys():
            if key in period_b_data:
                delta = period_b_data[key] - period_a_data[key]
                pct = (delta / period_a_data[key] * 100) if period_a_data[key] != 0 else 0
                context += f"{key}: {delta:+.2f} days ({pct:+.1f}%)\n"

    elif comparison_type == 'contact_effectiveness':
        # Format contact effectiveness data - period_a_data and period_b_data are DataFrames
        # Calculate aggregate metrics from the DataFrames

        # Sum totals across all attempt levels
        totals_a = {
            'Total Referrals': period_a_data['Total Referrals'].sum() if 'Total Referrals' in period_a_data.columns else 0,
            'Total StS': period_a_data['Total_StS'].sum() if 'Total_StS' in period_a_data.columns else 0,
            'Total ICF': period_a_data['Total_ICF'].sum() if 'Total_ICF' in period_a_data.columns else 0,
            'Total Enrolled': period_a_data['Total_Enrolled'].sum() if 'Total_Enrolled' in period_a_data.columns else 0
        }

        totals_b = {
            'Total Referrals': period_b_data['Total Referrals'].sum() if 'Total Referrals' in period_b_data.columns else 0,
            'Total StS': period_b_data['Total_StS'].sum() if 'Total_StS' in period_b_data.columns else 0,
            'Total ICF': period_b_data['Total_ICF'].sum() if 'Total_ICF' in period_b_data.columns else 0,
            'Total Enrolled': period_b_data['Total_Enrolled'].sum() if 'Total_Enrolled' in period_b_data.columns else 0
        }

        # Calculate overall rates
        rates_a = {
            'Overall StS Rate': totals_a['Total StS'] / totals_a['Total Referrals'] if totals_a['Total Referrals'] > 0 else 0,
            'Overall ICF Rate': totals_a['Total ICF'] / totals_a['Total StS'] if totals_a['Total StS'] > 0 else 0,
            'Overall Enrollment Rate': totals_a['Total Enrolled'] / totals_a['Total ICF'] if totals_a['Total ICF'] > 0 else 0
        }

        rates_b = {
            'Overall StS Rate': totals_b['Total StS'] / totals_b['Total Referrals'] if totals_b['Total Referrals'] > 0 else 0,
            'Overall ICF Rate': totals_b['Total ICF'] / totals_b['Total StS'] if totals_b['Total StS'] > 0 else 0,
            'Overall Enrollment Rate': totals_b['Total Enrolled'] / totals_b['Total ICF'] if totals_b['Total ICF'] > 0 else 0
        }

        context = f"""
=== PC Performance Contact Effectiveness Comparison ===
Analyzing how contact attempt frequency impacts conversion rates.

Period A ({label_a}) - Overall Metrics:
Total Referrals: {totals_a['Total Referrals']:,.0f}
Total StS: {totals_a['Total StS']:,.0f}
Total ICF: {totals_a['Total ICF']:,.0f}
Total Enrolled: {totals_a['Total Enrolled']:,.0f}
Overall StS Rate: {rates_a['Overall StS Rate']:.1%}
Overall ICF Rate: {rates_a['Overall ICF Rate']:.1%}
Overall Enrollment Rate: {rates_a['Overall Enrollment Rate']:.1%}

Period B ({label_b}) - Overall Metrics:
Total Referrals: {totals_b['Total Referrals']:,.0f}
Total StS: {totals_b['Total StS']:,.0f}
Total ICF: {totals_b['Total ICF']:,.0f}
Total Enrolled: {totals_b['Total Enrolled']:,.0f}
Overall StS Rate: {rates_b['Overall StS Rate']:.1%}
Overall ICF Rate: {rates_b['Overall ICF Rate']:.1%}
Overall Enrollment Rate: {rates_b['Overall Enrollment Rate']:.1%}

Key Changes:
"""
        # Calculate and format changes
        for key in totals_a.keys():
            delta = totals_b[key] - totals_a[key]
            if totals_a[key] != 0:
                pct = (delta / totals_a[key] * 100)
                context += f"{key}: {totals_a[key]:,.0f} → {totals_b[key]:,.0f} ({delta:+,.0f}, {pct:+.1f}%)\n"

        for key in rates_a.keys():
            delta = rates_b[key] - rates_a[key]
            if rates_a[key] != 0:
                pct = (delta / rates_a[key] * 100)
                context += f"{key}: {rates_a[key]:.1%} → {rates_b[key]:.1%} ({pct:+.1f}% change)\n"

    else:
        context = f"PC Performance {comparison_type} comparison between {label_a} and {label_b}"

    prompt = f"""
You are analyzing Pre-screening Coordinator (PC) team efficiency data for a clinical trial recruitment platform. Your audience is operations management.

{context}

Based on this comparison, provide:

1. **Key Changes** (3-5 bullet points):
   - Contact timing improvements or declines
   - Efficiency metrics changes
   - Staffing implications
   - Best practices to scale successful behaviors
   - Be specific with time metrics

2. **Management Summary** (3-4 sentences):
   - Overall PC team efficiency trend
   - Operational recommendations
   - Critical areas for training or process improvement
   - Next actions for operations team

Format your response EXACTLY as:
KEY_CHANGES:
• [First bullet point]
• [Second bullet point]
• [Third bullet point]
• [etc.]

MANAGEMENT_SUMMARY:
[Your 3-4 sentence paragraph here]
"""

    try:
        response = model.generate_content(prompt)
        return parse_ai_response(response.text)
    except Exception as e:
        return {
            'key_changes': [f"Error generating insights: {str(e)}"],
            'management_summary': "Unable to generate AI insights due to an error."
        }


def generate_site_outreach_insights(
    operational_kpis_a: Dict,
    operational_kpis_b: Dict,
    label_a: str,
    label_b: str
) -> Dict[str, Any]:
    """
    Generate AI insights for site outreach effectiveness comparison.

    Args:
        operational_kpis_a: Dict with operational KPIs for period A
        operational_kpis_b: Dict with operational KPIs for period B
        label_a: Label for period A
        label_b: Label for period B

    Returns:
        Dict with 'key_changes' (list) and 'management_summary' (str)
    """
    model = initialize_gemini_model()

    if model is None:
        return {
            'key_changes': ["AI insights unavailable - Gemini API not configured"],
            'management_summary': "Unable to generate AI insights at this time."
        }

    # Extract KPIs
    time_to_first_a = operational_kpis_a.get('avg_time_to_first_action', 0) or 0
    time_to_first_b = operational_kpis_b.get('avg_time_to_first_action', 0) or 0

    time_between_a = operational_kpis_a.get('avg_time_between_contacts', 0) or 0
    time_between_b = operational_kpis_b.get('avg_time_between_contacts', 0) or 0

    time_sts_appt_a = operational_kpis_a.get('avg_time_sts_to_appt', 0) or 0
    time_sts_appt_b = operational_kpis_b.get('avg_time_sts_to_appt', 0) or 0

    stale_refs_a = operational_kpis_a.get('referrals_awaiting_action_7d', 0) or 0
    stale_refs_b = operational_kpis_b.get('referrals_awaiting_action_7d', 0) or 0

    # Calculate deltas
    delta_time_to_first = time_to_first_b - time_to_first_a
    delta_time_between = time_between_b - time_between_a
    delta_time_sts_appt = time_sts_appt_b - time_sts_appt_a
    delta_stale_refs = stale_refs_b - stale_refs_a

    # Calculate percentage changes (avoid division by zero)
    pct_time_to_first = (delta_time_to_first / time_to_first_a * 100) if time_to_first_a != 0 else 0
    pct_time_between = (delta_time_between / time_between_a * 100) if time_between_a != 0 else 0
    pct_time_sts_appt = (delta_time_sts_appt / time_sts_appt_a * 100) if time_sts_appt_a != 0 else 0
    pct_stale_refs = (delta_stale_refs / stale_refs_a * 100) if stale_refs_a != 0 else 0

    context = f"""
=== Site Outreach Effectiveness Comparison ===
Analyzing how quickly sites respond to and process referrals.

Period A ({label_a}):
- Avg. Time to First Site Action: {time_to_first_a:.1f} days
- Avg. Time Between Site Contacts: {time_between_a:.1f} days
- Avg. Time StS to Appt. Sched.: {time_sts_appt_a:.1f} days
- Referrals Awaiting Action > 7 Days: {stale_refs_a:.0f} referrals

Period B ({label_b}):
- Avg. Time to First Site Action: {time_to_first_b:.1f} days
- Avg. Time Between Site Contacts: {time_between_b:.1f} days
- Avg. Time StS to Appt. Sched.: {time_sts_appt_b:.1f} days
- Referrals Awaiting Action > 7 Days: {stale_refs_b:.0f} referrals

Changes:
- Time to First Site Action: {delta_time_to_first:+.1f} days ({pct_time_to_first:+.1f}%)
- Time Between Site Contacts: {delta_time_between:+.1f} days ({pct_time_between:+.1f}%)
- Time StS to Appt. Sched.: {delta_time_sts_appt:+.1f} days ({pct_time_sts_appt:+.1f}%)
- Referrals Awaiting Action > 7 Days: {delta_stale_refs:+.0f} referrals ({pct_stale_refs:+.1f}%)
"""

    prompt = f"""
You are analyzing site operational efficiency data for a clinical trial recruitment platform. Your audience is site management and operations teams.

IMPORTANT TERMINOLOGY:
- "StS" means "Sent to Site" (the stage when a qualified referral is sent to a clinical trial site)
- "First Site Action" means the first time a site contacts or processes a referral after receiving it
- "Site Contacts" are follow-up attempts by sites to reach referrals
- "Appt. Sched." means "Appointment Scheduled"
- Lower times are better (faster site response = better patient experience)
- Lower "Referrals Awaiting Action > 7 Days" is better (fewer stale referrals)

{context}

Based on this comparison, provide:

1. **Key Changes** (3-5 bullet points):
   - Site responsiveness improvements or declines
   - Impact on patient experience (faster/slower engagement)
   - Workload indicators (stale referral trends)
   - Training or process recommendations
   - Be specific with time metrics and direction of change

2. **Management Summary** (3-4 sentences):
   - Overall site efficiency trend
   - Critical areas requiring attention or celebration
   - Recommended next actions for site coordination teams
   - Impact on patient conversion potential

Format your response EXACTLY as:
KEY_CHANGES:
• [First bullet point]
• [Second bullet point]
• [Third bullet point]
• [etc.]

MANAGEMENT_SUMMARY:
[Your 3-4 sentence paragraph here]
"""

    try:
        response = model.generate_content(prompt)
        return parse_ai_response(response.text)
    except Exception as e:
        return {
            'key_changes': [f"Error generating insights: {str(e)}"],
            'management_summary': "Unable to generate AI insights due to an error."
        }


def generate_funnel_insights(
    results_a: Dict,
    results_b: Dict,
    label_a: str,
    label_b: str
) -> Dict[str, Any]:
    """
    Generate AI insights for funnel analysis comparison.
    """
    model = initialize_gemini_model()

    if model is None:
        return {
            'key_changes': ["AI insights unavailable - Gemini API not configured"],
            'management_summary': "Unable to generate AI insights at this time."
        }

    # Extract key metrics
    icf_a = results_a['total_icf_yield']
    icf_b = results_b['total_icf_yield']
    enroll_a = results_a['total_enroll_yield']
    enroll_b = results_b['total_enroll_yield']

    icf_delta = icf_b - icf_a
    enroll_delta = enroll_b - enroll_a

    # Calculate percentage changes safely (avoid division by zero)
    icf_pct_change = (icf_delta / icf_a * 100) if icf_a != 0 else 0
    enroll_pct_change = (enroll_delta / enroll_a * 100) if enroll_a != 0 else 0

    context = f"""
=== Funnel Analysis Comparison ===
Period A ({label_a}):
- Expected ICF Yield: {icf_a:.1f}
- Expected Enrollment Yield: {enroll_a:.1f}

Period B ({label_b}):
- Expected ICF Yield: {icf_b:.1f}
- Expected Enrollment Yield: {enroll_b:.1f}

Changes:
- ICF Yield: {icf_delta:+.1f} ({icf_pct_change:+.1f}%)
- Enrollment Yield: {enroll_delta:+.1f} ({enroll_pct_change:+.1f}%)
"""

    prompt = f"""
You are analyzing clinical trial recruitment pipeline health data. Your audience is executive management tracking enrollment projections.

{context}

Based on this comparison, provide:

1. **Key Changes** (3-5 bullet points):
   - Pipeline health improvements or declines
   - Projected yield changes and implications
   - Stage-specific bottlenecks that improved/worsened
   - Risk areas requiring intervention
   - Be specific with projected numbers

2. **Management Summary** (3-4 sentences):
   - Overall pipeline health trend
   - Projected impact on enrollment goals
   - Strategic recommendations for pipeline optimization
   - Next actions for recruitment team

Format your response EXACTLY as:
KEY_CHANGES:
• [First bullet point]
• [Second bullet point]
• [Third bullet point]
• [etc.]

MANAGEMENT_SUMMARY:
[Your 3-4 sentence paragraph here]
"""

    try:
        response = model.generate_content(prompt)
        return parse_ai_response(response.text)
    except Exception as e:
        return {
            'key_changes': [f"Error generating insights: {str(e)}"],
            'management_summary': "Unable to generate AI insights due to an error."
        }


def parse_ai_response(response_text: str) -> Dict[str, Any]:
    """
    Parse AI response into structured format.

    Args:
        response_text: Raw text response from Gemini

    Returns:
        Dict with 'key_changes' (list) and 'management_summary' (str)
    """
    try:
        # Split by sections
        if 'MANAGEMENT_SUMMARY:' in response_text:
            parts = response_text.split('MANAGEMENT_SUMMARY:')
            key_changes_section = parts[0].replace('KEY_CHANGES:', '').strip()
            management_summary = parts[1].strip()
        else:
            # Fallback if format is not exact
            key_changes_section = response_text
            management_summary = "Summary not available in expected format."

        # Parse bullet points
        key_changes = []
        for line in key_changes_section.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                # Remove bullet point markers and clean markdown
                cleaned_line = line.lstrip('•-* ').strip()
                # Remove markdown bold formatting (** or __)
                cleaned_line = cleaned_line.replace('**', '').replace('__', '')
                if cleaned_line:
                    key_changes.append(cleaned_line)

        # Clean markdown from management summary as well
        management_summary = management_summary.replace('**', '').replace('__', '')

        if not key_changes:
            key_changes = ["Unable to parse key changes from AI response"]

        return {
            'key_changes': key_changes,
            'management_summary': management_summary
        }

    except Exception as e:
        return {
            'key_changes': [f"Error parsing AI response: {str(e)}"],
            'management_summary': response_text[:500] if response_text else "No response received."
        }


def display_ai_insights(insights: Dict[str, Any]) -> None:
    """
    Display AI insights in Streamlit with proper formatting.

    Args:
        insights: Dict with 'key_changes' and 'management_summary'
    """
    st.markdown("### 🤖 AI-Powered Insights")

    with st.container(border=True):
        # Key Changes
        st.markdown("#### 📊 Key Changes")
        for change in insights['key_changes']:
            st.markdown(f"• {change}")

        st.divider()

        # Management Summary
        st.markdown("#### 📝 Management Summary")
        st.markdown(insights['management_summary'])
