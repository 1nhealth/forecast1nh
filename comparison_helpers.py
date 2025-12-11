# comparison_helpers.py
"""
Helper functions for formatting and displaying comparison data.
Includes delta formatting, table styling, and chart generation.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple


def format_delta_indicator(value: float,
                           is_positive_good: bool = True,
                           format_type: str = 'number',
                           show_sign: bool = True) -> str:
    """
    Format delta with color-coded indicator using Streamlit markdown.

    Args:
        value: The delta value
        is_positive_good: Whether positive change is improvement
        format_type: 'number', 'percentage', 'decimal', or 'days'
        show_sign: Whether to show +/- sign

    Returns:
        Formatted string with markdown color styling
    """
    if pd.isna(value) or value == 0:
        return '—'

    # Determine if this is an improvement
    is_improvement = (value > 0) if is_positive_good else (value < 0)

    # Color (using brand colors)
    color = '#53CA97' if is_improvement else '#E76F51'  # Brand green vs red

    # Format the value
    sign = '+' if value > 0 and show_sign else ''

    if format_type == 'percentage':
        formatted = f"{sign}{value:.1f}%"
    elif format_type == 'number':
        formatted = f"{sign}{value:,.0f}"
    elif format_type == 'decimal':
        formatted = f"{sign}{value:.2f}"
    elif format_type == 'days':
        formatted = f"{sign}{value:.1f}d"
    else:
        formatted = f"{sign}{value}"

    # Return markdown-styled string
    return f":{color}[**{formatted}**]"


def format_delta_html(value: float,
                     is_positive_good: bool = True,
                     format_type: str = 'number') -> str:
    """
    Format delta with HTML/CSS for use in styled dataframes.

    Args:
        value: The delta value
        is_positive_good: Whether positive change is improvement
        format_type: 'number', 'percentage', 'decimal', or 'days'

    Returns:
        HTML string with CSS class styling
    """
    if pd.isna(value) or value == 0:
        return '<span class="delta-neutral">—</span>'

    is_improvement = (value > 0) if is_positive_good else (value < 0)
    css_class = 'delta-positive' if is_improvement else 'delta-negative'

    sign = '+' if value > 0 else ''

    if format_type == 'percentage':
        formatted = f"{sign}{value:.1f}%"
    elif format_type == 'number':
        formatted = f"{sign}{value:,.0f}"
    elif format_type == 'decimal':
        formatted = f"{sign}{value:.2f}"
    elif format_type == 'days':
        formatted = f"{sign}{value:.1f}d"
    else:
        formatted = f"{sign}{value}"

    return f'<span class="{css_class}">{formatted}</span>'


def create_comparison_display_df(comparison_df: pd.DataFrame,
                                 label_a: str,
                                 label_b: str,
                                 key_column: str,
                                 metrics_to_display: List[str],
                                 metric_configs: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a display-ready dataframe for comparison with formatted deltas.

    Args:
        comparison_df: Merged comparison dataframe with _A, _B, _Delta columns
        label_a: Label for period A
        label_b: Label for period B
        key_column: The key column (e.g., 'Site', 'UTM Source')
        metrics_to_display: List of metric names to include
        metric_configs: Configuration for each metric
            {
                'Score': {
                    'format': 'decimal',
                    'inverse': False,
                    'show_delta': True,
                    'show_pct_delta': False
                },
                ...
            }

    Returns:
        Display-ready DataFrame
    """
    display_df = pd.DataFrame()
    display_df[key_column] = comparison_df[key_column]

    for metric in metrics_to_display:
        col_a = f"{metric}_A"
        col_b = f"{metric}_B"
        col_delta = f"{metric}_Delta"
        col_delta_pct = f"{metric}_Delta_Pct"

        config = metric_configs.get(metric, {})
        format_type = config.get('format', 'number')
        is_inverse = config.get('inverse', False)
        show_delta = config.get('show_delta', True)
        show_pct_delta = config.get('show_pct_delta', True)

        # Add period columns
        if col_a in comparison_df.columns:
            display_df[f"{metric} ({label_a})"] = comparison_df[col_a]
        if col_b in comparison_df.columns:
            display_df[f"{metric} ({label_b})"] = comparison_df[col_b]

        # Add delta column
        if show_delta and col_delta in comparison_df.columns:
            if show_pct_delta and col_delta_pct in comparison_df.columns:
                # Show percentage change
                display_df[f"{metric} Δ%"] = comparison_df.apply(
                    lambda row: format_delta_indicator(
                        row[col_delta_pct],
                        not is_inverse,
                        'percentage'
                    ),
                    axis=1
                )
            else:
                # Show absolute change
                display_df[f"{metric} Δ"] = comparison_df.apply(
                    lambda row: format_delta_indicator(
                        row[col_delta],
                        not is_inverse,
                        format_type
                    ),
                    axis=1
                )

    # Add rank change if exists
    if 'Rank_Direction' in comparison_df.columns and 'Rank_Change' in comparison_df.columns:
        display_df['Rank Change'] = comparison_df.apply(
            lambda row: f"{row['Rank_Direction']} {abs(row['Rank_Change'])}"
            if row['Rank_Change'] != 0 else '→',
            axis=1
        )

    return display_df


def create_metric_card_comparison(metric_name: str,
                                 value_a: float,
                                 value_b: float,
                                 label_a: str,
                                 label_b: str,
                                 format_type: str = 'number',
                                 is_inverse: bool = False,
                                 help_text: Optional[str] = None) -> None:
    """
    Display a metric comparison as Streamlit metric cards in columns.

    Args:
        metric_name: Name of the metric
        value_a: Value for period A
        value_b: Value for period B
        label_a: Label for period A
        label_b: Label for period B
        format_type: How to format values
        is_inverse: Whether lower is better
        help_text: Optional help text
    """
    # Format values
    if format_type == 'percentage':
        formatted_a = f"{value_a:.1%}"
        formatted_b = f"{value_b:.1%}"
        delta = (value_b - value_a) * 100
        delta_str = f"{delta:+.1f}%"
    elif format_type == 'days':
        from helpers import format_days_to_dhm
        formatted_a = format_days_to_dhm(value_a)
        formatted_b = format_days_to_dhm(value_b)
        delta = value_b - value_a
        delta_str = f"{delta:+.1f}d"
    elif format_type == 'decimal':
        formatted_a = f"{value_a:.2f}"
        formatted_b = f"{value_b:.2f}"
        delta = value_b - value_a
        delta_str = f"{delta:+.2f}"
    else:  # number
        formatted_a = f"{value_a:,.0f}"
        formatted_b = f"{value_b:,.0f}"
        delta = value_b - value_a
        delta_str = f"{delta:+,.0f}"

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.metric(
                label=f"{metric_name} - {label_a}",
                value=formatted_a,
                help=help_text
            )

    with col2:
        with st.container(border=True):
            # Determine delta direction
            delta_value = value_b - value_a
            if is_inverse:
                delta_color = "inverse"
            else:
                delta_color = "normal"

            st.metric(
                label=f"{metric_name} - {label_b}",
                value=formatted_b,
                delta=delta_str,
                delta_color=delta_color,
                help=help_text
            )


def create_side_by_side_bar_chart(data_a: pd.DataFrame,
                                  data_b: pd.DataFrame,
                                  label_a: str,
                                  label_b: str,
                                  x_column: str,
                                  y_column: str,
                                  title: str,
                                  y_axis_title: Optional[str] = None) -> go.Figure:
    """
    Create side-by-side bar chart for comparison.

    Args:
        data_a: Period A data
        data_b: Period B data
        label_a: Label for period A
        label_b: Label for period B
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Chart title
        y_axis_title: Y-axis title (optional)

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Period A bars
    fig.add_trace(go.Bar(
        name=label_a,
        x=data_a[x_column],
        y=data_a[y_column],
        marker_color='#7991C6',  # Brand blue
        text=data_a[y_column],
        textposition='auto',
    ))

    # Period B bars
    fig.add_trace(go.Bar(
        name=label_b,
        x=data_b[x_column],
        y=data_b[y_column],
        marker_color='#53CA97',  # Brand green
        text=data_b[y_column],
        textposition='auto',
    ))

    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title=x_column,
        yaxis_title=y_axis_title or y_column,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    return fig


def create_side_by_side_heatmap(data_a: pd.DataFrame,
                                data_b: pd.DataFrame,
                                label_a: str,
                                label_b: str,
                                title: str,
                                colorscale: str = 'RdYlGn') -> go.Figure:
    """
    Create side-by-side heatmaps for comparison.

    Args:
        data_a: Period A heatmap data (rows=hours, columns=days)
        data_b: Period B heatmap data
        label_a: Label for period A
        label_b: Label for period B
        title: Main title
        colorscale: Plotly colorscale

    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[label_a, label_b],
        horizontal_spacing=0.1
    )

    # Determine common color scale range
    vmin = min(data_a.min().min(), data_b.min().min())
    vmax = max(data_a.max().max(), data_b.max().max())

    # Heatmap A
    fig.add_trace(
        go.Heatmap(
            z=data_a.values,
            x=data_a.columns,
            y=data_a.index,
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            colorbar=dict(x=0.45)
        ),
        row=1, col=1
    )

    # Heatmap B
    fig.add_trace(
        go.Heatmap(
            z=data_b.values,
            x=data_b.columns,
            y=data_b.index,
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            colorbar=dict(x=1.05)
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=title,
        height=600
    )

    fig.update_yaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_yaxes(title_text="Hour of Day", row=1, col=2)
    fig.update_xaxes(title_text="Day of Week", row=1, col=1)
    fig.update_xaxes(title_text="Day of Week", row=1, col=2)

    return fig


def create_comparison_line_chart(data_a: pd.DataFrame,
                                 data_b: pd.DataFrame,
                                 label_a: str,
                                 label_b: str,
                                 x_column: str,
                                 y_column: str,
                                 title: str,
                                 y_axis_title: Optional[str] = None) -> go.Figure:
    """
    Create line chart comparing trends over time.

    Args:
        data_a: Period A time series data
        data_b: Period B time series data
        label_a: Label for period A
        label_b: Label for period B
        x_column: Column name for x-axis (time)
        y_column: Column name for y-axis
        title: Chart title
        y_axis_title: Y-axis title (optional)

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Period A line
    fig.add_trace(go.Scatter(
        x=data_a[x_column],
        y=data_a[y_column],
        name=label_a,
        mode='lines+markers',
        line=dict(color='#7991C6', width=2),
        marker=dict(size=8)
    ))

    # Period B line
    fig.add_trace(go.Scatter(
        x=data_b[x_column],
        y=data_b[y_column],
        name=label_b,
        mode='lines+markers',
        line=dict(color='#53CA97', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_axis_title or y_column,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode='x unified'
    )

    return fig


def create_comparison_pie_charts(data_a: pd.Series,
                                 data_b: pd.Series,
                                 label_a: str,
                                 label_b: str,
                                 title: str) -> go.Figure:
    """
    Create side-by-side pie charts for categorical comparison.

    Args:
        data_a: Period A categorical data (Series with index as categories)
        data_b: Period B categorical data
        label_a: Label for period A
        label_b: Label for period B
        title: Main title

    Returns:
        Plotly figure with subplots
    """
    brand_colors = ['#53CA97', '#7991C6', '#F4A261', '#E76F51', '#2A9D8F', '#E9C46A', '#A2D2FF', '#FFB703']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{label_a} (Total: {data_a.sum():,.0f})",
                       f"{label_b} (Total: {data_b.sum():,.0f})"],
        specs=[[{'type':'domain'}, {'type':'domain'}]],
        horizontal_spacing=0.1
    )

    # Pie A
    fig.add_trace(
        go.Pie(
            labels=data_a.index,
            values=data_a.values,
            marker=dict(colors=brand_colors),
            textposition='inside',
            textinfo='percent+label'
        ),
        row=1, col=1
    )

    # Pie B
    fig.add_trace(
        go.Pie(
            labels=data_b.index,
            values=data_b.values,
            marker=dict(colors=brand_colors),
            textposition='inside',
            textinfo='percent+label'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=title,
        height=500,
        showlegend=True
    )

    return fig


def display_validation_messages(validation: Dict) -> None:
    """
    Display validation errors and warnings from comparison calculation.

    Args:
        validation: Validation dict with 'errors', 'warnings', and data counts
    """
    if validation.get('errors'):
        for error in validation['errors']:
            st.error(error)

    if validation.get('warnings'):
        for warning in validation['warnings']:
            st.warning(warning)

    # Display data summary
    if 'period_a_count' in validation and 'period_b_count' in validation:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Period A: {validation['period_a_count']} referrals over {validation.get('period_a_days', '?')} days")
        with col2:
            st.info(f"Period B: {validation['period_b_count']} referrals over {validation.get('period_b_days', '?')} days")


def highlight_improvements(df: pd.DataFrame, column: str, is_inverse: bool = False) -> pd.DataFrame:
    """
    Apply background color highlighting to a dataframe column based on improvement.

    Args:
        df: DataFrame to style
        column: Column name to highlight
        is_inverse: Whether lower values are better

    Returns:
        Styled DataFrame
    """
    def color_delta(val):
        """Color code based on value sign and inverse flag"""
        if pd.isna(val) or val == 0:
            return 'background-color: #f5f6f8'

        is_improvement = (val > 0) if not is_inverse else (val < 0)

        if is_improvement:
            return 'background-color: #E8F5E9; color: #2E7D32'  # Light green background, dark green text
        else:
            return 'background-color: #FFEBEE; color: #C62828'  # Light red background, dark red text

    return df.style.applymap(color_delta, subset=[column])


def create_summary_stats_table(comparison_df: pd.DataFrame,
                               metrics: List[str],
                               label_a: str,
                               label_b: str,
                               period_a_df: Optional[pd.DataFrame] = None,
                               period_b_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create a summary statistics table showing average changes across all entities.

    Args:
        comparison_df: Merged comparison dataframe
        metrics: List of metric names to summarize
        label_a: Label for period A
        label_b: Label for period B
        period_a_df: Optional original Period A dataframe (before merge) for accurate averages
        period_b_df: Optional original Period B dataframe (before merge) for accurate averages

    Returns:
        Summary DataFrame with formatted values
    """
    summary_data = []

    for metric in metrics:
        col_a = f"{metric}_A"
        col_b = f"{metric}_B"
        col_delta = f"{metric}_Delta"
        col_delta_pct = f"{metric}_Delta_Pct"

        if col_a in comparison_df.columns and col_b in comparison_df.columns:
            # Use original period DataFrames if provided (avoids 0-filled merge issues)
            # This ensures averages are calculated only from entities that exist in each period
            if period_a_df is not None and metric in period_a_df.columns:
                avg_a = period_a_df[metric].mean()
            else:
                avg_a = comparison_df[col_a].mean()

            if period_b_df is not None and metric in period_b_df.columns:
                avg_b = period_b_df[metric].mean()
            else:
                avg_b = comparison_df[col_b].mean()

            avg_delta = avg_b - avg_a

            # Determine if this is a percentage metric (contains '%' in name)
            is_percentage = '%' in metric
            is_time = 'time' in metric.lower()

            # Format values based on metric type
            if is_percentage:
                # Percentage metrics - values are stored as decimals (0.25 = 25%)
                formatted_a = f"{avg_a*100:.1f}%"
                formatted_b = f"{avg_b*100:.1f}%"
                # Calculate percentage change (not percentage point change)
                # Example: 25% to 30% = (30-25)/25 * 100 = 20% increase
                pct_change = (avg_delta / avg_a * 100) if avg_a != 0 else 0
                formatted_delta = f"{pct_change:+.1f}%"
            elif is_time:
                # Time metrics - show in days
                formatted_a = f"{avg_a:.1f}d"
                formatted_b = f"{avg_b:.1f}d"
                formatted_delta = f"{avg_delta:+.1f}d"
            else:
                # Numeric metrics (Score, counts)
                formatted_a = f"{avg_a:.1f}"
                formatted_b = f"{avg_b:.1f}"
                formatted_delta = f"{avg_delta:+.1f}"

            summary_data.append({
                'Metric': metric,
                f'Avg {label_a}': formatted_a,
                f'Avg {label_b}': formatted_b,
                'Avg Change': formatted_delta
            })

    return pd.DataFrame(summary_data)
