"""
Dashboard Components for Advanced Analytics
Following the comprehensive layout specification
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import streamlit as st
from datetime import datetime, timedelta

# Color schemes (colorblind-safe)
SENTIMENT_COLORS = {
    'Positive': '#2E7D32',  # Green
    'Neutral': '#1976D2',   # Blue
    'Negative': '#D32F2F'   # Red
}

INTENT_COLORS = {
    'complaint': '#D32F2F',
    'praise': '#2E7D32',
    'question': '#1976D2',
    'suggestion': '#7B1FA2',
    'comparison': '#F57C00',
    'neutral': '#757575'
}

def extract_aspects_list(aspects_value) -> List[str]:
    """Safely extract aspects from various formats"""
    # Handle None first
    if aspects_value is None:
        return []
    
    # Handle list/tuple (most common case for your data)
    if isinstance(aspects_value, (list, tuple)):
        return [str(a).strip() for a in aspects_value if a and str(a).strip()]
    
    # Handle NaN for non-list types
    try:
        if pd.isna(aspects_value):
            return []
    except (TypeError, ValueError):
        # pd.isna() can fail on lists, ignore and continue
        pass
    
    # Handle string representations
    if isinstance(aspects_value, str):
        aspects_str = aspects_value.strip()
        if not aspects_str or aspects_str == '[]':
            return []
        try:
            # Try to evaluate string as Python literal
            aspects = eval(aspects_str)
            if isinstance(aspects, list):
                return [str(a).strip() for a in aspects if a and str(a).strip()]
            elif isinstance(aspects, tuple):
                return [str(a).strip() for a in aspects if a and str(a).strip()]
        except:
            # If eval fails, treat the whole string as one aspect
            return [aspects_str]
    
    # Fallback for other types
    return []

def get_all_unique_aspects(df: pd.DataFrame) -> List[str]:
    """Extract all unique aspects from dataframe (alphabetically sorted)"""
    all_aspects = set()
    if 'aspects' not in df.columns:
        return []
    
    for aspects_value in df['aspects']:
        aspects = extract_aspects_list(aspects_value)
        all_aspects.update(aspects)
    
    return sorted(list(all_aspects))

def get_top_aspects_by_frequency(df: pd.DataFrame, top_n: int = 15) -> List[str]:
    """Get top N most frequent aspects from dataframe"""
    if 'aspects' not in df.columns:
        return []
    
    aspect_counts = {}
    for aspects_value in df['aspects']:
        aspects = extract_aspects_list(aspects_value)
        for aspect in aspects:
            aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
    
    # Sort by frequency and return top N
    sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
    return [asp for asp, count in sorted_aspects[:top_n]]

def calculate_kpi_metrics(df: pd.DataFrame) -> Dict:
    """Calculate KPI metrics for dashboard"""
    metrics = {
        'total_reviews': len(df),
        'pct_positive': 0,
        'pct_negative': 0,
        'pct_neutral': 0,
        'top_negative_aspect': 'N/A',
        'top_positive_aspect': 'N/A',
        'avg_confidence': 0,
        'dominant_intent': 'N/A'
    }
    
    # Sentiment percentages
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        metrics['pct_positive'] = (sentiment_counts.get('Positive', 0) / total * 100) if total > 0 else 0
        metrics['pct_negative'] = (sentiment_counts.get('Negative', 0) / total * 100) if total > 0 else 0
        metrics['pct_neutral'] = (sentiment_counts.get('Neutral', 0) / total * 100) if total > 0 else 0
    
    # Top aspects by sentiment
    if 'aspects' in df.columns and 'sentiment' in df.columns:
        aspect_sentiment = []
        for idx, row in df.iterrows():
            aspects = extract_aspects_list(row.get('aspects'))
            sentiment = row.get('sentiment')
            for aspect in aspects:
                aspect_sentiment.append({'aspect': aspect, 'sentiment': sentiment})
        
        if aspect_sentiment:
            asp_df = pd.DataFrame(aspect_sentiment)
            
            # Top negative aspect
            neg_aspects = asp_df[asp_df['sentiment'] == 'Negative']['aspect'].value_counts()
            if len(neg_aspects) > 0:
                metrics['top_negative_aspect'] = neg_aspects.index[0]
            
            # Top positive aspect
            pos_aspects = asp_df[asp_df['sentiment'] == 'Positive']['aspect'].value_counts()
            if len(pos_aspects) > 0:
                metrics['top_positive_aspect'] = pos_aspects.index[0]
    
    # Average confidence
    if 'confidence' in df.columns:
        metrics['avg_confidence'] = df['confidence'].mean()
    
    # Dominant intent
    if 'intent' in df.columns:
        intent_counts = df['intent'].value_counts()
        if len(intent_counts) > 0:
            metrics['dominant_intent'] = intent_counts.index[0]
    
    return metrics

def create_enhanced_kpi_cards(df: pd.DataFrame):
    """Create KPI cards with sparklines and click interactions"""
    metrics = calculate_kpi_metrics(df)
    
    cols = st.columns(5)
    
    with cols[0]:
        st.metric(
            label="📝 Total Reviews",
            value=f"{metrics['total_reviews']:,}",
            delta=None
        )
    
    with cols[1]:
        st.metric(
            label="😊 Positive",
            value=f"{metrics['pct_positive']:.1f}%",
            delta=None
        )
    
    with cols[2]:
        st.metric(
            label="😞 Negative",
            value=f"{metrics['pct_negative']:.1f}%",
            delta=None
        )
    
    with cols[3]:
        st.metric(
            label="🔴 Top Complaint",
            value=metrics['top_negative_aspect'],
            delta=None,
            help="Most mentioned aspect in negative reviews"
        )
    
    with cols[4]:
        st.metric(
            label="🎯 Dominant Intent",
            value=metrics['dominant_intent'].title(),
            delta=None
        )

def create_sentiment_pie_chart(df: pd.DataFrame) -> go.Figure:
    """ROW 1.1: Sentiment distribution donut chart"""
    if 'sentiment' not in df.columns:
        return go.Figure()
    
    sentiment_counts = df['sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="📊 Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_intent_aspect_heatmap(df: pd.DataFrame) -> go.Figure:
    """ROW 1.2: Aspect (rows) × Intent (cols) heatmap with counts"""
    if 'aspects' not in df.columns or 'intent' not in df.columns:
        return go.Figure()
    
    # Build aspect-intent matrix
    aspect_intent_data = []
    for idx, row in df.iterrows():
        aspects = extract_aspects_list(row.get('aspects'))
        intent = row.get('intent', 'Unknown')
        for aspect in aspects:
            aspect_intent_data.append({'aspect': aspect, 'intent': intent})
    
    if not aspect_intent_data:
        return go.Figure()
    
    matrix_df = pd.DataFrame(aspect_intent_data)
    pivot = matrix_df.pivot_table(
        index='aspect',
        columns='intent',
        aggfunc='size',
        fill_value=0
    )
    
    # Sort by total count
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=False).head(15)
    pivot = pivot.drop('total', axis=1)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='YlOrRd',
        text=pivot.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Aspect: %{y}<br>Intent: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="🎯 Aspect × Intent Heatmap (Top 15 Aspects)",
        xaxis_title="Intent",
        yaxis_title="Aspect",
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_sentiment_aspect_heatmap(df: pd.DataFrame) -> go.Figure:
    """ROW 2.1: Aspect (rows) × Sentiment (cols) heatmap"""
    if 'aspects' not in df.columns or 'sentiment' not in df.columns:
        return go.Figure()
    
    # Build aspect-sentiment matrix
    aspect_sentiment_data = []
    for idx, row in df.iterrows():
        aspects = extract_aspects_list(row.get('aspects'))
        sentiment = row.get('sentiment', 'Unknown')
        for aspect in aspects:
            aspect_sentiment_data.append({'aspect': aspect, 'sentiment': sentiment})
    
    if not aspect_sentiment_data:
        return go.Figure()
    
    matrix_df = pd.DataFrame(aspect_sentiment_data)
    pivot = matrix_df.pivot_table(
        index='aspect',
        columns='sentiment',
        aggfunc='size',
        fill_value=0
    )
    
    # Ensure all sentiment columns exist
    for sent in ['Positive', 'Neutral', 'Negative']:
        if sent not in pivot.columns:
            pivot[sent] = 0
    
    pivot = pivot[['Positive', 'Neutral', 'Negative']]
    
    # Sort by total
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=False).head(15)
    pivot = pivot.drop('total', axis=1)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        text=pivot.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Aspect: %{y}<br>Sentiment: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="💭 Aspect × Sentiment Heatmap (Top 15 Aspects)",
        xaxis_title="Sentiment",
        yaxis_title="Aspect",
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_reviews_timeline(df: pd.DataFrame) -> go.Figure:
    """ROW 2.2: Timeline of review volume and sentiment trend"""
    if 'date' not in df.columns:
        return go.Figure()
    
    try:
        df['date'] = pd.to_datetime(df['date'])
    except:
        return go.Figure()
    
    # Daily counts
    daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='count')
    daily_counts.columns = ['date', 'count']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_counts['date'],
        y=daily_counts['count'],
        mode='lines+markers',
        name='Review Count',
        line=dict(color='#1976D2', width=2),
        fill='tozeroy',
        fillcolor='rgba(25, 118, 210, 0.1)'
    ))
    
    # Add sentiment breakdown if available
    if 'sentiment' in df.columns:
        for sentiment, color in SENTIMENT_COLORS.items():
            sent_df = df[df['sentiment'] == sentiment]
            if len(sent_df) > 0:
                sent_daily = sent_df.groupby(sent_df['date'].dt.date).size().reset_index(name='count')
                sent_daily.columns = ['date', 'count']
                
                fig.add_trace(go.Scatter(
                    x=sent_daily['date'],
                    y=sent_daily['count'],
                    mode='lines',
                    name=sentiment,
                    line=dict(color=color, width=1.5),
                    opacity=0.7
                ))
    
    fig.update_layout(
        title="📈 Reviews Timeline",
        xaxis_title="Date",
        yaxis_title="Number of Reviews",
        height=400,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_priority_leaderboard(df: pd.DataFrame) -> go.Figure:
    """ROW 3.1: Priority aspects ranking (needs urgent attention)"""
    if 'aspects' not in df.columns or 'sentiment' not in df.columns:
        return go.Figure()
    
    # Calculate priority scores
    aspect_data = []
    for idx, row in df.iterrows():
        aspects = extract_aspects_list(row.get('aspects'))
        sentiment = row.get('sentiment')
        for aspect in aspects:
            aspect_data.append({'aspect': aspect, 'sentiment': sentiment})
    
    if not aspect_data:
        return go.Figure()
    
    asp_df = pd.DataFrame(aspect_data)
    aspect_summary = []
    
    for aspect in asp_df['aspect'].unique():
        aspect_rows = asp_df[asp_df['aspect'] == aspect]
        total_count = len(aspect_rows)
        neg_count = len(aspect_rows[aspect_rows['sentiment'] == 'Negative'])
        neg_ratio = neg_count / total_count if total_count > 0 else 0
        
        # Priority score: negativity ratio * log(frequency)
        priority_score = neg_ratio * np.log1p(total_count)
        
        aspect_summary.append({
            'aspect': aspect,
            'priority_score': priority_score,
            'count': total_count,
            'neg_count': neg_count,
            'neg_ratio': neg_ratio
        })
    
    priority_df = pd.DataFrame(aspect_summary).sort_values('priority_score', ascending=False).head(10)
    
    # Color by severity
    colors = ['#D32F2F' if x > 0.5 else '#F57C00' if x > 0.3 else '#FDD835' 
              for x in priority_df['neg_ratio']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=priority_df['aspect'],
            x=priority_df['priority_score'],
            orientation='h',
            marker=dict(color=colors),
            text=priority_df['priority_score'].round(2),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Priority Score: %{x:.2f}<br>' +
                         'Mentions: ' + priority_df['count'].astype(str) + '<br>' +
                         'Negative: ' + (priority_df['neg_ratio'] * 100).round(1).astype(str) + '%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="🚨 Priority Aspects (Need Attention)",
        xaxis_title="Priority Score",
        yaxis_title="Aspect",
        height=500,
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_aspect_cooccurrence_heatmap(df: pd.DataFrame) -> go.Figure:
    """ROW 4.1: Aspect co-occurrence matrix"""
    if 'aspects' not in df.columns:
        return go.Figure()
    
    # Get aspect pairs
    from itertools import combinations
    
    # Get top 15 most frequent aspects (not just alphabetically sorted)
    top_aspects = get_top_aspects_by_frequency(df, top_n=15)
    
    if not top_aspects:
        fig = go.Figure()
        fig.add_annotation(
            text="No aspects found in the dataset",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Build co-occurrence matrix
    cooccur = {asp: {asp2: 0 for asp2 in top_aspects} for asp in top_aspects}
    
    # Track statistics for debugging
    total_reviews = 0
    reviews_with_multiple_aspects = 0
    
    for idx, row in df.iterrows():
        aspects = extract_aspects_list(row.get('aspects'))
        total_reviews += 1
        
        # Filter to only top aspects that exist in this review
        aspects_in_top = [a for a in aspects if a in top_aspects]
        
        # Count co-occurrences (only if we have at least 2 aspects)
        if len(aspects_in_top) >= 2:
            reviews_with_multiple_aspects += 1
            for asp1, asp2 in combinations(aspects_in_top, 2):
                cooccur[asp1][asp2] += 1
                cooccur[asp2][asp1] += 1  # Symmetric
    
    # Convert to matrix
    matrix = [[cooccur[asp1][asp2] for asp2 in top_aspects] for asp1 in top_aspects]
    
    # Check if matrix is all zeros
    total_cooccurrences = sum(sum(row) for row in matrix) // 2  # Divide by 2 since symmetric
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=top_aspects,
        y=top_aspects,
        colorscale='Blues',
        text=matrix,
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate='%{y} + %{x}<br>Co-occurrences: %{z}<extra></extra>'
    ))
    
    title_text = f"🔗 Aspect Co-occurrence Matrix (Top {len(top_aspects)} Aspects)"
    if total_cooccurrences == 0:
        title_text += f"<br><sub>⚠️ No co-occurrences found. Reviews with 2+ aspects: {reviews_with_multiple_aspects}/{total_reviews}</sub>"
    else:
        title_text += f"<br><sub>Total co-occurrences: {total_cooccurrences}</sub>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Aspect",
        yaxis_title="Aspect",
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_confidence_funnel(df: pd.DataFrame) -> go.Figure:
    """ROW 5.2: Model confidence distribution"""
    if 'confidence' not in df.columns:
        return go.Figure()
    
    # Create confidence buckets
    df_conf = df.copy()
    df_conf['conf_bucket'] = pd.cut(
        df_conf['confidence'],
        bins=[0, 0.6, 0.8, 0.9, 1.0],
        labels=['<0.6 (Low)', '0.6-0.8 (Medium)', '0.8-0.9 (Good)', '>0.9 (High)']
    )
    
    bucket_counts = df_conf['conf_bucket'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Funnel(
            y=bucket_counts.index,
            x=bucket_counts.values,
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(
                color=['#D32F2F', '#F57C00', '#FDD835', '#2E7D32']
            )
        )
    ])
    
    fig.update_layout(
        title="📊 Model Confidence Distribution",
        height=400,
        template='plotly_white'
    )
    
    return fig
