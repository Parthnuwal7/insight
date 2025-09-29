"""
Enhanced visualization components for the advanced analytics dashboard.
Creates interactive charts, plots, network graphs, Sankey diagrams, and visual analytics.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import base64
from io import BytesIO
import networkx as nx
from collections import defaultdict

class KPIEngine:
    """Handles KPI calculations and display for the dashboard header."""
    
    @staticmethod
    def calculate_kpis(df: pd.DataFrame, areas_of_improvement: pd.DataFrame, 
                      strength_anchors: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all KPIs for the dashboard header.
        
        Args:
            df: Processed dataframe
            areas_of_improvement: Problem areas dataframe
            strength_anchors: Strength areas dataframe
            
        Returns:
            Dictionary with all KPI values
        """
        total_reviews = len(df)
        
        # Sentiment percentages
        positive_count = (df['overall_sentiment'] == 'Positive').sum()
        negative_count = (df['overall_sentiment'] == 'Negative').sum()
        neutral_count = (df['overall_sentiment'] == 'Neutral').sum()
        
        positive_pct = (positive_count / total_reviews * 100) if total_reviews > 0 else 0
        negative_pct = (negative_count / total_reviews * 100) if total_reviews > 0 else 0
        neutral_pct = (neutral_count / total_reviews * 100) if total_reviews > 0 else 0
        
        # Problem areas and strength anchors counts
        problem_areas_count = len(areas_of_improvement)
        strength_anchors_count = len(strength_anchors)
        
        # Language distribution
        languages = df['detected_language'].value_counts().to_dict()
        
        # Intent distribution
        intent_distribution = df['intent'].value_counts().to_dict()
        complaint_pct = (df['intent'] == 'complaint').mean() * 100
        
        return {
            'total_reviews': total_reviews,
            'positive_pct': round(positive_pct, 1),
            'negative_pct': round(negative_pct, 1),
            'neutral_pct': round(neutral_pct, 1),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'problem_areas_count': problem_areas_count,
            'strength_anchors_count': strength_anchors_count,
            'languages': languages,
            'intent_distribution': intent_distribution,
            'complaint_pct': round(complaint_pct, 1)
        }
    
    @staticmethod
    def create_kpi_header(kpis: Dict[str, Any]) -> None:
        """
        Create and display the KPI header section.
        
        Args:
            kpis: Dictionary containing all KPI values
        """
        # Main KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📊 Total Reviews",
                value=f"{kpis['total_reviews']:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="😊 Positive %",
                value=f"{kpis['positive_pct']}%",
                delta=f"{kpis['positive_count']} reviews"
            )
        
        with col3:
            st.metric(
                label="😞 Negative %", 
                value=f"{kpis['negative_pct']}%",
                delta=f"{kpis['negative_count']} reviews"
            )
        
        with col4:
            st.metric(
                label="🔴 Problem Areas",
                value=kpis['problem_areas_count'],
                delta="Aspects needing attention"
            )
        
        with col5:
            st.metric(
                label="🟢 Strength Anchors",
                value=kpis['strength_anchors_count'], 
                delta="Positive aspects"
            )


class AdvancedVisualizationEngine:
    """Enhanced visualization engine with advanced charts and network analysis."""
    
    def __init__(self):
        self.color_schemes = {
            'sentiment': {'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#708090'},
            'intent': px.colors.qualitative.Set3,
            'default': px.colors.qualitative.Plotly,
            'priority': px.colors.sequential.Reds,
            'strength': px.colors.sequential.Greens
        }
    
    def create_dual_ranking_tables(self, areas_of_improvement: pd.DataFrame, 
                                  strength_anchors: pd.DataFrame) -> None:
        """
        Create and display the dual ranking tables.
        
        Args:
            areas_of_improvement: Problem areas dataframe
            strength_anchors: Strength areas dataframe
        """
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Areas of Improvement")
            if len(areas_of_improvement) > 0:
                # Format the dataframe for display
                display_improvement = areas_of_improvement.copy()
                display_improvement['Negativity %'] = display_improvement['negativity_pct'].astype(str) + '%'
                display_improvement['Intent Severity'] = display_improvement['intent_severity']
                display_improvement['Frequency'] = display_improvement['frequency']
                display_improvement['Priority Score'] = display_improvement['priority_score']
                
                # Select columns for display
                display_cols = ['aspect', 'Negativity %', 'Intent Severity', 'Frequency', 'Priority Score']
                display_improvement = display_improvement[display_cols].rename(columns={'aspect': 'Aspect'})
                
                st.dataframe(
                    display_improvement,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No significant problem areas identified.")
        
        with col2:
            st.subheader("🟢 Strength Anchors")
            if len(strength_anchors) > 0:
                # Format the dataframe for display
                display_strength = strength_anchors.copy()
                display_strength['Positivity %'] = display_strength['positivity_pct'].astype(str) + '%'
                display_strength['Intent Type'] = display_strength['intent_type']
                display_strength['Frequency'] = display_strength['frequency']
                display_strength['Strength Score'] = display_strength['strength_score']
                
                # Select columns for display
                display_cols = ['aspect', 'Positivity %', 'Intent Type', 'Frequency', 'Strength Score']
                display_strength = display_strength[display_cols].rename(columns={'aspect': 'Aspect'})
                
                st.dataframe(
                    display_strength,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No significant strength anchors identified.")
    
    def create_aspect_network_graph(self, aspect_network: nx.Graph) -> go.Figure:
        """
        Create interactive network graph showing aspect relationships.
        
        Args:
            aspect_network: NetworkX graph with aspect co-occurrence data
            
        Returns:
            Plotly figure object
        """
        if len(aspect_network.nodes()) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No aspect relationships found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Aspect Co-occurrence Network")
            return fig
        
        # Calculate positions using spring layout
        pos = nx.spring_layout(aspect_network, k=3, iterations=50)
        
        # Extract node information
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_info = []
        
        for node in aspect_network.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            freq = aspect_network.nodes[node].get('frequency', 1)
            sentiment_score = aspect_network.nodes[node].get('sentiment_score', 0)
            color = aspect_network.nodes[node].get('color', 'gray')
            positive_pct = aspect_network.nodes[node].get('positive_pct', 0) * 100
            negative_pct = aspect_network.nodes[node].get('negative_pct', 0) * 100
            
            node_text.append(node)
            node_size.append(max(20, freq * 5))  # Size based on frequency
            
            # Color based on sentiment
            if color == 'green':
                node_color.append('#2E8B57')
            elif color == 'red':
                node_color.append('#DC143C')
            else:
                node_color.append('#708090')
            
            node_info.append(f"Aspect: {node}<br>" +
                           f"Frequency: {freq}<br>" +
                           f"Positive: {positive_pct:.1f}%<br>" +
                           f"Negative: {negative_pct:.1f}%")
        
        # Extract edge information
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in aspect_network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = aspect_network.edges[edge].get('weight', 1)
            edge_weights.append(weight)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            hovertext=node_info,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                            title=dict(text="Aspect Co-occurrence Network", font=dict(size=16)),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="Node size = frequency, Color = sentiment (green=positive, red=negative)",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(color='gray', size=10)
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600
                        ))
        
        return fig
    
    def create_intent_aspect_sankey(self, df: pd.DataFrame) -> go.Figure:
        """
        Create Sankey diagram showing Intent → Aspect → Sentiment flow.
        
        Args:
            df: Processed dataframe
            
        Returns:
            Plotly figure object
        """
        # Prepare data for Sankey diagram
        sankey_data = []
        
        for idx, row in df.iterrows():
            aspects = row['aspects'] if isinstance(row['aspects'], list) else []
            sentiments = row['aspect_sentiments'] if isinstance(row['aspect_sentiments'], list) else []
            intent = row['intent']
            
            for aspect, sentiment in zip(aspects, sentiments):
                sankey_data.append({
                    'intent': intent,
                    'aspect': aspect,
                    'sentiment': sentiment
                })
        
        if not sankey_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for Sankey diagram",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Intent → Aspect → Sentiment Flow")
            return fig
        
        sankey_df = pd.DataFrame(sankey_data)
        
        # Create node lists
        intents = sorted(sankey_df['intent'].unique())
        aspects = sorted(sankey_df['aspect'].unique())
        sentiments = sorted(sankey_df['sentiment'].unique())
        
        # Create labels and indices
        all_labels = intents + aspects + sentiments
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        
        # Create flows
        flows = []
        
        # Intent → Aspect flows
        intent_aspect_flows = sankey_df.groupby(['intent', 'aspect']).size().reset_index(name='count')
        for _, row in intent_aspect_flows.iterrows():
            flows.append({
                'source': label_to_idx[row['intent']],
                'target': label_to_idx[row['aspect']],
                'value': row['count']
            })
        
        # Aspect → Sentiment flows
        aspect_sentiment_flows = sankey_df.groupby(['aspect', 'sentiment']).size().reset_index(name='count')
        for _, row in aspect_sentiment_flows.iterrows():
            flows.append({
                'source': label_to_idx[row['aspect']],
                'target': label_to_idx[row['sentiment']],
                'value': row['count']
            })
        
        # Create colors
        intent_colors = ['rgba(255, 127, 14, 0.8)'] * len(intents)
        aspect_colors = ['rgba(31, 119, 180, 0.8)'] * len(aspects)
        sentiment_colors = []
        for sentiment in sentiments:
            if sentiment == 'Positive':
                sentiment_colors.append('rgba(46, 139, 87, 0.8)')
            elif sentiment == 'Negative':
                sentiment_colors.append('rgba(220, 20, 60, 0.8)')
            else:
                sentiment_colors.append('rgba(112, 128, 144, 0.8)')
        
        node_colors = intent_colors + aspect_colors + sentiment_colors
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=node_colors
            ),
            link=dict(
                source=[flow['source'] for flow in flows],
                target=[flow['target'] for flow in flows],
                value=[flow['value'] for flow in flows]
            )
        )])
        
        fig.update_layout(
            title=dict(text="Intent → Aspect → Sentiment Flow", font=dict(size=12)),
            font_size=12,
            height=600
        )
        
        return fig
    
    def create_enhanced_timeline_chart(self, df: pd.DataFrame, 
                                     annotations: Optional[List[Dict]] = None) -> go.Figure:
        """
        Create enhanced timeline chart with annotation support.
        
        Args:
            df: Processed dataframe with date and sentiment columns
            annotations: List of event annotations to add to chart
            
        Returns:
            Plotly figure object
        """
        try:
            # Ensure we have data
            if df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available for timeline chart",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font_size=16
                )
                fig.update_layout(title="Sentiment Trends Over Time")
                return fig
            
            # Ensure date column is properly formatted
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
            
            # Remove any rows with invalid dates
            df_copy = df_copy.dropna(subset=['date'])
            
            if df_copy.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No valid dates found in data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font_size=16
                )
                fig.update_layout(title="Sentiment Trends Over Time")
                return fig
            
            # Group by date and sentiment
            timeline_data = df_copy.groupby([df_copy['date'].dt.date, 'overall_sentiment']).size().reset_index(name='count')
            timeline_data['date'] = pd.to_datetime(timeline_data['date'])
            
            # Sort by date for proper line chart
            timeline_data = timeline_data.sort_values('date')
            
            # Create figure manually
            fig = go.Figure()
            
            # Add traces for each sentiment
            sentiments = timeline_data['overall_sentiment'].unique()
            for sentiment in sentiments:
                sentiment_data = timeline_data[timeline_data['overall_sentiment'] == sentiment].copy()
                
                fig.add_trace(go.Scatter(
                    x=sentiment_data['date'],
                    y=sentiment_data['count'],
                    mode='lines+markers',
                    name=sentiment,
                    line=dict(
                        color=self.color_schemes['sentiment'].get(sentiment, '#999999'),
                        width=2
                    ),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{sentiment}</b><br>Date: %{{x|%Y-%m-%d}}<br>Count: %{{y}}<extra></extra>'
                ))
            
            # Add simple text annotations (no arrows or complex shapes)
            if annotations and len(timeline_data) > 0:
                max_count = timeline_data['count'].max()
                y_range = max_count * 0.3  # Space for annotations
                
                for i, annotation in enumerate(annotations):
                    try:
                        # Convert annotation date
                        if isinstance(annotation['date'], str):
                            ann_date = pd.to_datetime(annotation['date'])
                        else:
                            ann_date = pd.to_datetime(annotation['date'])
                        
                        # Add simple text annotation
                        fig.add_annotation(
                            x=ann_date,
                            y=max_count + y_range * (0.2 + i * 0.3),
                            text=f"📌 {annotation['text']}",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="purple",
                            borderwidth=1,
                            font=dict(size=10, color="purple"),
                            xanchor="center"
                        )
                    except Exception:
                        # Skip problematic annotations silently
                        continue
            
            fig.update_layout(
                title="Sentiment Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Reviews",
                template='plotly_white',
                height=500,
                showlegend=True,
                xaxis=dict(
                    tickangle=45,
                    type='date'
                ),
                yaxis=dict(
                    rangemode='tozero'
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            # Fallback: return empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating timeline chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14, font_color="red"
            )
            fig.update_layout(title="Sentiment Trends Over Time")
            return fig
    
    def create_regional_language_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive language-wise sentiment and intent analysis.
        
        Args:
            df: Processed dataframe
            
        Returns:
            Plotly figure object with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Language Distribution', 'Sentiment by Language', 
                          'Intent by Language', 'Language Trends'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Language Distribution (Pie Chart)
        lang_counts = df['detected_language'].value_counts()
        fig.add_trace(
            go.Pie(labels=lang_counts.index, values=lang_counts.values, name="Languages"),
            row=1, col=1
        )
        
        # 2. Sentiment by Language (Stacked Bar)
        sentiment_lang = df.groupby(['detected_language', 'overall_sentiment']).size().unstack(fill_value=0)
        for sentiment in sentiment_lang.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment,
                    x=sentiment_lang.index,
                    y=sentiment_lang[sentiment],
                    marker_color=self.color_schemes['sentiment'].get(sentiment, '#808080')
                ),
                row=1, col=2
            )
        
        # 3. Intent by Language (Stacked Bar)
        intent_lang = df.groupby(['detected_language', 'intent']).size().unstack(fill_value=0)
        for intent in intent_lang.columns:
            fig.add_trace(
                go.Bar(
                    name=intent,
                    x=intent_lang.index,
                    y=intent_lang[intent],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Language Trends Over Time
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['date_str'] = df_copy['date'].dt.strftime('%Y-%m-%d')
        
        daily_lang = df_copy.groupby(['date_str', 'detected_language']).size().unstack(fill_value=0)
        daily_lang.index = pd.to_datetime(daily_lang.index)
        
        for lang in daily_lang.columns:
            fig.add_trace(
                go.Scatter(
                    x=daily_lang.index,
                    y=daily_lang[lang],
                    mode='lines',
                    name=f'{lang} trend',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title=dict(text="Regional/Language Analysis Dashboard", font=dict(size=16)),
            showlegend=True
        )
        
        return fig
    
    def create_alert_dashboard(self, sentiment_alerts: List[Dict[str, Any]]) -> None:
        """
        Create alert dashboard showing sentiment spikes.
        
        Args:
            sentiment_alerts: List of sentiment spike alerts
        """
        st.subheader("🚨 Sentiment Alerts")
        
        if not sentiment_alerts:
            st.info("No sentiment spikes detected in recent data.")
            return
        
        # Display alerts
        for i, alert in enumerate(sentiment_alerts[:5]):  # Show top 5 alerts
            severity_color = "🔴" if alert['alert_severity'] == 'high' else "🟡"
            
            with st.expander(f"{severity_color} {alert['aspect']} - {alert['spike_magnitude']}% increase"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Recent Avg Negative", f"{alert['recent_avg_negative']:.1f}")
                
                with col2:
                    st.metric("Previous Avg Negative", f"{alert['previous_avg_negative']:.1f}")
                
                with col3:
                    st.metric("Spike Magnitude", f"{alert['spike_magnitude']}%")
                
                st.warning(f"Aspect '{alert['aspect']}' showing {alert['alert_severity']} severity spike in negative sentiment.")
    
    def create_impact_simulation_tool(self, df: pd.DataFrame) -> None:
        """
        Create what-if analysis tool for aspect improvements.
        
        Args:
            df: Processed dataframe
        """
        st.subheader("🎯 Impact Simulation")
        st.write("Simulate the impact of fixing specific aspects on overall sentiment.")
        
        # Get list of negative aspects
        negative_aspects = []
        for idx, row in df.iterrows():
            aspects = row['aspects'] if isinstance(row['aspects'], list) else []
            sentiments = row['aspect_sentiments'] if isinstance(row['aspect_sentiments'], list) else []
            
            for aspect, sentiment in zip(aspects, sentiments):
                if sentiment == 'Negative':
                    negative_aspects.append(aspect)
        
        if not negative_aspects:
            st.info("No negative aspects found for simulation.")
            return
        
        unique_negative_aspects = list(set(negative_aspects))
        
        # Aspect selection
        selected_aspects = st.multiselect(
            "Select aspects to 'fix' (simulate removing negative reviews):",
            unique_negative_aspects,
            default=unique_negative_aspects[:3] if len(unique_negative_aspects) >= 3 else unique_negative_aspects
        )
        
        if selected_aspects:
            # Calculate current sentiment distribution
            current_sentiment = df['overall_sentiment'].value_counts()
            current_positive_pct = (current_sentiment.get('Positive', 0) / len(df)) * 100
            current_negative_pct = (current_sentiment.get('Negative', 0) / len(df)) * 100
            
            # Simulate fixing aspects (remove reviews with selected negative aspects)
            df_simulated = df.copy()
            for aspect in selected_aspects:
                # Remove reviews that mention this aspect negatively
                mask = df_simulated.apply(lambda row: not (
                    isinstance(row['aspects'], list) and 
                    isinstance(row['aspect_sentiments'], list) and
                    any(asp == aspect and sent == 'Negative' 
                        for asp, sent in zip(row['aspects'], row['aspect_sentiments']))
                ), axis=1)
                df_simulated = df_simulated[mask]
            
            # Calculate new sentiment distribution
            if len(df_simulated) > 0:
                new_sentiment = df_simulated['overall_sentiment'].value_counts()
                new_positive_pct = (new_sentiment.get('Positive', 0) / len(df_simulated)) * 100
                new_negative_pct = (new_sentiment.get('Negative', 0) / len(df_simulated)) * 100
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current Positive %", f"{current_positive_pct:.1f}%")
                    st.metric("Current Negative %", f"{current_negative_pct:.1f}%")
                
                with col2:
                    positive_change = new_positive_pct - current_positive_pct
                    negative_change = new_negative_pct - current_negative_pct
                    
                    st.metric(
                        "Simulated Positive %", 
                        f"{new_positive_pct:.1f}%",
                        delta=f"{positive_change:+.1f}%"
                    )
                    st.metric(
                        "Simulated Negative %", 
                        f"{new_negative_pct:.1f}%",
                        delta=f"{negative_change:+.1f}%"
                    )
                
                reviews_removed = len(df) - len(df_simulated)
                st.info(f"Simulation removed {reviews_removed} negative reviews mentioning the selected aspects.")
    
    def create_summary_sections(self, macro_summary: Dict[str, str], 
                               micro_summaries: Dict[str, str]) -> None:
        """
        Create and display macro and micro summary sections.
        
        Args:
            macro_summary: High-level insights
            micro_summaries: Aspect-specific insights
        """
        # Macro Summary
        st.subheader("📊 Executive Summary")
        for category, summary in macro_summary.items():
            st.write(f"**{category.replace('_', ' ').title()}:** {summary}")
        
        st.divider()
        
        # Micro Summaries
        st.subheader("🔍 Aspect-Level Insights")
        if micro_summaries:
            for aspect, summary in micro_summaries.items():
                with st.expander(f"📌 {aspect.title()} Analysis"):
                    st.write(summary)
        else:
            st.info("No detailed aspect summaries available.")


class ExportEngine:
    """Handles export functionality for reports and insights."""
    
    @staticmethod
    def generate_pdf_report(df: pd.DataFrame, kpis: Dict[str, Any], 
                           areas_of_improvement: pd.DataFrame,
                           strength_anchors: pd.DataFrame) -> bytes:
        """
        Generate PDF report with key insights.
        
        Args:
            df: Processed dataframe
            kpis: KPI dictionary
            areas_of_improvement: Problem areas dataframe
            strength_anchors: Strength areas dataframe
            
        Returns:
            PDF bytes
        """
        # Placeholder for PDF generation
        # In a real implementation, you would use libraries like reportlab or weasyprint
        pdf_content = f"""
        SENTIMENT ANALYSIS REPORT
        ========================
        
        Executive Summary:
        - Total Reviews: {kpis['total_reviews']}
        - Positive Sentiment: {kpis['positive_pct']}%
        - Negative Sentiment: {kpis['negative_pct']}%
        - Problem Areas: {kpis['problem_areas_count']}
        - Strength Anchors: {kpis['strength_anchors_count']}
        
        Top Issues:
        {areas_of_improvement[['aspect', 'priority_score']].to_string() if len(areas_of_improvement) > 0 else 'None'}
        
        Top Strengths:
        {strength_anchors[['aspect', 'strength_score']].to_string() if len(strength_anchors) > 0 else 'None'}
        """
        
        return pdf_content.encode('utf-8')
    
    @staticmethod
    def generate_excel_export(df: pd.DataFrame, areas_of_improvement: pd.DataFrame,
                             strength_anchors: pd.DataFrame) -> bytes:
        """
        Generate Excel export with multiple sheets.
        
        Args:
            df: Processed dataframe
            areas_of_improvement: Problem areas dataframe
            strength_anchors: Strength areas dataframe
            
        Returns:
            Excel bytes
        """
        from io import BytesIO
        import pandas as pd
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            areas_of_improvement.to_excel(writer, sheet_name='Problem Areas', index=False)
            strength_anchors.to_excel(writer, sheet_name='Strengths', index=False)
        
        return output.getvalue()
    
    def create_timeline_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create timeline chart showing sentiment trends over time.
        
        Args:
            df: Processed dataframe with date and sentiment columns
            
        Returns:
            Plotly figure object
        """
        # Ensure date column is properly formatted
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Group by date and sentiment - convert to string dates to avoid timestamp issues
        df_copy['date_str'] = df_copy['date'].dt.strftime('%Y-%m-%d')
        timeline_data = df_copy.groupby(['date_str', 'overall_sentiment']).size().reset_index(name='count')
        
        # Convert date strings back to datetime objects for plotting
        timeline_data['date'] = pd.to_datetime(timeline_data['date_str'])
        
        # Create the line chart
        fig = go.Figure()
        
        # Add traces for each sentiment
        for sentiment in timeline_data['overall_sentiment'].unique():
            sentiment_data = timeline_data[timeline_data['overall_sentiment'] == sentiment]
            
            fig.add_trace(go.Scatter(
                x=sentiment_data['date'],
                y=sentiment_data['count'],
                mode='lines+markers',
                name=sentiment,
                line=dict(color=self.color_schemes['sentiment'].get(sentiment, '#999999')),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Reviews",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_sentiment_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create pie chart for sentiment distribution."""
        sentiment_counts = df['overall_sentiment'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=.3,
            marker_colors=[self.color_schemes['sentiment'].get(label, '#808080') for label in sentiment_counts.index]
        )])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            title="Overall Sentiment Distribution",
            template='plotly_white'
        )
        
        return fig
    
    def create_intent_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart for intent distribution."""
        intent_counts = df['intent'].value_counts()
        
        fig = px.bar(
            x=intent_counts.index,
            y=intent_counts.values,
            title="Intent Classification Distribution",
            labels={'x': 'Intent Type', 'y': 'Number of Reviews'},
            color=intent_counts.index,
            color_discrete_sequence=self.color_schemes['intent']
        )
        
        fig.update_layout(
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_aspect_sentiment_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        Create heatmap showing sentiment distribution across aspects.
        
        Args:
            df: Processed dataframe with aspects and aspect_sentiments
            
        Returns:
            Plotly figure object
        """
        # Extract all aspects and their sentiments
        aspect_sentiment_data = []
        
        for idx, row in df.iterrows():
            aspects = row['aspects'] if isinstance(row['aspects'], list) else []
            sentiments = row['aspect_sentiments'] if isinstance(row['aspect_sentiments'], list) else []
            
            for aspect, sentiment in zip(aspects, sentiments):
                aspect_sentiment_data.append({'aspect': aspect, 'sentiment': sentiment})
        
        if not aspect_sentiment_data:
            # Return empty heatmap if no aspect data
            fig = go.Figure()
            fig.add_annotation(
                text="No aspect data available for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Aspect-Sentiment Heatmap")
            return fig
        
        # Create dataframe and pivot table
        aspect_df = pd.DataFrame(aspect_sentiment_data)
        heatmap_data = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=heatmap_data.values,
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title="Aspect-Sentiment Correlation Heatmap",
            xaxis_title="Sentiment",
            yaxis_title="Aspects",
            template='plotly_white'
        )
        
        return fig
    
    def create_wordcloud(self, df: pd.DataFrame, sentiment_filter: str = None) -> str:
        """
        Create word cloud from reviews.
        
        Args:
            df: Processed dataframe
            sentiment_filter: Filter by sentiment ('Positive', 'Negative', 'Neutral')
            
        Returns:
            Base64 encoded image string
        """
        # Filter data if sentiment filter is provided
        if sentiment_filter:
            filtered_df = df[df['overall_sentiment'] == sentiment_filter]
        else:
            filtered_df = df
        
        # Combine all translated reviews
        text = ' '.join(filtered_df['translated_review'].astype(str))
        
        if not text.strip():
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Convert to base64 for display
        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_b64 = base64.b64encode(img.read()).decode()
        
        return img_b64
    
    def create_language_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create pie chart for language distribution."""
        lang_counts = df['detected_language'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=lang_counts.index,
            values=lang_counts.values,
            hole=.3
        )])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            title="Language Distribution",
            template='plotly_white'
        )
        
        return fig
    
    def create_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Create correlation matrix for numerical relationships.
        
        Args:
            df: Processed dataframe
            
        Returns:
            Plotly figure object
        """
        # Create numerical features for correlation
        correlation_df = pd.DataFrame()
        
        # Add sentiment as numerical
        sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        correlation_df['sentiment_score'] = df['overall_sentiment'].map(sentiment_mapping)
        
        # Add intent as categorical numerical
        intent_mapping = {intent: idx for idx, intent in enumerate(df['intent'].unique())}
        correlation_df['intent_code'] = df['intent'].map(intent_mapping)
        
        # Add review length
        correlation_df['review_length'] = df['translated_review'].str.len()
        
        # Add number of aspects
        correlation_df['aspect_count'] = df['aspects'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Calculate correlation matrix
        corr_matrix = correlation_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            template='plotly_white'
        )
        
        return fig
    
    def create_aspect_frequency_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart showing most frequent aspects."""
        # Extract all aspects
        all_aspects = []
        for aspects in df['aspects']:
            if isinstance(aspects, list):
                all_aspects.extend(aspects)
        
        if not all_aspects:
            fig = go.Figure()
            fig.add_annotation(
                text="No aspects extracted from reviews",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Most Frequent Aspects")
            return fig
        
        # Count frequency
        aspect_counts = pd.Series(all_aspects).value_counts().head(15)
        
        fig = px.bar(
            x=aspect_counts.values,
            y=aspect_counts.index,
            orientation='h',
            title="Most Frequent Aspects (Top 15)",
            labels={'x': 'Frequency', 'y': 'Aspects'}
        )
        
        fig.update_layout(
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_daily_volume_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create chart showing daily review volume."""
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['date_str'] = df_copy['date'].dt.strftime('%Y-%m-%d')
        
        daily_counts = df_copy.groupby('date_str').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date_str'])
        daily_counts.columns = ['date_str', 'review_count', 'date']
        
        fig = px.bar(
            daily_counts,
            x='date',
            y='review_count',
            title="Daily Review Volume",
            labels={'review_count': 'Number of Reviews', 'date': 'Date'}
        )
        
        fig.update_layout(
            template='plotly_white',
            xaxis_title="Date",
            yaxis_title="Number of Reviews"
        )
        
        return fig


class FilterEngine:
    """Handles data filtering based on user selections."""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply multiple filters to the dataframe.
        
        Args:
            df: Original dataframe
            filters: Dictionary containing filter criteria
            
        Returns:
            Filtered dataframe
        """
        filtered_df = df.copy()
        
        # Date range filter
        if filters.get('date_range'):
            start_date, end_date = filters['date_range']
            # Convert to pandas datetime for comparison
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
            start_date_pd = pd.to_datetime(start_date)
            end_date_pd = pd.to_datetime(end_date)
            
            filtered_df = filtered_df[
                (filtered_df['date'] >= start_date_pd) & 
                (filtered_df['date'] <= end_date_pd)
            ]
        
        # Sentiment filter
        if filters.get('sentiment') and filters['sentiment'] != 'All':
            filtered_df = filtered_df[filtered_df['overall_sentiment'] == filters['sentiment']]
        
        # Intent filter
        if filters.get('intent') and filters['intent'] != 'All':
            filtered_df = filtered_df[filtered_df['intent'] == filters['intent']]
        
        # Language filter
        if filters.get('language') and filters['language'] != 'All':
            filtered_df = filtered_df[filtered_df['detected_language'] == filters['language']]
        
        # Aspect filter
        if filters.get('aspect') and filters['aspect'] != 'All':
            filtered_df = filtered_df[
                filtered_df['aspects'].apply(
                    lambda x: filters['aspect'] in x if isinstance(x, list) else False
                )
            ]
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame) -> Dict[str, List]:
        """
        Extract available filter options from the dataframe.
        
        Args:
            df: Processed dataframe
            
        Returns:
            Dictionary containing filter options
        """
        # Extract unique aspects
        all_aspects = set()
        for aspects in df['aspects']:
            if isinstance(aspects, list):
                all_aspects.update(aspects)
        
        return {
            'sentiments': ['All'] + sorted(df['overall_sentiment'].unique().tolist()),
            'intents': ['All'] + sorted(df['intent'].unique().tolist()),
            'languages': ['All'] + sorted(df['detected_language'].unique().tolist()),
            'aspects': ['All'] + sorted(list(all_aspects)),
            'date_range': (df['date'].min().date(), df['date'].max().date())
        }