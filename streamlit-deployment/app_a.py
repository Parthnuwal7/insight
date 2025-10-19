"""
API-Driven Dashboard for Streamlit Cloud deployment
Processes actual backend API responses from HF Spaces
Uses real PyABSA sentiment analysis results
"""

import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta, date
import base64
from io import BytesIO
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Install streamlit-option-menu if not available
try:
    from streamlit_option_menu import option_menu
except ImportError:
    st.error("Please install streamlit-option-menu: pip install streamlit-option-menu")
    st.stop()

# Enhanced page configuration
st.set_page_config(
    page_title="Advanced Sentiment Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
HF_SPACES_API_URL = "https://parthnuwal7-absa.hf.space"

# Enhanced CSS styling for professional dashboard
def apply_custom_css():
    st.markdown("""
    <style>
        /* Main app styling */
        .main {
            padding-top: 0rem;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(45deg, #ff6b6b, #ff8e53);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        .metric-title {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-delta {
            font-size: 12px;
            opacity: 0.8;
        }
        
        /* Charts and visualizations */
        .stPlotlyChart {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Buttons and inputs */
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border-radius: 20px;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

# Enhanced color schemes for visualizations
COLOR_SCHEMES = {
    'sentiment': {
        'POSITIVE': '#2ecc71',
        'NEGATIVE': '#e74c3c', 
        'NEUTRAL': '#95a5a6',
        'Positive': '#2ecc71',
        'Negative': '#e74c3c',
        'Neutral': '#95a5a6'
    },
    'intent': {
        'COMPLAINT': '#e74c3c',
        'APPRECIATION': '#2ecc71',
        'QUESTION': '#f39c12',
        'SUGGESTION': '#3498db',
        'OTHER': '#95a5a6',
        'complaint': '#e74c3c',
        'appreciation': '#2ecc71',
        'question': '#f39c12',
        'suggestion': '#3498db',
        'neutral': '#95a5a6'
    },
    'language': {
        'HINDI': '#ff7675',
        'ENGLISH': '#74b9ff',
        'OTHER': '#a29bfe',
        'hi': '#ff7675',
        'en': '#74b9ff'
    },
    'gradient': ['#667eea', '#764ba2', '#ff6b6b', '#ff8e53']
}

def call_ml_backend(data: Dict, user_id: str = "default") -> Dict:
    """
    Call the ML backend API on HF Spaces with task tracking support.
    
    Args:
        data: Request data containing reviews
        user_id: User identifier for task tracking
        
    Returns:
        Response dict with status, data, and task_id
    """
    try:
        # Add user_id to request
        request_data = {**data, "user_id": user_id}
        
        response = requests.post(
            f"{HF_SPACES_API_URL}/process-reviews",
            json=request_data,
            timeout=900  # 15 minutes max timeout (server will handle its own timeout)
        )
        
        if response.status_code == 200:
            result = response.json()
            # Store task_id in session state if present
            if 'data' in result and 'task_id' in result['data']:
                st.session_state.current_task_id = result['data']['task_id']
            return result
        else:
            try:
                error_detail = response.json()
                return {
                    "success": False,
                    "error": f"API Error {response.status_code}: {error_detail.get('detail', response.text)}"
                }
            except:
                return {
                    "success": False,
                    "error": f"API Error {response.status_code}: {response.text}"
                }
                
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Backend processing timeout (>15 minutes)"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Backend connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def cancel_task(task_id: str) -> bool:
    """
    Cancel a running backend task.
    
    Args:
        task_id: Task identifier to cancel
        
    Returns:
        True if cancellation successful, False otherwise
    """
    try:
        response = requests.post(
            f"{HF_SPACES_API_URL}/cancel-task/{task_id}",
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('status') == 'success'
        return False
        
    except Exception as e:
        st.error(f"Failed to cancel task: {str(e)}")
        return False

def get_task_status(task_id: str) -> Optional[Dict]:
    """
    Get status of a backend task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task status dict or None if error
    """
    try:
        response = requests.get(
            f"{HF_SPACES_API_URL}/task-status/{task_id}",
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('task')
        return None
        
    except Exception:
        return None

def cancel_current_task() -> bool:
    """
    Cancel the current task stored in session state.
    
    Returns:
        True if cancellation successful
    """
    if 'current_task_id' in st.session_state and st.session_state.current_task_id:
        task_id = st.session_state.current_task_id
        
        if cancel_task(task_id):
            st.success(f"✅ Task {task_id[:8]}... cancelled successfully!")
            st.session_state.current_task_id = None
            st.session_state.processing = False
            return True
        else:
            st.error("❌ Failed to cancel task")
            return False
    else:
        st.warning("⚠️ No active task to cancel")
        return False

def parse_backend_response(response: Dict) -> Optional[pd.DataFrame]:
    """
    Parse backend API response and extract processed data.
    Handles different response formats from FastAPI backend.
    """
    try:
        # Check response status
        if not response:
            st.error("Empty response from backend")
            return None
        
        # Debug: Show response structure
        with st.expander("🔍 Debug: Raw Backend Response", expanded=False):
            st.write("**Response type:**", type(response))
            st.write("**Response keys:**", list(response.keys()) if isinstance(response, dict) else "Not a dict")
            st.json(response)
        
        # Handle error responses
        if response.get("success") == False or "error" in response:
            st.error(f"❌ Backend Error: {response.get('error', 'Unknown error')}")
            return None
        
        # Extract processed data from different possible response structures
        processed_data = None
        
        # Format 1: {status: "success", data: {processed_data: [...]}}
        if "data" in response:
            data_section = response["data"]
            if isinstance(data_section, dict) and "processed_data" in data_section:
                processed_data = data_section["processed_data"]
                st.info("✓ Found data in: response['data']['processed_data']")
            elif isinstance(data_section, list):
                processed_data = data_section
                st.info("✓ Found data in: response['data'] (direct array)")
        
        # Format 2: {processed_data: [...]}
        elif "processed_data" in response:
            processed_data = response["processed_data"]
            st.info("✓ Found data in: response['processed_data']")
        
        # Format 3: Direct array response
        elif isinstance(response, list):
            processed_data = response
            st.info("✓ Response is a direct array")
        
        # Format 4: Check for status field
        elif "status" in response and response["status"] == "success":
            # Look for data in other common keys
            for key in ['results', 'output', 'items', 'records']:
                if key in response:
                    processed_data = response[key]
                    st.info(f"✓ Found data in: response['{key}']")
                    break
        
        if processed_data is None:
            st.error("❌ Could not find processed data in response")
            st.write("Available keys:", list(response.keys()) if isinstance(response, dict) else "N/A")
            return None
        
        if not processed_data:
            st.warning("⚠️ Backend returned empty data")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        with st.expander("🔍 Debug: DataFrame Before Normalization", expanded=False):
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", list(df.columns))
            st.write("**Sample record:**")
            if len(df) > 0:
                st.json(df.iloc[0].to_dict())
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error parsing backend response: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def normalize_backend_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize backend column names to frontend expectations.
    Backend may return: overall_sentiment, detected_language
    Frontend expects: sentiment, language
    """
    if df is None or len(df) == 0:
        return df
    
    df_normalized = df.copy()
    
    # Column mapping from backend to frontend
    column_mapping = {
        'overall_sentiment': 'sentiment',
        'detected_language': 'language',
        'aspect_sentiments': 'aspect_sentiments',  # Keep as is
        'aspects': 'aspects'  # Keep as is
    }
    
    st.write("🔄 Normalizing columns...")
    
    # Apply mapping
    for backend_col, frontend_col in column_mapping.items():
        if backend_col in df_normalized.columns and backend_col != frontend_col:
            df_normalized[frontend_col] = df_normalized[backend_col]
            st.write(f"  ✓ Mapped {backend_col} → {frontend_col}")
    
    # Ensure required columns exist with defaults
    required_columns = {
        'sentiment': 'Neutral',
        'language': 'en',
        'intent': 'neutral',
        'aspects': '[]',
        'aspect_sentiments': '[]'
    }
    
    for col, default_value in required_columns.items():
        if col not in df_normalized.columns:
            # Try to find it from backend columns first
            backend_alternatives = {
                'sentiment': ['overall_sentiment', 'sentiment_label', 'polarity'],
                'language': ['detected_language', 'lang', 'language_code'],
                'intent': ['intent_label', 'classification'],
                'aspects': ['extracted_aspects', 'aspect_terms'],
                'aspect_sentiments': ['aspect_polarities', 'aspect_labels']
            }
            
            found = False
            if col in backend_alternatives:
                for alt_col in backend_alternatives[col]:
                    if alt_col in df_normalized.columns:
                        df_normalized[col] = df_normalized[alt_col]
                        st.write(f"  ✓ Mapped alternative {alt_col} → {col}")
                        found = True
                        break
            
            if not found:
                df_normalized[col] = default_value
                st.write(f"  ⚠️ Added missing '{col}' column with default: {default_value}")
    
    # Standardize sentiment values (capitalize first letter)
    if 'sentiment' in df_normalized.columns:
        df_normalized['sentiment'] = df_normalized['sentiment'].astype(str).str.capitalize()
    
    # Standardize language values (lowercase)
    if 'language' in df_normalized.columns:
        df_normalized['language'] = df_normalized['language'].astype(str).str.lower()
    
    with st.expander("🔍 Debug: DataFrame After Normalization", expanded=True):
        st.write("**Shape:**", df_normalized.shape)
        st.write("**Columns:**", list(df_normalized.columns))
        st.write("**Required columns check:**")
        for col in required_columns.keys():
            exists = col in df_normalized.columns
            st.write(f"  - {col}: {'✅ Present' if exists else '❌ Missing'}")
        if len(df_normalized) > 0:
            st.write("**Sample record:**")
            st.json(df_normalized.iloc[0].to_dict())
    
    return df_normalized

class SessionManager:
    """Lightweight session management for frontend-only deployment"""
    
    def __init__(self):
        self.session_id = f"session_{int(time.time())}"
        
    def save_session(self, data: pd.DataFrame, filename: str):
        """Save session data to browser session state"""
        if 'saved_sessions' not in st.session_state:
            st.session_state.saved_sessions = {}
        
        session_info = {
            'data': data.to_dict('records'),
            'columns': list(data.columns),
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'total_reviews': len(data)
        }
        
        st.session_state.saved_sessions[self.session_id] = session_info
        return self.session_id
    
    def load_session(self, session_id: str) -> Optional[pd.DataFrame]:
        """Load session data from browser session state"""
        if 'saved_sessions' not in st.session_state:
            return None
            
        if session_id in st.session_state.saved_sessions:
            session_info = st.session_state.saved_sessions[session_id]
            return pd.DataFrame(session_info['data'])
        return None
    
    def get_all_sessions(self) -> Dict:
        """Get all saved sessions"""
        return st.session_state.get('saved_sessions', {})

# Visualization functions
def create_sentiment_timeline(df: pd.DataFrame) -> go.Figure:
    """Create timeline chart of sentiment trends"""
    if 'sentiment' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Sentiment data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        timeline_data = df.groupby([pd.Grouper(key='date', freq='D'), 'sentiment']).size().reset_index(name='count')
        
        fig = px.line(
            timeline_data,
            x='date',
            y='count',
            color='sentiment',
            color_discrete_map=COLOR_SCHEMES['sentiment'],
            title="📈 Sentiment Timeline",
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Reviews",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    else:
        # If no date column, show distribution
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map=COLOR_SCHEMES['sentiment'],
            title="📊 Sentiment Distribution"
        )
        
        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Count",
            template='plotly_white'
        )
        
        return fig

def create_kpi_cards(df: pd.DataFrame):
    """Create KPI metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_reviews = len(df)
    
    # Positive sentiment percentage
    if 'sentiment' in df.columns:
        positive_pct = (df['sentiment'].str.upper() == 'POSITIVE').mean() * 100
    else:
        positive_pct = 0
    
    # Multilingual content
    if 'language' in df.columns:
        multilingual_pct = (df['language'].str.upper() != 'EN').mean() * 100
    else:
        multilingual_pct = 0
    
    # Average aspects per review
    if 'aspects' in df.columns:
        def count_aspects(x):
            try:
                if isinstance(x, str):
                    aspects = eval(x) if x and x != '[]' else []
                elif isinstance(x, list):
                    aspects = x
                else:
                    aspects = []
                return len(aspects) if isinstance(aspects, list) else 0
            except:
                return 0
        
        avg_aspects = df['aspects'].apply(count_aspects).mean()
    else:
        avg_aspects = 0
    
    with col1:
        st.metric(
            label="📊 Total Reviews",
            value=f"{total_reviews:,}",
            delta="Processed"
        )
    
    with col2:
        st.metric(
            label="😊 Positive Sentiment",
            value=f"{positive_pct:.1f}%",
            delta=f"{int(positive_pct * total_reviews / 100)} reviews"
        )
    
    with col3:
        st.metric(
            label="🌍 Multilingual Content",
            value=f"{multilingual_pct:.1f}%",
            delta=f"{int(multilingual_pct * total_reviews / 100)} reviews"
        )
    
    with col4:
        st.metric(
            label="🎯 Avg Aspects",
            value=f"{avg_aspects:.1f}",
            delta="Per review"
        )

def create_wordcloud(df: pd.DataFrame, sentiment: str = 'all') -> Optional[str]:
    """
    Create word cloud for specific sentiment
    Returns base64 encoded image
    """
    try:
        if 'review' not in df.columns:
            return None
        
        # Filter by sentiment if specified
        if sentiment != 'all' and 'sentiment' in df.columns:
            filtered_df = df[df['sentiment'].str.upper() == sentiment.upper()]
        else:
            filtered_df = df
        
        if len(filtered_df) == 0:
            return None
        
        # Combine all reviews
        text = ' '.join(filtered_df['review'].astype(str).tolist())
        
        if not text.strip():
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis' if sentiment == 'all' else ('Greens' if sentiment.upper() == 'POSITIVE' else 'Reds'),
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Convert to image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close()
        
        # Encode to base64
        img_base64 = base64.b64encode(buf.read()).decode()
        return img_base64
        
    except Exception as e:
        st.warning(f"Could not generate word cloud: {str(e)}")
        return None

def create_aspect_sentiment_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap of aspect-sentiment relationships"""
    try:
        if 'aspects' not in df.columns or 'sentiment' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Aspect and sentiment data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="🔥 Aspect-Sentiment Heatmap", template='plotly_white')
            return fig
        
        # Expand aspects and create aspect-sentiment combinations
        aspect_sentiment_data = []
        
        for idx, row in df.iterrows():
            try:
                aspects_value = row['aspects']
                sentiment_value = row['sentiment']
                
                # Skip if either is NA
                if pd.isna(aspects_value) or pd.isna(sentiment_value):
                    continue
                
                # Handle different types for aspects
                if isinstance(aspects_value, str):
                    aspects_str = aspects_value.strip()
                    if not aspects_str or aspects_str == '[]':
                        continue
                    try:
                        aspects = eval(aspects_str)
                    except:
                        aspects = [aspects_str]
                elif isinstance(aspects_value, (list, tuple)):
                    aspects = list(aspects_value)
                elif hasattr(aspects_value, '__iter__') and not isinstance(aspects_value, str):
                    try:
                        aspects = list(aspects_value)
                    except:
                        aspects = [str(aspects_value)]
                else:
                    aspects = [str(aspects_value)]
                
                # Add to data - aspects should now always be a list
                if aspects:
                    for aspect in aspects:
                        if aspect and str(aspect).strip():
                            aspect_sentiment_data.append({
                                'aspect': str(aspect),
                                'sentiment': str(sentiment_value)
                            })
            except Exception:
                continue
        
        if not aspect_sentiment_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No aspect data available for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="🔥 Aspect-Sentiment Heatmap", template='plotly_white')
            return fig
        
        aspect_df = pd.DataFrame(aspect_sentiment_data)
        
        # Create pivot table
        heatmap_data = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        
        # Take top 15 aspects
        top_aspects = heatmap_data.sum(axis=1).nlargest(15).index
        heatmap_data = heatmap_data.loc[top_aspects]
        
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            title="🔥 Aspect-Sentiment Heatmap (Top 15 Aspects)",
            labels=dict(x="Sentiment", y="Aspect", color="Count")
        )
        
        fig.update_layout(
            template='plotly_white',
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate heatmap: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_intent_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart of intent vs sentiment"""
    try:
        if 'intent' not in df.columns or 'sentiment' not in df.columns:
            return go.Figure()
        
        # Group by intent and sentiment
        grouped = df.groupby(['intent', 'sentiment']).size().reset_index(name='count')
        
        fig = px.bar(
            grouped,
            x='intent',
            y='count',
            color='sentiment',
            title="🎯 Intent vs Sentiment Distribution",
            color_discrete_map=COLOR_SCHEMES['sentiment'],
            barmode='stack'
        )
        
        fig.update_layout(
            xaxis_title="Intent",
            yaxis_title="Count",
            template='plotly_white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate intent chart: {str(e)}")
        return go.Figure()

def create_language_distribution(df: pd.DataFrame) -> go.Figure:
    """Create language distribution donut chart"""
    try:
        if 'language' not in df.columns:
            return go.Figure()
        
        lang_counts = df['language'].value_counts()
        
        # Map language codes to names
        lang_names = {
            'en': 'English',
            'hi': 'Hindi',
            'ENGLISH': 'English',
            'HINDI': 'Hindi'
        }
        
        lang_labels = [lang_names.get(lang, lang) for lang in lang_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=lang_labels,
            values=lang_counts.values,
            hole=0.4,
            marker=dict(colors=[COLOR_SCHEMES['language'].get(lang, '#a29bfe') for lang in lang_counts.index])
        )])
        
        fig.update_layout(
            title="🌍 Language Distribution",
            template='plotly_white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate language chart: {str(e)}")
        return go.Figure()

def create_aspect_network(df: pd.DataFrame) -> go.Figure:
    """Create network graph of aspect co-occurrences"""
    try:
        if 'aspects' not in df.columns:
            return go.Figure()
        
        # Extract aspects
        all_aspects = []
        for idx, row in df.iterrows():
            try:
                aspects_value = row['aspects']
                
                # Skip if NA
                if pd.isna(aspects_value):
                    continue
                
                # Handle different types
                if isinstance(aspects_value, str):
                    aspects_str = aspects_value.strip()
                    if not aspects_str or aspects_str == '[]':
                        continue
                    try:
                        aspects = eval(aspects_str)
                    except:
                        aspects = [aspects_str]
                elif isinstance(aspects_value, (list, tuple)):
                    aspects = list(aspects_value)
                elif hasattr(aspects_value, '__iter__') and not isinstance(aspects_value, str):
                    try:
                        aspects = list(aspects_value)
                    except:
                        aspects = [str(aspects_value)]
                else:
                    aspects = [str(aspects_value)]
                
                # Only add if we have multiple aspects (for co-occurrence)
                if aspects and len(aspects) > 1:
                    all_aspects.append([str(a) for a in aspects if a and str(a).strip()])
            except Exception:
                continue
        
        if not all_aspects:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough aspect co-occurrences for network",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="🕸️ Aspect Network", template='plotly_white')
            return fig
        
        # Build network
        G = nx.Graph()
        
        for aspects in all_aspects:
            for i in range(len(aspects)):
                for j in range(i + 1, len(aspects)):
                    if G.has_edge(aspects[i], aspects[j]):
                        G[aspects[i]][aspects[j]]['weight'] += 1
                    else:
                        G.add_edge(aspects[i], aspects[j], weight=1)
        
        # Get top edges
        edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:30]
        G_filtered = nx.Graph()
        for u, v, data in edges:
            G_filtered.add_edge(u, v, weight=data['weight'])
        
        # Layout
        pos = nx.spring_layout(G_filtered, k=1, iterations=50)
        
        # Create edges
        edge_x = []
        edge_y = []
        for edge in G_filtered.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G_filtered.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G_filtered.degree(node) * 10 + 10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color='#667eea',
                line=dict(width=2, color='white')
            ),
            hoverinfo='text'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="🕸️ Aspect Co-occurrence Network",
            showlegend=False,
            hovermode='closest',
            template='plotly_white',
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate network: {str(e)}")
        return go.Figure()

def create_top_aspects_chart(df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of top aspects"""
    try:
        if 'aspects' not in df.columns:
            return go.Figure()
        
        # Extract all aspects
        aspect_list = []
        for idx, row in df.iterrows():
            try:
                aspects_value = row['aspects']
                
                # Skip if NA or empty
                if pd.isna(aspects_value):
                    continue
                
                # Handle different types
                if isinstance(aspects_value, str):
                    aspects_str = aspects_value.strip()
                    if not aspects_str or aspects_str == '[]':
                        continue
                    # Try to evaluate string representation
                    try:
                        aspects = eval(aspects_str)
                    except:
                        # If eval fails, treat as single aspect
                        aspects = [aspects_str]
                elif isinstance(aspects_value, (list, tuple)):
                    aspects = list(aspects_value)
                elif hasattr(aspects_value, '__iter__') and not isinstance(aspects_value, str):
                    # Handle numpy arrays or other iterables
                    try:
                        aspects = list(aspects_value)
                    except:
                        aspects = [str(aspects_value)]
                else:
                    # Single value
                    aspects = [str(aspects_value)]
                
                # Add to list - aspects should now always be a list
                if aspects:  # Check if list is not empty
                    for aspect in aspects:
                        if aspect and str(aspect).strip():  # Skip empty strings
                            aspect_list.append(str(aspect))
                            
            except Exception as e:
                # Debug: show which row caused issue
                continue
        
        if not aspect_list:
            return go.Figure()
        
        # Count aspects
        aspect_counts = pd.Series(aspect_list).value_counts().head(15)
        
        fig = go.Figure(go.Bar(
            x=aspect_counts.values,
            y=aspect_counts.index,
            orientation='h',
            marker=dict(
                color=aspect_counts.values,
                colorscale='Viridis',
                showscale=True
            ),
            text=aspect_counts.values,
            textposition='outside'
        ))
        
        fig.update_layout(
            title="🏆 Top 15 Mentioned Aspects",
            xaxis_title="Frequency",
            yaxis_title="Aspect",
            template='plotly_white',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate top aspects chart: {str(e)}")
        return go.Figure()

def show_home_page():
    """Display the home page with file upload and processing"""
    st.markdown('<div class="dashboard-header"><h1>🎯 Abstract Sentiment Analysis Dashboard</h1><p>AI-Powered Review Analytics with PyABSA</p></div>', unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Reviews Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with columns: id, reviews_title, review, date, user_id",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['id', 'reviews_title', 'review', 'date', 'user_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: id, reviews_title, review, date, user_id")
                return
            
            st.success(f"✅ Loaded {len(df)} reviews successfully!")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # Initialize session state for processing control
            if 'processing' not in st.session_state:
                st.session_state.processing = False
            if 'current_task_id' not in st.session_state:
                st.session_state.current_task_id = None
            
            # Create columns for Process and Stop buttons
            col1, col2 = st.columns([3, 1])
            
            with col1:
                process_btn = st.button(
                    "🚀 Process Reviews with AI", 
                    type="primary",
                    disabled=st.session_state.processing,
                    use_container_width=True
                )
            
            with col2:
                stop_btn = st.button(
                    "🛑 Stop",
                    type="secondary",
                    disabled=not st.session_state.processing,
                    use_container_width=True
                )
            
            # Handle stop button
            if stop_btn:
                if cancel_current_task():
                    st.rerun()
            
            # Process data
            if process_btn:
                st.session_state.processing = True
                
                # Create progress tracking containers
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                with status_placeholder.container():
                    with st.spinner("🤖 Processing reviews with PyABSA backend..."):
                        # Prepare data for API
                        records = []
                        for _, row in df.iterrows():
                            record = {
                                "id": int(row.get('id', 0)),
                                "reviews_title": str(row.get('reviews_title', '')),
                                "review": str(row.get('review', '')),
                                "date": str(row.get('date', '2024-01-01')),
                                "user_id": str(row.get('user_id', 'unknown'))
                            }
                            records.append(record)
                        
                        api_data = {
                            "data": records,
                            "options": {
                                "include_translation": True,
                                "include_aspects": True
                            }
                        }
                        
                        # Debug: Show API request
                        with st.expander("🔍 Debug: API Request", expanded=False):
                            st.json({
                                "url": f"{HF_SPACES_API_URL}/process-reviews",
                                "sample_record": records[0] if records else {},
                                "total_records": len(records)
                            })
                        
                        # Get user ID from session
                        user_id = st.session_state.get('user_id', 'streamlit_user')
                        
                        # Call ML backend with progress tracking
                        result = call_ml_backend(api_data, user_id=user_id)
                        
                        # Check if processing was cancelled or timed out
                        if result.get('status') == 'cancelled':
                            st.warning("⚠️ Processing was cancelled")
                            st.session_state.processing = False
                            st.stop()
                        elif result.get('status') == 'timeout':
                            st.error(f"⏱️ {result.get('message', 'Processing timeout')}")
                            st.info("💡 Try processing with fewer reviews or retry later")
                            st.session_state.processing = False
                            st.stop()
                        
                        # Parse backend response
                        processed_df = parse_backend_response(result)
                        
                        if processed_df is not None and len(processed_df) > 0:
                            # Normalize column names
                            processed_df = normalize_backend_columns(processed_df)
                            
                            # Save to session state
                            st.session_state.processed_data = processed_df
                            st.session_state.filename = uploaded_file.name
                            st.session_state.processing = False
                            
                            # Save session
                            session_manager = SessionManager()
                            session_id = session_manager.save_session(processed_df, uploaded_file.name)
                            
                            st.success("✅ Analysis completed! Check the Analytics tab for detailed insights.")
                            
                            # Show quick stats
                            st.markdown("### 📊 Quick Stats")
                            create_kpi_cards(processed_df)
                            
                            # Show sample results
                            with st.expander("🔍 Sample Analysis Results", expanded=True):
                                display_cols = ['review', 'sentiment', 'aspects', 'intent', 'language']
                                available_cols = [col for col in display_cols if col in processed_df.columns]
                                if available_cols:
                                    st.dataframe(processed_df.head(5)[available_cols], use_container_width=True)
                                else:
                                    st.dataframe(processed_df.head(5), use_container_width=True)
                        else:
                            st.error("❌ Failed to process data. Check debug sections above for details.")
                            st.session_state.processing = False
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            import traceback
            with st.expander("🔍 Debug: Error Details", expanded=True):
                st.code(traceback.format_exc())
    
    else:
        # Show sample data format
        st.markdown("### 📋 Sample Data Format")
        sample_data = {
            'id': [1, 2, 3],
            'reviews_title': ['Great Product', 'Slow Service', 'Need Help'],
            'review': [
                'Great product! Love the quality and design.',
                'Delivery was slow but the item is good.',
                'How do I verify my account? Need assistance.'
            ],
            'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'user_id': ['user123', 'user456', 'user789']
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

def show_analytics_page():
    """Display the analytics page with advanced visualizations"""
    st.markdown("## 📈 Advanced Analytics Dashboard")
    
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Please upload and process data first on the Home page.")
        return
    
    df = st.session_state.processed_data
    
    st.info(f"📊 Analyzing {len(df)} reviews from **{st.session_state.get('filename', 'uploaded file')}**")
    
    # Show KPIs
    create_kpi_cards(df)
    
    # Add filters
    st.markdown("---")
    st.markdown("### 🎛️ Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        if 'sentiment' in df.columns:
            sentiments = ['All'] + list(df['sentiment'].unique())
            selected_sentiment = st.selectbox("Filter by Sentiment", sentiments)
    
    with filter_col2:
        if 'intent' in df.columns:
            intents = ['All'] + list(df['intent'].unique())
            selected_intent = st.selectbox("Filter by Intent", intents)
    
    with filter_col3:
        if 'language' in df.columns:
            languages = ['All'] + list(df['language'].unique())
            selected_language = st.selectbox("Filter by Language", languages)
    
    # Apply filters
    filtered_df = df.copy()
    if 'sentiment' in df.columns and selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    if 'intent' in df.columns and selected_intent != 'All':
        filtered_df = filtered_df[filtered_df['intent'] == selected_intent]
    if 'language' in df.columns and selected_language != 'All':
        filtered_df = filtered_df[filtered_df['language'] == selected_language]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} reviews")
    
    # Visualizations
    st.markdown("---")
    st.markdown("### 📊 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_sentiment_timeline(filtered_df)
        st.plotly_chart(fig, use_container_width=True, key="sentiment_timeline_chart")
    
    with col2:
        if 'intent' in filtered_df.columns:
            fig = create_intent_sentiment_chart(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="intent_sentiment_chart")
    
    # Word Clouds Section
    st.markdown("---")
    st.markdown("### ☁️ Word Clouds")
    
    wc_col1, wc_col2, wc_col3 = st.columns(3)
    
    with wc_col1:
        st.markdown("#### 😊 Positive Reviews")
        positive_wc = create_wordcloud(filtered_df, 'positive')
        if positive_wc:
            st.image(f"data:image/png;base64,{positive_wc}", use_container_width=True)
        else:
            st.info("No positive reviews found")
    
    with wc_col2:
        st.markdown("#### 😐 Neutral Reviews")
        neutral_wc = create_wordcloud(filtered_df, 'neutral')
        if neutral_wc:
            st.image(f"data:image/png;base64,{neutral_wc}", use_container_width=True)
        else:
            st.info("No neutral reviews found")
    
    with wc_col3:
        st.markdown("#### 😞 Negative Reviews")
        negative_wc = create_wordcloud(filtered_df, 'negative')
        if negative_wc:
            st.image(f"data:image/png;base64,{negative_wc}", use_container_width=True)
        else:
            st.info("No negative reviews found")
    
    # Aspect Analysis Section
    st.markdown("---")
    st.markdown("### 🎯 Aspect Analysis")
    
    aspect_col1, aspect_col2 = st.columns(2)
    
    with aspect_col1:
        fig = create_top_aspects_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True, key="top_aspects_chart")
    
    with aspect_col2:
        fig = create_aspect_sentiment_heatmap(filtered_df)
        st.plotly_chart(fig, use_container_width=True, key="aspect_heatmap_chart")
    
    # Additional Charts
    st.markdown("---")
    st.markdown("### 📈 Additional Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        fig = create_language_distribution(filtered_df)
        st.plotly_chart(fig, use_container_width=True, key="language_dist_chart")
    
    with insight_col2:
        fig = create_aspect_network(filtered_df)
        st.plotly_chart(fig, use_container_width=True, key="aspect_network_chart")
    
    # Intent Distribution Pie Chart
    if 'intent' in filtered_df.columns and len(filtered_df) > 0:
        st.markdown("---")
        st.markdown("### 🎯 Intent Distribution")
        intent_counts = filtered_df['intent'].value_counts()
        fig = px.pie(
            values=intent_counts.values,
            names=intent_counts.index,
            title="Complete Intent Breakdown",
            color=intent_counts.index,
            color_discrete_map=COLOR_SCHEMES['intent'],
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True, key="intent_pie_full_chart")
    
    # Full data table with download
    st.markdown("---")
    st.markdown("### 📋 Detailed Results")
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.dataframe(filtered_df, use_container_width=True)

def main():
    """Main application"""
    apply_custom_css()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=ABSA+AI", use_container_width=True)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Analytics", "Documentation"],
            icons=["house", "bar-chart", "book"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("AI-powered sentiment analysis using PyABSA and M2M100 translation. Processes reviews from backend API with real ML models.")
    
    # Page routing
    if selected == "Home":
        show_home_page()
    elif selected == "Analytics":
        show_analytics_page()
    elif selected == "Documentation":
        st.markdown("## 📚 Documentation")
        st.markdown("""
        ### Features
        - **Real PyABSA Processing**: Uses actual ML backend for sentiment analysis
        - **Aspect Extraction**: Identifies specific aspects in reviews
        - **Multilingual Support**: Hindi-to-English translation
        - **Intent Classification**: Categorizes review intent
        
        ### Backend API
        - **Endpoint**: `{HF_SPACES_API_URL}/process-reviews`
        - **Method**: POST
        - **Response**: JSON with processed sentiment data
        
        ### Column Mapping
        - `overall_sentiment` → `sentiment`
        - `detected_language` → `language`
        - Handles various response formats automatically
        """)

if __name__ == "__main__":
    main()
