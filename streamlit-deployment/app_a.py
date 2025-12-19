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
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import telemetry helpers
try:
    from frontend_helpers import initialize_telemetry, log_event, submit_analysis_job, get_job_status
except ImportError:
    # Fallback if frontend_helpers not available
    def initialize_telemetry(*args, **kwargs): pass
    def log_event(*args, **kwargs): pass
    def submit_analysis_job(*args, **kwargs): return None
    def get_job_status(*args, **kwargs): return None

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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

# Initialize telemetry (session tracking, dashboard view event)
try:
    initialize_telemetry(HF_SPACES_API_URL)
except:
    pass  # Silently fail if backend not available

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

def create_aspect_network(df: pd.DataFrame, network_data: dict = None) -> go.Figure:
    """Create network graph of aspect co-occurrences"""
    try:
        # If backend provided network data, use it
        if network_data:
            try:
                import networkx as nx
                from networkx.readwrite import json_graph
                
                # Reconstruct graph from JSON
                G = json_graph.node_link_graph(network_data)
                
                if len(G.nodes()) == 0:
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No aspect relationships found",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    fig.update_layout(title="🕸️ Aspect Network", template='plotly_white')
                    return fig
                
                # Filter to top edges
                if len(G.edges()) > 30:
                    edges = sorted(G.edges(data=True), key=lambda x: x[2].get('weight', 1), reverse=True)[:30]
                    G_filtered = nx.Graph()
                    for u, v, data in edges:
                        G_filtered.add_edge(u, v, weight=data.get('weight', 1))
                else:
                    G_filtered = G
                
                # Layout
                pos = nx.spring_layout(G_filtered, k=1, iterations=50)
                
                # Create edges
                edge_x = []
                edge_y = []
                edge_weights = []
                
                for edge in G_filtered.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(edge[2].get('weight', 1))
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
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
                    node_text.append(str(node))
                    node_size.append(G_filtered.degree(node) * 10 + 15)
                
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
                    hoverinfo='text',
                    hovertext=node_text
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
                st.warning(f"Failed to load network from backend: {str(e)}")
                # Fall through to manual construction
        
        # Fallback: Build network manually from DataFrame
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
                
                # Add all aspects (not just multiple)
                if aspects:
                    cleaned_aspects = [str(a) for a in aspects if a and str(a).strip()]
                    if len(cleaned_aspects) > 1:
                        # For co-occurrence network
                        all_aspects.append(cleaned_aspects)
                    elif len(cleaned_aspects) == 1:
                        # Track single aspects too
                        all_aspects.append(cleaned_aspects)
            except Exception:
                continue
        
        if not all_aspects:
            fig = go.Figure()
            fig.add_annotation(
                text="No aspects found in the data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="🕸️ Aspect Network", template='plotly_white')
            return fig
        
        # Build network (with co-occurrences if available)
        G = nx.Graph()
        
        # Add edges for co-occurrences
        for aspects in all_aspects:
            if len(aspects) > 1:
                for i in range(len(aspects)):
                    for j in range(i + 1, len(aspects)):
                        if G.has_edge(aspects[i], aspects[j]):
                            G[aspects[i]][aspects[j]]['weight'] += 1
                        else:
                            G.add_edge(aspects[i], aspects[j], weight=1)
            else:
                # Add isolated node for single aspects
                G.add_node(aspects[0])
        
        if len(G.nodes()) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No aspects to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="🕸️ Aspect Network", template='plotly_white')
            return fig
        
        # If we have edges, filter to top ones
        if len(G.edges()) > 0:
            edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:30]
            G_filtered = nx.Graph()
            for u, v, data in edges:
                G_filtered.add_edge(u, v, weight=data['weight'])
        else:
            # No edges, just show top nodes
            G_filtered = G
            # Limit to top 20 most frequent aspects
            aspect_counts = {}
            for aspects in all_aspects:
                for aspect in aspects:
                    aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
            
            top_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            G_filtered = nx.Graph()
            for aspect, count in top_aspects:
                G_filtered.add_node(aspect, count=count)
        
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

# ========== ADMIN DASHBOARD FUNCTIONS ==========

def check_admin_auth():
    """Check if admin is authenticated."""
    if "admin_token" not in st.session_state:
        st.session_state.admin_token = None
    
    if not st.session_state.admin_token:
        st.warning("⚠️ Please enter admin token to access analytics")
        token = st.text_input("Admin Token", type="password", key="token_input")
        if st.button("Login"):
            st.session_state.admin_token = token
            st.rerun()
        return False
    
    return True


def get_admin_headers():
    """Get authorization headers for admin endpoints."""
    return {
        "Authorization": f"Bearer {st.session_state.admin_token}"
    }


def fetch_metrics_summary(api_url: str):
    """Fetch metrics summary from admin endpoint."""
    try:
        response = requests.get(
            f"{api_url}/admin/metrics/summary",
            headers=get_admin_headers(),
            timeout=10
        )
        
        if response.status_code == 401:
            st.error("❌ Invalid admin token. Please check and try again.")
            st.session_state.admin_token = None
            return None
        
        response.raise_for_status()
        return response.json()["data"]
    
    except Exception as e:
        st.error(f"Failed to fetch summary: {str(e)}")
        return None


def fetch_events_timeline(api_url: str, days: int = 7):
    """Fetch events timeline."""
    try:
        response = requests.get(
            f"{api_url}/admin/metrics/events?days={days}",
            headers=get_admin_headers(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()["data"]
    
    except Exception as e:
        st.error(f"Failed to fetch timeline: {str(e)}")
        return None


def fetch_funnel_analysis(api_url: str, days: int = 7):
    """Fetch funnel analysis."""
    try:
        response = requests.get(
            f"{api_url}/admin/metrics/funnel?days={days}",
            headers=get_admin_headers(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()["data"]
    
    except Exception as e:
        st.error(f"Failed to fetch funnel: {str(e)}")
        return None


def fetch_rate_limit_stats(api_url: str, days: int = 7):
    """Fetch rate limit statistics."""
    try:
        response = requests.get(
            f"{api_url}/admin/metrics/rate-limits?days={days}",
            headers=get_admin_headers(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()["data"]
    
    except Exception as e:
        st.error(f"Failed to fetch rate limit stats: {str(e)}")
        return None


def show_admin_page():
    """Display the admin analytics page."""
    st.markdown("## 🔒 Admin Analytics Dashboard")
    
    # Check authentication
    if not check_admin_auth():
        return
    
    # Logout button
    if st.button("Logout", key="admin_logout"):
        st.session_state.admin_token = None
        st.rerun()
    
    # Time range selector in sidebar
    st.sidebar.markdown("### ⚙️ Admin Settings")
    days = st.sidebar.slider(
        "Days to analyze",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to include in analysis"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    st.divider()
    
    # === METRICS SUMMARY ===
    st.markdown("### 📊 Metrics Summary")
    
    summary = fetch_metrics_summary(HF_SPACES_API_URL)
    
    if summary:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Unique Devices", summary["unique_devices"])
        
        with col2:
            st.metric("Unique Users", summary["unique_users"])
        
        with col3:
            total_events = sum(summary["events_by_type"].values())
            st.metric("Total Events", total_events)
        
        # Events by type
        st.markdown("#### Events by Type")
        
        events_df = pd.DataFrame([
            {"Event Type": k, "Count": v}
            for k, v in summary["events_by_type"].items()
        ])
        
        if not events_df.empty:
            fig = px.bar(
                events_df,
                x="Event Type",
                y="Count",
                color="Event Type",
                title="Event Distribution"
            )
            st.plotly_chart(fig, width='stretch', key="admin_events_bar")
    
    st.divider()
    
    # === EVENTS TIMELINE ===
    st.markdown("### 📈 Events Timeline")
    
    timeline_data = fetch_events_timeline(HF_SPACES_API_URL, days)
    
    if timeline_data and timeline_data["timeline"]:
        timeline_df = pd.DataFrame(timeline_data["timeline"])
        
        fig = px.line(
            timeline_df,
            x="date",
            y="count",
            color="event_type",
            title=f"Events Over Last {days} Days",
            labels={"count": "Event Count", "date": "Date", "event_type": "Event Type"}
        )
        st.plotly_chart(fig, width='stretch', key="admin_timeline")
    else:
        st.info("No timeline data available")
    
    st.divider()
    
    # === FUNNEL ANALYSIS ===
    st.markdown("### 🔀 Conversion Funnel")
    
    funnel_data = fetch_funnel_analysis(HF_SPACES_API_URL, days)
    
    if funnel_data:
        stages = funnel_data["funnel_stages"]
        conversions = funnel_data["conversion_rates"]
        
        # Funnel chart
        funnel_stages = ["DASHBOARD_VIEW", "ANALYSIS_REQUEST", "TASK_QUEUED", "TASK_COMPLETED"]
        funnel_values = [stages.get(stage, 0) for stage in funnel_stages]
        
        fig = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_values,
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(title=f"User Journey Funnel (Last {days} Days)")
        st.plotly_chart(fig, width='stretch', key="admin_funnel")
        
        # Conversion metrics
        st.markdown("#### Conversion Rates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("View → Request", f"{conversions['view_to_request']:.1f}%")
            st.metric("Request → Queued", f"{conversions['request_to_queued']:.1f}%")
        
        with col2:
            st.metric("Queued → Completed", f"{conversions['queued_to_completed']:.1f}%")
            st.metric("Overall Completion", f"{conversions['overall_completion']:.1f}%")
    
    st.divider()
    
    # === RATE LIMIT STATS ===
    st.markdown("### 🚦 Rate Limiting Statistics")
    
    rate_limit_data = fetch_rate_limit_stats(HF_SPACES_API_URL, days)
    
    if rate_limit_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Rate Limit Hits", rate_limit_data["total_hits"])
        
        with col2:
            top_devices = len(rate_limit_data["top_devices"])
            st.metric("Unique Devices Hit", top_devices)
        
        # Top offenders
        if rate_limit_data["top_devices"]:
            st.markdown("#### Top Rate Limited Devices")
            
            devices_df = pd.DataFrame(rate_limit_data["top_devices"])
            st.dataframe(devices_df, width='stretch')
        
        # Timeline
        if rate_limit_data["timeline"]:
            st.markdown("#### Rate Limit Hits Over Time")
            
            timeline_df = pd.DataFrame(rate_limit_data["timeline"])
            
            fig = px.bar(
                timeline_df,
                x="date",
                y="count",
                title="Daily Rate Limit Hits",
                labels={"count": "Hits", "date": "Date"}
            )
            st.plotly_chart(fig, width='stretch', key="admin_rate_limits")


def show_home_page():
    """Display the home page with file upload and processing"""
    st.markdown('<div class="dashboard-header"><h1>🎯 Aspect-Based Sentiment Analysis</h1><p>AI-Powered Multi-Aspect Review Analytics</p></div>', unsafe_allow_html=True)
    
    # Data source selection
    st.markdown("### 📊 Choose Data Source")
    
    data_source = st.radio(
        "Select how you want to provide data:",
        options=["📁 Upload Your Own Data", "🎲 Try Sample Datasets"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    uploaded_file = None
    df = None
    
    if data_source == "📁 Upload Your Own Data":
        st.markdown("#### 📤 Upload Your CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with columns: id, reviews_title, review (optional: date, user_id)",
            type=['csv'],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate minimum required columns
                if 'review' not in df.columns:
                    st.error("❌ Missing required column: 'review'")
                    st.info("Your CSV must have at least a 'review' column with review text")
                    return
                
                # Add missing optional columns with defaults
                if 'id' not in df.columns:
                    df['id'] = range(1, len(df) + 1)
                if 'reviews_title' not in df.columns:
                    df['reviews_title'] = 'Review ' + df['id'].astype(str)
                if 'date' not in df.columns:
                    df['date'] = datetime.now().strftime('%Y-%m-%d')
                if 'user_id' not in df.columns:
                    df['user_id'] = 'user_' + df['id'].astype(str)
                
                st.success(f"✅ Loaded {len(df)} reviews successfully!")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                return
    
    else:  # Sample datasets
        st.markdown("#### 🎲 Select a Sample Dataset")
        
        sample_choice = st.selectbox(
            "Choose a dataset to explore:",
            options=[
                "E-Commerce Product Reviews (22 reviews)",
                "Restaurant & Dining Reviews (15 reviews)",
                "Mobile App Reviews (30 reviews)"
            ],
            help="Sample datasets demonstrate different review types and analysis scenarios"
        )
        
        # Dataset descriptions
        if "E-Commerce" in sample_choice:
            st.info("""
            **📦 E-Commerce Product Reviews**
            - Mix of positive, negative, and neutral reviews
            - Aspects: Quality, Delivery, Price, Customer Service, Packaging
            - Includes mixed sentiment examples (good quality but slow delivery)
            - Contains Hindi reviews for multilingual testing
            """)
            sample_file = "test_data_ecommerce.csv"
        elif "Restaurant" in sample_choice:
            st.info("""
            **🍽️ Restaurant & Dining Reviews**
            - Variety of dining experiences
            - Aspects: Food Quality, Service, Ambiance, Portions, Price
            - Range from food safety complaints to 5-star experiences
            - Includes Hindi reviews
            """)
            sample_file = "test_data_restaurant.csv"
        else:  # App reviews
            st.info("""
            **📱 Mobile App Reviews**
            - Software and app user feedback
            - Aspects: Performance, UI/UX, Features, Battery, Privacy, Support
            - Mix of technical issues and feature praise
            - Includes Hindi reviews and update-related feedback
            """)
            sample_file = "test_data_app_reviews.csv"
        
        # Load sample data
        try:
            sample_path = os.path.join(os.path.dirname(__file__), sample_file)
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                
                # Add missing optional columns with defaults
                if 'date' not in df.columns:
                    df['date'] = datetime.now().strftime('%Y-%m-%d')
                if 'user_id' not in df.columns:
                    df['user_id'] = 'user_' + df['id'].astype(str)
                
                st.success(f"✅ Loaded {len(df)} sample reviews!")
            else:
                st.warning(f"⚠️ Sample file not found: {sample_file}")
                st.info("Please upload your own data instead.")
                return
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            return
    
    # Continue with processing if data is loaded
    if df is not None:
        # Show data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(), width='stretch')
        
        # Initialize session state for processing control
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'current_task_id' not in st.session_state:
            st.session_state.current_task_id = None
        
        # Process button
        process_btn = st.button(
            "🚀 Process Reviews with AI", 
            type="primary",
            disabled=st.session_state.processing,
            use_container_width=True
        )
        
        # Process data
        if process_btn:
            try:
                st.session_state.processing = True
                
                # Store filename for later reference
                if data_source == "📁 Upload Your Own Data":
                    st.session_state.filename = uploaded_file.name if uploaded_file else "uploaded_data.csv"
                else:
                    st.session_state.filename = sample_file
                
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
                        
                        # Also extract aspect_network if available
                        aspect_network = None
                        if isinstance(result, dict):
                            if "data" in result and isinstance(result["data"], dict):
                                aspect_network = result["data"].get("aspect_network")
                            elif "aspect_network" in result:
                                aspect_network = result["aspect_network"]
                        
                        if processed_df is not None and len(processed_df) > 0:
                            # Normalize column names
                            processed_df = normalize_backend_columns(processed_df)
                            
                            # Save to session state
                            st.session_state.processed_data = processed_df
                            st.session_state.aspect_network = aspect_network  # Store network data
                            
                            # Extract and store aspect-level data if available
                            if isinstance(result, dict):
                                if "data" in result and isinstance(result["data"], dict):
                                    # Store aspect-level data
                                    aspect_level = result["data"].get("aspect_level_data")
                                    if aspect_level:
                                        st.session_state.aspect_level_data = pd.DataFrame(aspect_level)
                                    
                                    # Store mixed sentiment reviews
                                    mixed_sentiment = result["data"].get("mixed_sentiment_reviews")
                                    if mixed_sentiment:
                                        st.session_state.mixed_sentiment_reviews = pd.DataFrame(mixed_sentiment)
                                    
                                    # Store summary statistics
                                    summary = result["data"].get("summary", {})
                                    if summary:
                                        st.session_state.analysis_summary = summary
                            
                            st.session_state.processing = False
                            
                            # Save session
                            session_manager = SessionManager()
                            filename = st.session_state.get('filename', 'data.csv')
                            session_id = session_manager.save_session(processed_df, filename)
                            
                            st.success("✅ Analysis completed! Check the Analytics tab for detailed insights.")
                            
                            # Show enhanced stats if available
                            st.markdown("### 📊 Quick Stats")
                            create_kpi_cards(processed_df)
                            
                            # Show aspect-level stats if available
                            if 'analysis_summary' in st.session_state:
                                summary = st.session_state.analysis_summary
                                st.markdown("#### 🔍 Aspect-Level Statistics")
                                
                                stat_col1, stat_col2, stat_col3 = st.columns(3)
                                with stat_col1:
                                    total_aspects = summary.get('total_aspects', 0)
                                    st.metric("Total Aspect Mentions", total_aspects)
                                with stat_col2:
                                    mixed_count = summary.get('mixed_sentiment_count', 0)
                                    st.metric("Mixed Sentiment Reviews", mixed_count)
                                with stat_col3:
                                    mixed_pct = summary.get('mixed_sentiment_pct', 0)
                                    st.metric("Mixed Sentiment %", f"{mixed_pct:.1f}%")
                            
                            # Show sample results
                            with st.expander("🔍 Sample Analysis Results", expanded=True):
                                display_cols = ['review', 'sentiment', 'aspects', 'intent', 'language']
                                available_cols = [col for col in display_cols if col in processed_df.columns]
                                if available_cols:
                                    st.dataframe(processed_df.head(5)[available_cols], width='stretch')
                                else:
                                    st.dataframe(processed_df.head(5), width='stretch')
                        else:
                            st.error("❌ Failed to process data. Check debug sections above for details.")
                            st.session_state.processing = False
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                import traceback
                with st.expander("🔍 Debug: Error Details", expanded=True):
                    st.code(traceback.format_exc())


def show_analytics_page():
    """Display the analytics page with advanced visualizations in tabs"""
    st.markdown("## 📈 Advanced Analytics Dashboard")
    
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Please upload and process data first on the Home page.")
        return
    
    df = st.session_state.processed_data
    
    # Get additional data if available
    aspect_level_df = st.session_state.get('aspect_level_data', pd.DataFrame())
    mixed_sentiment_df = st.session_state.get('mixed_sentiment_reviews', pd.DataFrame())
    
    from dashboard_components import (
        create_enhanced_kpi_cards,
        create_sentiment_pie_chart,
        create_intent_aspect_heatmap,
        create_sentiment_aspect_heatmap,
        create_reviews_timeline,
        create_priority_leaderboard,
        create_aspect_cooccurrence_heatmap,
        create_confidence_funnel,
        get_all_unique_aspects,
        extract_aspects_list
    )
    
    st.info(f"📊 Analyzing {len(df)} reviews from **{st.session_state.get('filename', 'uploaded file')}**")
    
    # ========== TOP ROW: ENHANCED KPI CARDS ==========
    create_enhanced_kpi_cards(df)
    
    # ========== TABS FOR DIFFERENT ANALYSIS VIEWS ==========
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔗 Multi-Aspect Analysis", "🔍 Deep Dive"])
    
    # ========== TAB 1: OVERVIEW ==========
    with tab1:
        show_overview_tab(df)
    
    # ========== TAB 2: MULTI-ASPECT ANALYSIS ==========
    with tab2:
        show_multi_aspect_tab(df, aspect_level_df, mixed_sentiment_df)
    
    # ========== TAB 3: DEEP DIVE ==========
    with tab3:
        show_deep_dive_tab(df)


def show_overview_tab(df):
    """Overview tab with existing univariate charts"""
    from dashboard_components import (
        create_sentiment_pie_chart,
        create_intent_aspect_heatmap,
        create_sentiment_aspect_heatmap,
        create_reviews_timeline,
        create_priority_leaderboard,
        create_aspect_cooccurrence_heatmap,
        create_confidence_funnel,
        get_all_unique_aspects,
        extract_aspects_list
    )
    
    # ========== FILTER BAR (TAB-SPECIFIC) ==========
    st.markdown("### 🎛️ Filters")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        if 'sentiment' in df.columns:
            sentiments = list(df['sentiment'].unique())
            selected_sentiments = st.multiselect(
                "Sentiment",
                options=sentiments,
                default=sentiments,
                help="Select one or more sentiments"
            )
        else:
            selected_sentiments = []
    
    with filter_col2:
        if 'intent' in df.columns:
            intents = list(df['intent'].unique())
            selected_intents = st.multiselect(
                "Intent",
                options=intents,
                default=intents,
                help="Select one or more intents"
            )
        else:
            selected_intents = []
    
    with filter_col3:
        if 'language' in df.columns:
            languages = list(df['language'].unique())
            selected_languages = st.multiselect(
                "Language",
                options=languages,
                default=languages,
                help="Select one or more languages"
            )
        else:
            selected_languages = []
    
    with filter_col4:
        if 'aspects' in df.columns:
            all_aspects = get_all_unique_aspects(df)
            selected_aspects = st.multiselect(
                "Aspects",
                options=all_aspects,
                default=[],
                help="Select one or more aspects (leave empty for all)"
            )
        else:
            selected_aspects = []
    
    # Date range filter
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("From Date", value=min_date, min_value=min_date, max_value=max_date)
            with date_col2:
                end_date = st.date_input("To Date", value=max_date, min_value=min_date, max_value=max_date)
        except:
            start_date = None
            end_date = None
    else:
        start_date = None
        end_date = None
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'sentiment' in df.columns and selected_sentiments:
        filtered_df = filtered_df[filtered_df['sentiment'].isin(selected_sentiments)]
    
    if 'intent' in df.columns and selected_intents:
        filtered_df = filtered_df[filtered_df['intent'].isin(selected_intents)]
    
    if 'language' in df.columns and selected_languages:
        filtered_df = filtered_df[filtered_df['language'].isin(selected_languages)]
    
    # Apply aspect filter
    if selected_aspects and 'aspects' in filtered_df.columns:
        def contains_any_aspect(aspects_value):
            aspects = extract_aspects_list(aspects_value)
            return any(asp in selected_aspects for asp in aspects)
        
        filtered_df = filtered_df[filtered_df['aspects'].apply(contains_any_aspect)]
    
    # Apply date filter
    if start_date and end_date and 'date' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) &
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    st.info(f"📊 Showing **{len(filtered_df)}** of **{len(df)}** reviews after filters")
    
    # Check if we have data to display
    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
        return
    
    # ========== ROW 1: SENTIMENT PIE + INTENT-ASPECT HEATMAP ==========
    st.markdown("---")
    st.markdown("### � Overview")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        fig = create_sentiment_pie_chart(filtered_df)
        st.plotly_chart(fig, width='stretch', key="sentiment_pie")
    
    with row1_col2:
        fig = create_intent_aspect_heatmap(filtered_df)
        st.plotly_chart(fig, width='stretch', key="intent_aspect_heatmap")
    
    # ========== ROW 2: ASPECT-SENTIMENT HEATMAP + REVIEWS TIMELINE ==========
    st.markdown("---")
    st.markdown("### 📈 Sentiment Patterns")
    
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        fig = create_sentiment_aspect_heatmap(filtered_df)
        st.plotly_chart(fig, width='stretch', key="sentiment_aspect_heatmap")
    
    with row2_col2:
        fig = create_reviews_timeline(filtered_df)
        st.plotly_chart(fig, width='stretch', key="reviews_timeline")
    
    # ========== ROW 3: PRIORITY LEADERBOARD + ASPECT SENTIMENT TRENDS ==========
    st.markdown("---")
    st.markdown("### � Priority Insights")
    
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        fig = create_priority_leaderboard(filtered_df)
        st.plotly_chart(fig, width='stretch', key="priority_leaderboard")
    
    with row3_col2:
        # Aspect sentiment trends over time (placeholder for now)
        st.markdown("#### 📊 Aspect Sentiment Trends")
        st.info("Coming soon: Time-series sentiment trends for top aspects")
        # TODO: Implement aspect sentiment trends chart
    
    # ========== ROW 4: CO-OCCURRENCE HEATMAP + LLM INSIGHTS ==========
    st.markdown("---")
    st.markdown("### 🔗 Correlations & Insights")
    
    row4_col1, row4_col2 = st.columns(2)
    
    with row4_col1:
        fig = create_aspect_cooccurrence_heatmap(filtered_df)
        st.plotly_chart(fig, width='stretch', key="cooccurrence_heatmap")
    
    with row4_col2:
        # LLM Insight Cards (placeholder)
        st.markdown("#### � AI-Generated Insights")
        st.info("**Coming soon:** Automated recommendations for priority aspects")
        
        # Placeholder insight cards
        st.markdown("""
        <div style='background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
        <h4 style='margin:0; color: #0369a1;'>🔍 Insight: Top Priority</h4>
        <p style='margin:5px 0 0 0;'>Based on analysis, <strong>Design</strong> requires immediate attention with 65% negative sentiment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #f0fdf4; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
        <h4 style='margin:0; color: #15803d;'>✅ Insight: Strength Area</h4>
        <p style='margin:5px 0 0 0;'><strong>Performance</strong> consistently receives positive feedback (78% positive).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # TODO: Implement actual LLM insights with caching
    
    # # ========== DIAGNOSTIC SECTION (TEMPORARY - Remove after debugging) ==========
    # from diagnostic_component import show_aspect_diagnostics
    # show_aspect_diagnostics(filtered_df)
    
    # ========== ROW 5: DRILL-DOWN PANEL + CONFIDENCE FUNNEL ==========
    st.markdown("---")
    st.markdown("### 🔍 Deep Dive Analysis")
    
    row5_col1, row5_col2 = st.columns([2, 1])
    
    with row5_col1:
        # Drill-down panel
        st.markdown("#### � Review Details")
        
        with st.expander("🔍 Click to explore individual reviews", expanded=False):
            # Add mini-filters for drill-down
            drill_col1, drill_col2 = st.columns(2)
            with drill_col1:
                sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Confidence (High)", "Confidence (Low)"])
            with drill_col2:
                page_size = st.selectbox("Reviews per page", [10, 25, 50, 100], index=1)
            
            # Sort filtered data
            drill_df = filtered_df.copy()
            if sort_by == "Date (Newest)" and 'date' in drill_df.columns:
                drill_df = drill_df.sort_values('date', ascending=False)
            elif sort_by == "Date (Oldest)" and 'date' in drill_df.columns:
                drill_df = drill_df.sort_values('date', ascending=True)
            elif sort_by == "Confidence (High)" and 'confidence' in drill_df.columns:
                drill_df = drill_df.sort_values('confidence', ascending=False)
            elif sort_by == "Confidence (Low)" and 'confidence' in drill_df.columns:
                drill_df = drill_df.sort_values('confidence', ascending=True)
            
            # Display paginated reviews
            total_pages = (len(drill_df) - 1) // page_size + 1
            page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
            
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(drill_df))
            
            st.info(f"Showing reviews {start_idx + 1} to {end_idx} of {len(drill_df)}")
            
            # Display reviews
            for idx, row in drill_df.iloc[start_idx:end_idx].iterrows():
                review_text = row.get('review', 'N/A')
                sentiment = row.get('sentiment', 'N/A')
                intent = row.get('intent', 'N/A')
                confidence = row.get('confidence', 0)
                aspects = extract_aspects_list(row.get('aspects', []))
                
                # Color code by sentiment
                if sentiment == 'Positive':
                    bg_color = '#f0fdf4'
                    border_color = '#22c55e'
                elif sentiment == 'Negative':
                    bg_color = '#fef2f2'
                    border_color = '#ef4444'
                else:
                    bg_color = '#f0f9ff'
                    border_color = '#3b82f6'
                
                st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 12px; border-left: 4px solid {border_color}; border-radius: 4px; margin-bottom: 10px;'>
                <p style='margin:0;'><strong>Review:</strong> {review_text[:200]}{'...' if len(review_text) > 200 else ''}</p>
                <p style='margin:5px 0 0 0; font-size: 0.9em;'>
                <strong>Sentiment:</strong> {sentiment} | 
                <strong>Intent:</strong> {intent} | 
                <strong>Confidence:</strong> {confidence:.2%} | 
                <strong>Aspects:</strong> {', '.join(aspects) if aspects else 'None'}
                </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Export drill-down data
            drill_csv = drill_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Drill-Down Data",
                data=drill_csv,
                file_name=f"drilldown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with row5_col2:
        # Confidence funnel
        fig = create_confidence_funnel(filtered_df)
        st.plotly_chart(fig, width='stretch', key="confidence_funnel")
    
    # ========== LEGACY SECTIONS (WORD CLOUDS & NETWORK) ==========
    st.markdown("---")
    st.markdown("### ☁️ Word Clouds")
    
    wc_col1, wc_col2, wc_col3 = st.columns(3)
    
    with wc_col1:
        st.markdown("#### 😊 Positive Reviews")
        positive_wc = create_wordcloud(filtered_df, 'positive')
        if positive_wc:
            st.image(f"data:image/png;base64,{positive_wc}", width='stretch')
        else:
            st.info("No positive reviews found")
    
    with wc_col2:
        st.markdown("#### 😐 Neutral Reviews")
        neutral_wc = create_wordcloud(filtered_df, 'neutral')
        if neutral_wc:
            st.image(f"data:image/png;base64,{neutral_wc}", width='stretch')
        else:
            st.info("No neutral reviews found")
    
    with wc_col3:
        st.markdown("#### 😞 Negative Reviews")
        negative_wc = create_wordcloud(filtered_df, 'negative')
        if negative_wc:
            st.image(f"data:image/png;base64,{negative_wc}", width='stretch')
        else:
            st.info("No negative reviews found")
    
    # ========== FOOTER: EXPORT & UTILITIES ==========
    st.markdown("---")
    st.markdown("### � Export & Utilities")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # Full CSV export
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width='stretch'
        )
    
    with export_col2:
        # Summary report
        st.button("📄 Generate PDF Report", disabled=True, width='stretch', help="Coming soon")
    
    with export_col3:
        # Alert setup
        st.button("🔔 Setup Alerts", disabled=True, width='stretch', help="Coming soon: Get notified when priority aspects spike")


def show_multi_aspect_tab(df, aspect_level_df, mixed_sentiment_df):
    """Multi-aspect analysis tab with relationship patterns and RAG insights"""
    from dashboard_components import extract_aspects_list
    
    st.markdown("### 🔗 Multi-Aspect Relationship Analysis")
    st.info("This tab shows how different aspects relate to each other and provides AI-generated insights about aspect patterns.")
    
    # Check if we have aspect-level data
    if aspect_level_df.empty:
        st.warning("⚠️ Aspect-level data not available. This may be an older analysis. Please re-process your data to enable multi-aspect analysis.")
        return
    
    # ========== TAB 2 FILTERS (INDEPENDENT FROM OVERVIEW) ==========
    st.markdown("#### 🎛️ Multi-Aspect Filters")
    
    ma_col1, ma_col2, ma_col3 = st.columns(3)
    
    with ma_col1:
        # Sentiment filter for aspect-level data
        if 'aspect_sentiment' in aspect_level_df.columns:
            ma_sentiments = list(aspect_level_df['aspect_sentiment'].unique())
            ma_selected_sentiments = st.multiselect(
                "Aspect Sentiment",
                options=ma_sentiments,
                default=ma_sentiments,
                help="Filter by sentiment at aspect level",
                key="ma_sentiment_filter"
            )
        else:
            ma_selected_sentiments = []
    
    with ma_col2:
        # Aspect filter
        if 'aspect' in aspect_level_df.columns:
            ma_aspects = list(aspect_level_df['aspect'].unique())
            ma_selected_aspects = st.multiselect(
                "Aspects",
                options=ma_aspects,
                default=ma_aspects[:10] if len(ma_aspects) > 10 else ma_aspects,
                help="Select aspects to analyze (showing top 10 by default)",
                key="ma_aspect_filter"
            )
        else:
            ma_selected_aspects = []
    
    with ma_col3:
        # Overall sentiment filter
        if 'overall_sentiment' in aspect_level_df.columns:
            ma_overall_sentiments = list(aspect_level_df['overall_sentiment'].unique())
            ma_selected_overall = st.multiselect(
                "Overall Review Sentiment",
                options=ma_overall_sentiments,
                default=ma_overall_sentiments,
                help="Filter by overall review sentiment",
                key="ma_overall_filter"
            )
        else:
            ma_selected_overall = []
    
    # Apply filters to aspect-level data
    ma_filtered_df = aspect_level_df.copy()
    
    if ma_selected_sentiments and 'aspect_sentiment' in ma_filtered_df.columns:
        ma_filtered_df = ma_filtered_df[ma_filtered_df['aspect_sentiment'].isin(ma_selected_sentiments)]
    
    if ma_selected_aspects and 'aspect' in ma_filtered_df.columns:
        ma_filtered_df = ma_filtered_df[ma_filtered_df['aspect'].isin(ma_selected_aspects)]
    
    if ma_selected_overall and 'overall_sentiment' in ma_filtered_df.columns:
        ma_filtered_df = ma_filtered_df[ma_filtered_df['overall_sentiment'].isin(ma_selected_overall)]
    
    st.info(f"📊 Analyzing **{len(ma_filtered_df)}** aspect mentions from **{ma_filtered_df['review_id'].nunique() if 'review_id' in ma_filtered_df.columns else 0}** reviews")
    
    if len(ma_filtered_df) == 0:
        st.warning("⚠️ No data matches the selected filters. Please adjust your criteria.")
        return
    
    # ========== MIXED SENTIMENT HIGHLIGHT ==========
    st.markdown("---")
    st.markdown("### ⚠️ Mixed Sentiment Reviews")
    
    if not mixed_sentiment_df.empty:
        mixed_count = len(mixed_sentiment_df)
        total_reviews = df['review_id'].nunique() if 'review_id' in df.columns else len(df)
        mixed_pct = (mixed_count / total_reviews * 100) if total_reviews > 0 else 0
        
        # Mixed sentiment KPI card
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); 
                    padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='margin:0; color: white; font-size: 1.2em;'>🔀 Mixed Sentiment Reviews</h3>
        <p style='margin: 8px 0 0 0; color: white; font-size: 2em; font-weight: bold;'>{mixed_count}</p>
        <p style='margin: 5px 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.95em;'>{mixed_pct:.1f}% of reviews have conflicting aspect sentiments</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show examples of mixed sentiment reviews
        with st.expander("🔍 View Mixed Sentiment Review Examples", expanded=False):
            st.markdown("These reviews mention multiple aspects with different sentiments (e.g., positive Quality but negative Delivery):")
            
            for idx, row in mixed_sentiment_df.head(5).iterrows():
                review_text = row.get('review', 'N/A')
                aspects_str = row.get('aspects', '')
                aspects = extract_aspects_list(aspects_str)
                
                st.markdown(f"""
                <div style='background-color: #fef3c7; padding: 12px; border-left: 4px solid #f59e0b; 
                            border-radius: 4px; margin-bottom: 10px;'>
                <p style='margin:0;'><strong>Review:</strong> {review_text[:250]}{'...' if len(review_text) > 250 else ''}</p>
                <p style='margin:5px 0 0 0; font-size: 0.9em;'>
                <strong>Aspects Mentioned:</strong> {', '.join(aspects[:5]) if aspects else 'None'}
                </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No mixed sentiment reviews detected in current dataset.")
    
    # ========== ASPECT-SENTIMENT MATRIX ==========
    st.markdown("---")
    st.markdown("### 📊 Aspect-Sentiment Distribution Matrix")
    
    if 'aspect' in ma_filtered_df.columns and 'aspect_sentiment' in ma_filtered_df.columns:
        # Create pivot table for aspect x sentiment
        aspect_sentiment_matrix = pd.crosstab(
            ma_filtered_df['aspect'],
            ma_filtered_df['aspect_sentiment'],
            normalize='index'
        ) * 100  # Convert to percentage
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=aspect_sentiment_matrix.values,
            x=aspect_sentiment_matrix.columns,
            y=aspect_sentiment_matrix.index,
            colorscale='RdYlGn',
            text=aspect_sentiment_matrix.values.round(1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Percentage")
        ))
        
        fig.update_layout(
            title="Aspect-Level Sentiment Distribution (%)",
            xaxis_title="Sentiment",
            yaxis_title="Aspect",
            height=max(400, len(aspect_sentiment_matrix) * 25),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True, key="aspect_sentiment_matrix")
    
    # ========== ASPECT CO-OCCURRENCE WITH SENTIMENT BREAKDOWN ==========
    st.markdown("---")
    st.markdown("### 🔗 Aspect Co-Occurrence Patterns")
    
    # Build co-occurrence matrix with sentiment breakdown
    if 'review_id' in ma_filtered_df.columns and 'aspect' in ma_filtered_df.columns:
        # Get aspects per review
        review_aspects = ma_filtered_df.groupby('review_id')['aspect'].apply(list).to_dict()
        
        # Get all unique aspects
        all_aspects = sorted(ma_filtered_df['aspect'].unique())
        
        # Build co-occurrence matrix
        cooccurrence = pd.DataFrame(0, index=all_aspects, columns=all_aspects)
        
        for aspects_list in review_aspects.values():
            for i, asp1 in enumerate(aspects_list):
                for asp2 in aspects_list[i+1:]:
                    if asp1 in all_aspects and asp2 in all_aspects:
                        cooccurrence.loc[asp1, asp2] += 1
                        cooccurrence.loc[asp2, asp1] += 1
        
        # Show top co-occurring pairs
        cooccur_pairs = []
        for i in range(len(all_aspects)):
            for j in range(i+1, len(all_aspects)):
                count = cooccurrence.iloc[i, j]
                if count > 0:
                    cooccur_pairs.append({
                        'Aspect 1': all_aspects[i],
                        'Aspect 2': all_aspects[j],
                        'Co-occurrences': int(count)
                    })
        
        cooccur_df = pd.DataFrame(cooccur_pairs).sort_values('Co-occurrences', ascending=False)
        
        if len(cooccur_df) > 0:
            st.markdown("#### Top Aspect Pairs Mentioned Together")
            
            # Show top 10 pairs as bar chart
            top_pairs = cooccur_df.head(10).copy()
            top_pairs['Pair'] = top_pairs['Aspect 1'] + ' + ' + top_pairs['Aspect 2']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_pairs['Co-occurrences'],
                    y=top_pairs['Pair'],
                    orientation='h',
                    marker=dict(
                        color=top_pairs['Co-occurrences'],
                        colorscale='Blues',
                        showscale=False
                    ),
                    text=top_pairs['Co-occurrences'],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Most Frequently Co-Occurring Aspect Pairs",
                xaxis_title="Number of Reviews",
                yaxis_title="Aspect Pair",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True, key="cooccurrence_bar")
        else:
            st.info("No aspect co-occurrences found in filtered data.")
    
    # ========== AI INSIGHTS AND RECOMMENDATIONS ==========
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Insights & Recommendations")
    
    # Generate LLM-powered insights
    with st.spinner("🧠 Generating AI insights..."):
        llm_insights = generate_llm_insights(ma_filtered_df)
    
    if llm_insights:
        # Display AI-generated insights in a highlighted box
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
        <h3 style='margin:0 0 15px 0; color: white; font-size: 1.3em;'>🤖 AI Analysis</h3>
        <div style='background-color: rgba(255,255,255,0.95); padding: 20px; border-radius: 8px; color: #1f2937;'>
        {llm_insights.replace(chr(10), '<br>')}
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("💡 AI insights are currently unavailable. Showing pattern-based insights instead.")
        
        # Fallback to rule-based insights
        insights = generate_rag_insights(ma_filtered_df)
        
        for insight in insights:
            st.markdown(f"""
            <div style='background-color: {insight['bg_color']}; padding: 15px; border-radius: 8px; margin-bottom: 15px;
                        border-left: 4px solid {insight['border_color']};'>
            <h4 style='margin:0; color: {insight['text_color']};'>{insight['icon']} {insight['title']}</h4>
            <p style='margin:8px 0 0 0; color: #1f2937; font-size: 1.05em;'>{insight['message']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== ASPECT-LEVEL DATA EXPORT ==========
    st.markdown("---")
    st.markdown("### 📥 Export Aspect-Level Data")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Export filtered aspect-level data
        csv = ma_filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Aspect-Level Data (CSV)",
            data=csv,
            file_name=f"aspect_level_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # Export mixed sentiment reviews
        if not mixed_sentiment_df.empty:
            mixed_csv = mixed_sentiment_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Mixed Sentiment Reviews (CSV)",
                data=mixed_csv,
                file_name=f"mixed_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )


def generate_llm_insights(aspect_level_df: pd.DataFrame) -> str:
    """Generate AI-powered insights using OpenRouter LLM"""
    
    if aspect_level_df.empty or not OPENROUTER_API_KEY:
        return ""
    
    try:
        # Prepare context data from aspect-level analysis
        context_data = prepare_analysis_context(aspect_level_df)
        
        # Create prompt for LLM
        prompt = f"""You are an expert business analyst reviewing customer feedback data. Based on the aspect-level sentiment analysis below, provide actionable insights and recommendations.

ANALYSIS DATA:
{context_data}

Please provide:
1. **Key Findings** (2-3 most important patterns)
2. **Strengths** (aspects performing well)
3. **Areas for Improvement** (aspects needing attention)
4. **Actionable Recommendations** (specific steps to take)

Keep your response concise, focused, and business-oriented. Use markdown formatting for clarity."""

        # Call OpenRouter API
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "ABSA Insights Dashboard"
        }
        
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
        else:
            st.error(f"OpenRouter API Error: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        st.error(f"Error generating LLM insights: {str(e)}")
        return ""


def prepare_analysis_context(aspect_level_df: pd.DataFrame) -> str:
    """Prepare structured context for LLM from aspect-level data"""
    
    context_parts = []
    
    # 1. Overall Statistics
    total_aspects = len(aspect_level_df)
    total_reviews = aspect_level_df['review_id'].nunique() if 'review_id' in aspect_level_df.columns else len(aspect_level_df)
    
    context_parts.append(f"**Total Reviews Analyzed:** {total_reviews}")
    context_parts.append(f"**Total Aspect Mentions:** {total_aspects}")
    context_parts.append("")
    
    # 2. Aspect-Level Sentiment Distribution
    if 'aspect' in aspect_level_df.columns and 'aspect_sentiment' in aspect_level_df.columns:
        aspect_sentiments = aspect_level_df.groupby(['aspect', 'aspect_sentiment']).size().unstack(fill_value=0)
        
        # Calculate percentages
        aspect_sentiments_pct = aspect_sentiments.div(aspect_sentiments.sum(axis=1), axis=0) * 100
        
        # Get top 8 aspects by mention count
        top_aspects = aspect_level_df['aspect'].value_counts().head(8)
        
        context_parts.append("**Top Aspects by Sentiment:**")
        for aspect in top_aspects.index:
            if aspect in aspect_sentiments_pct.index:
                row = aspect_sentiments_pct.loc[aspect]
                pos = row.get('Positive', 0)
                neg = row.get('Negative', 0)
                neu = row.get('Neutral', 0)
                total = top_aspects[aspect]
                
                context_parts.append(f"- **{aspect}** ({total} mentions): {pos:.0f}% Positive, {neg:.0f}% Negative, {neu:.0f}% Neutral")
        
        context_parts.append("")
    
    # 3. Co-occurrence Patterns (top 5 pairs)
    if 'review_id' in aspect_level_df.columns and 'aspect' in aspect_level_df.columns:
        review_aspects = aspect_level_df.groupby('review_id')['aspect'].apply(list).to_dict()
        
        cooccur_counts = {}
        for aspects_list in review_aspects.values():
            unique_aspects = list(set(aspects_list))
            for i in range(len(unique_aspects)):
                for j in range(i+1, len(unique_aspects)):
                    pair = tuple(sorted([unique_aspects[i], unique_aspects[j]]))
                    cooccur_counts[pair] = cooccur_counts.get(pair, 0) + 1
        
        if cooccur_counts:
            top_pairs = sorted(cooccur_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            context_parts.append("**Frequently Co-Mentioned Aspects:**")
            for (asp1, asp2), count in top_pairs:
                context_parts.append(f"- {asp1} + {asp2}: {count} reviews")
            
            context_parts.append("")
    
    # 4. Mixed Sentiment Cases
    if 'review_id' in aspect_level_df.columns and 'aspect_sentiment' in aspect_level_df.columns:
        review_sentiments = aspect_level_df.groupby('review_id')['aspect_sentiment'].apply(list)
        mixed_count = sum(1 for sentiments in review_sentiments if 'Positive' in sentiments and 'Negative' in sentiments)
        
        if mixed_count > 0:
            mixed_pct = (mixed_count / total_reviews * 100) if total_reviews > 0 else 0
            context_parts.append(f"**Mixed Sentiment Reviews:** {mixed_count} ({mixed_pct:.1f}%) - reviews with both positive and negative aspects")
            context_parts.append("")
    
    return "\n".join(context_parts)


def show_deep_dive_tab(df):
    """Deep dive tab for detailed aspect exploration"""
    from dashboard_components import extract_aspects_list
    
    st.markdown("### 🔍 Deep Dive: Aspect-Level Exploration")
    st.info("Drill down into specific aspects to see all reviews mentioning them with detailed sentiment analysis.")
    
    # ========== TAB 3 FILTERS (INDEPENDENT FROM OTHER TABS) ==========
    st.markdown("#### 🎛️ Deep Dive Filters")
    
    dd_col1, dd_col2, dd_col3 = st.columns(3)
    
    with dd_col1:
        # Aspect selector
        if 'aspects' in df.columns:
            all_aspects = []
            for aspects_val in df['aspects'].dropna():
                all_aspects.extend(extract_aspects_list(aspects_val))
            unique_aspects = sorted(set(all_aspects))
            
            dd_selected_aspect = st.selectbox(
                "Select Aspect to Analyze",
                options=['All'] + unique_aspects,
                help="Choose an aspect to see all reviews mentioning it",
                key="dd_aspect_selector"
            )
        else:
            dd_selected_aspect = 'All'
    
    with dd_col2:
        # Sentiment filter for deep dive
        if 'sentiment' in df.columns:
            dd_sentiments = list(df['sentiment'].unique())
            dd_selected_sentiments = st.multiselect(
                "Review Sentiment",
                options=dd_sentiments,
                default=dd_sentiments,
                help="Filter by overall review sentiment",
                key="dd_sentiment_filter"
            )
        else:
            dd_selected_sentiments = []
    
    with dd_col3:
        # Sort order
        dd_sort_by = st.selectbox(
            "Sort Reviews By",
            options=["Date (Newest)", "Date (Oldest)", "Confidence (High)", "Confidence (Low)"],
            key="dd_sort_selector"
        )
    
    # Apply filters
    dd_filtered_df = df.copy()
    
    if dd_selected_sentiments and 'sentiment' in dd_filtered_df.columns:
        dd_filtered_df = dd_filtered_df[dd_filtered_df['sentiment'].isin(dd_selected_sentiments)]
    
    # Filter by selected aspect
    if dd_selected_aspect != 'All' and 'aspects' in dd_filtered_df.columns:
        def contains_aspect(aspects_value):
            aspects = extract_aspects_list(aspects_value)
            return dd_selected_aspect in aspects
        
        dd_filtered_df = dd_filtered_df[dd_filtered_df['aspects'].apply(contains_aspect)]
    
    # Apply sorting
    if dd_sort_by == "Date (Newest)" and 'date' in dd_filtered_df.columns:
        dd_filtered_df = dd_filtered_df.sort_values('date', ascending=False)
    elif dd_sort_by == "Date (Oldest)" and 'date' in dd_filtered_df.columns:
        dd_filtered_df = dd_filtered_df.sort_values('date', ascending=True)
    elif dd_sort_by == "Confidence (High)" and 'confidence' in dd_filtered_df.columns:
        dd_filtered_df = dd_filtered_df.sort_values('confidence', ascending=False)
    elif dd_sort_by == "Confidence (Low)" and 'confidence' in dd_filtered_df.columns:
        dd_filtered_df = dd_filtered_df.sort_values('confidence', ascending=True)
    
    st.info(f"📊 Found **{len(dd_filtered_df)}** reviews" + (f" mentioning **{dd_selected_aspect}**" if dd_selected_aspect != 'All' else ""))
    
    if len(dd_filtered_df) == 0:
        st.warning("⚠️ No reviews match the selected criteria. Try adjusting your filters.")
        return
    
    # ========== ASPECT STATISTICS PANEL ==========
    if dd_selected_aspect != 'All':
        st.markdown("---")
        st.markdown(f"### 📊 Statistics for: **{dd_selected_aspect}**")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Total Mentions", len(dd_filtered_df))
        
        with stat_col2:
            if 'sentiment' in dd_filtered_df.columns:
                positive_pct = (dd_filtered_df['sentiment'] == 'Positive').sum() / len(dd_filtered_df) * 100
                st.metric("Positive %", f"{positive_pct:.1f}%")
        
        with stat_col3:
            if 'sentiment' in dd_filtered_df.columns:
                negative_pct = (dd_filtered_df['sentiment'] == 'Negative').sum() / len(dd_filtered_df) * 100
                st.metric("Negative %", f"{negative_pct:.1f}%")
        
        with stat_col4:
            if 'confidence' in dd_filtered_df.columns:
                avg_confidence = dd_filtered_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # ========== PAGINATED REVIEW DISPLAY ==========
    st.markdown("---")
    st.markdown("### 📝 Review Details")
    
    page_size = st.selectbox("Reviews per page", [10, 25, 50, 100], index=1, key="dd_page_size")
    
    total_pages = (len(dd_filtered_df) - 1) // page_size + 1
    page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, key="dd_page_number")
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(dd_filtered_df))
    
    st.info(f"Showing reviews {start_idx + 1} to {end_idx} of {len(dd_filtered_df)}")
    
    # Display reviews
    for idx, row in dd_filtered_df.iloc[start_idx:end_idx].iterrows():
        review_text = row.get('review', 'N/A')
        sentiment = row.get('sentiment', 'N/A')
        intent = row.get('intent', 'N/A')
        confidence = row.get('confidence', 0)
        aspects = extract_aspects_list(row.get('aspects', []))
        date = row.get('date', 'N/A')
        
        # Color code by sentiment
        if sentiment == 'Positive':
            bg_color = '#f0fdf4'
            border_color = '#22c55e'
        elif sentiment == 'Negative':
            bg_color = '#fef2f2'
            border_color = '#ef4444'
        else:
            bg_color = '#f0f9ff'
            border_color = '#3b82f6'
        
        # Highlight selected aspect
        display_text = review_text
        if dd_selected_aspect != 'All' and dd_selected_aspect in review_text:
            display_text = review_text.replace(
                dd_selected_aspect,
                f"<mark style='background-color: #fef08a; padding: 2px 4px; border-radius: 3px;'>{dd_selected_aspect}</mark>"
            )
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 15px; border-left: 4px solid {border_color}; 
                    border-radius: 4px; margin-bottom: 12px;'>
        <p style='margin:0; font-size: 1.05em;'>{display_text}</p>
        <hr style='margin: 10px 0; border: none; border-top: 1px solid #e5e7eb;'>
        <p style='margin:0; font-size: 0.9em; color: #4b5563;'>
        <strong>Sentiment:</strong> {sentiment} | 
        <strong>Intent:</strong> {intent} | 
        <strong>Confidence:</strong> {confidence:.2%} | 
        <strong>Date:</strong> {date}
        </p>
        <p style='margin:5px 0 0 0; font-size: 0.9em; color: #4b5563;'>
        <strong>All Aspects:</strong> {', '.join(aspects) if aspects else 'None detected'}
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========== EXPORT DEEP DIVE DATA ==========
    st.markdown("---")
    csv = dd_filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Deep Dive Results (CSV)",
        data=csv,
        file_name=f"deep_dive_{dd_selected_aspect}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def generate_rag_insights(aspect_level_df):
    """Generate RAG-style insights from aspect-level data"""
    insights = []
    
    if aspect_level_df.empty or 'aspect' not in aspect_level_df.columns:
        return [{
            'icon': 'ℹ️',
            'title': 'Insufficient Data',
            'message': 'Not enough aspect-level data to generate insights.',
            'bg_color': '#f0f9ff',
            'border_color': '#3b82f6',
            'text_color': '#1e40af'
        }]
    
    # Insight 1: Aspect co-occurrence patterns with sentiment
    if 'review_id' in aspect_level_df.columns and 'aspect_sentiment' in aspect_level_df.columns:
        # Find aspect pairs and their sentiment patterns
        review_groups = aspect_level_df.groupby('review_id')
        
        cooccur_sentiments = {}
        for review_id, group in review_groups:
            if len(group) > 1:
                aspects = group['aspect'].tolist()
                sentiments = group['aspect_sentiment'].tolist()
                
                for i in range(len(aspects)):
                    for j in range(i+1, len(aspects)):
                        pair = tuple(sorted([aspects[i], aspects[j]]))
                        if pair not in cooccur_sentiments:
                            cooccur_sentiments[pair] = {'positive': 0, 'negative': 0, 'neutral': 0}
                        
                        # Check if both are positive, both negative, or mixed
                        if sentiments[i] == 'Positive' and sentiments[j] == 'Positive':
                            cooccur_sentiments[pair]['positive'] += 1
                        elif sentiments[i] == 'Negative' and sentiments[j] == 'Negative':
                            cooccur_sentiments[pair]['negative'] += 1
                        else:
                            cooccur_sentiments[pair]['neutral'] += 1
        
        # Find most interesting pattern
        if cooccur_sentiments:
            # Find pair with highest positive correlation
            best_pair = max(cooccur_sentiments.items(), key=lambda x: x[1]['positive'])
            total = sum(best_pair[1].values())
            positive_pct = (best_pair[1]['positive'] / total * 100) if total > 0 else 0
            
            if positive_pct >= 70:
                insights.append({
                    'icon': '✅',
                    'title': 'Strong Positive Correlation',
                    'message': f"When **{best_pair[0][0]}** and **{best_pair[0][1]}** are mentioned together, they're both positive {positive_pct:.0f}% of the time ({best_pair[1]['positive']} out of {total} occurrences).",
                    'bg_color': '#f0fdf4',
                    'border_color': '#22c55e',
                    'text_color': '#15803d'
                })
            
            # Find pair with conflicting sentiments
            mixed_pair = max(cooccur_sentiments.items(), key=lambda x: x[1]['neutral'])
            mixed_total = sum(mixed_pair[1].values())
            mixed_pct = (mixed_pair[1]['neutral'] / mixed_total * 100) if mixed_total > 0 else 0
            
            if mixed_pct >= 50 and mixed_total >= 3:
                insights.append({
                    'icon': '⚠️',
                    'title': 'Mixed Sentiment Pattern',
                    'message': f"**{mixed_pair[0][0]}** and **{mixed_pair[0][1]}** often have conflicting sentiments when mentioned together ({mixed_pct:.0f}% of {mixed_total} cases).",
                    'bg_color': '#fef3c7',
                    'border_color': '#f59e0b',
                    'text_color': '#92400e'
                })
    
    # Insight 2: Aspect sentiment dominance
    aspect_sentiments = aspect_level_df.groupby(['aspect', 'aspect_sentiment']).size().reset_index(name='count')
    
    for aspect in aspect_level_df['aspect'].unique()[:5]:  # Top 5 aspects
        aspect_data = aspect_sentiments[aspect_sentiments['aspect'] == aspect]
        total = aspect_data['count'].sum()
        
        if total >= 5:  # Only if we have enough data
            positive_count = aspect_data[aspect_data['aspect_sentiment'] == 'Positive']['count'].sum()
            negative_count = aspect_data[aspect_data['aspect_sentiment'] == 'Negative']['count'].sum()
            
            positive_pct = (positive_count / total * 100) if total > 0 else 0
            negative_pct = (negative_count / total * 100) if total > 0 else 0
            
            if positive_pct >= 75:
                insights.append({
                    'icon': '🌟',
                    'title': f'Strength: {aspect}',
                    'message': f"**{aspect}** consistently receives positive feedback ({positive_pct:.0f}% positive across {total} mentions).",
                    'bg_color': '#f0fdf4',
                    'border_color': '#22c55e',
                    'text_color': '#15803d'
                })
            elif negative_pct >= 60:
                insights.append({
                    'icon': '🔴',
                    'title': f'Priority: {aspect}',
                    'message': f"**{aspect}** needs attention - {negative_pct:.0f}% negative sentiment across {total} mentions.",
                    'bg_color': '#fef2f2',
                    'border_color': '#ef4444',
                    'text_color': '#991b1b'
                })
    
    # Insight 3: Overall sentiment vs aspect sentiment discrepancy
    if 'overall_sentiment' in aspect_level_df.columns and 'aspect_sentiment' in aspect_level_df.columns:
        # Find cases where overall is positive but aspect is negative (and vice versa)
        discrepancy = aspect_level_df[
            ((aspect_level_df['overall_sentiment'] == 'Positive') & (aspect_level_df['aspect_sentiment'] == 'Negative')) |
            ((aspect_level_df['overall_sentiment'] == 'Negative') & (aspect_level_df['aspect_sentiment'] == 'Positive'))
        ]
        
        if len(discrepancy) > 0:
            discrepancy_pct = (len(discrepancy) / len(aspect_level_df) * 100)
            
            insights.append({
                'icon': '🔍',
                'title': 'Sentiment Nuance Detected',
                'message': f"{len(discrepancy)} aspect mentions ({discrepancy_pct:.1f}%) have sentiment different from overall review sentiment, indicating nuanced feedback.",
                'bg_color': '#ede9fe',
                'border_color': '#8b5cf6',
                'text_color': '#6b21a8'
            })
    
    # If no insights generated, add a default one
    if not insights:
        insights.append({
            'icon': '📊',
            'title': 'Analysis Complete',
            'message': f"Analyzed {len(aspect_level_df)} aspect mentions across {aspect_level_df['review_id'].nunique() if 'review_id' in aspect_level_df.columns else 0} reviews. Use the visualizations above to explore patterns.",
            'bg_color': '#f0f9ff',
            'border_color': '#3b82f6',
            'text_color': '#1e40af'
        })
    
    return insights[:6]  # Limit to 6 insights


def main():
    """Main application"""
    apply_custom_css()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=ABSA+AI", width='stretch')
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Analytics", "Admin"],
            icons=["house", "bar-chart", "lock"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About ABSA Insights")
        st.markdown("""
        **Powered by:**
        - 🤖 PyABSA for aspect extraction
        - 🌐 IndicTrans2 for translation
        - 🧠 Nvidia Nemotron for AI insights
        - 📊 Real-time analytics
        
        **Features:**
        - Multi-aspect sentiment analysis
        - Mixed sentiment detection
        - AI-powered recommendations
        - Multilingual support (Hindi + English)
        """)
        
        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit")
    
    # Page routing
    if selected == "Home":
        show_home_page()
    elif selected == "Analytics":
        show_analytics_page()
    elif selected == "Admin":
        show_admin_page()

if __name__ == "__main__":
    main()
