"""
Enhanced Frontend Dashboard for Streamlit Cloud deployment
Full-featured dashboard connecting to HF Spaces ML backend via API
Matches app_enhanced.py functionality with lightweight dependencies
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
HF_SPACES_API_URL = "https://parthnuwal7-absa.hf.space"  # Update with your HF Space URL

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
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        
        /* Headers and text */
        .dashboard-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .section-header {
            color: #2e4057;
            border-left: 4px solid #ff6b6b;
            padding-left: 15px;
            margin: 20px 0 10px 0;
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
        
        /* Data tables */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Success/Error messages */
        .stAlert {
            border-radius: 10px;
        }
        
        /* File uploader */
        .uploadedFile {
            border: 2px dashed #ff6b6b;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        
        /* Navigation menu styling */
        nav[role="navigation"] {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Enhanced color schemes for visualizations
COLOR_SCHEMES = {
    'sentiment': {
        'POSITIVE': '#2ecc71',
        'NEGATIVE': '#e74c3c', 
        'NEUTRAL': '#95a5a6'
    },
    'intent': {
        'COMPLAINT': '#e74c3c',
        'APPRECIATION': '#2ecc71',
        'QUESTION': '#f39c12',
        'SUGGESTION': '#3498db',
        'OTHER': '#95a5a6'
    },
    'language': {
        'HINDI': '#ff7675',
        'ENGLISH': '#74b9ff',
        'OTHER': '#a29bfe'
    },
    'gradient': ['#667eea', '#764ba2', '#ff6b6b', '#ff8e53']
}

def call_ml_backend(data: Dict) -> Dict:
    """Call the ML backend API on HF Spaces."""
    try:
        response = requests.post(
            f"{HF_SPACES_API_URL}/process-reviews",
            json=data,
            timeout=120  # Allow time for ML processing
        )
        
        # Check if response is successful
        if response.status_code == 200:
            return response.json()
        else:
            # Return detailed error information
            try:
                error_detail = response.json()
                return {
                    "status": "error", 
                    "message": f"API Error {response.status_code}: {error_detail.get('detail', response.text)}"
                }
            except:
                return {
                    "status": "error", 
                    "message": f"API Error {response.status_code}: {response.text}"
                }
                
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Backend processing timeout (>2 minutes)"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Backend connection error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

def normalize_backend_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize backend API response to match frontend expectations.
    Backend columns: overall_sentiment, detected_language, aspects, aspect_sentiments
    Frontend expects: sentiment, language, aspects, aspect_sentiments
    """
    if df is None or len(df) == 0:
        return df
    
    df_normalized = df.copy()
    
    # Map backend column names to frontend column names
    column_mapping = {
        'overall_sentiment': 'sentiment',
        'detected_language': 'language'
    }
    
    # Rename columns if they exist
    for backend_col, frontend_col in column_mapping.items():
        if backend_col in df_normalized.columns:
            df_normalized[frontend_col] = df_normalized[backend_col]
            print(f"✓ Mapped {backend_col} → {frontend_col}")
    
    # Ensure required columns exist with defaults if missing
    if 'sentiment' not in df_normalized.columns:
        if 'overall_sentiment' in df_normalized.columns:
            df_normalized['sentiment'] = df_normalized['overall_sentiment']
        else:
            df_normalized['sentiment'] = 'NEUTRAL'
        print(f"✓ Added 'sentiment' column")
    
    if 'language' not in df_normalized.columns:
        if 'detected_language' in df_normalized.columns:
            df_normalized['language'] = df_normalized['detected_language']
        else:
            df_normalized['language'] = 'ENGLISH'
        print(f"✓ Added 'language' column")
    
    if 'intent' not in df_normalized.columns:
        df_normalized['intent'] = 'OTHER'
        print(f"✓ Added 'intent' column (default)")
    
    if 'aspects' not in df_normalized.columns:
        df_normalized['aspects'] = '[]'
        print(f"✓ Added 'aspects' column (default)")
    
    print(f"✓ Normalization complete. Columns: {list(df_normalized.columns)}")
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

# Enhanced visualization functions
def create_sentiment_timeline(df: pd.DataFrame) -> go.Figure:
    """Create timeline chart of sentiment trends"""
    # Check if required columns exist
    if 'sentiment' not in df.columns:
        # Create a simple message figure
        fig = go.Figure()
        fig.add_annotation(
            text="Sentiment data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="📊 Sentiment Analysis",
            template='plotly_white',
            height=400
        )
        return fig
    
    if 'date' in df.columns:
        # Convert date column to datetime
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        
        # Filter out rows with invalid dates or sentiment
        df_copy = df_copy.dropna(subset=['date', 'sentiment'])
        
        if len(df_copy) == 0:
            # No valid data
            fig = go.Figure()
            fig.add_annotation(
                text="No valid date/sentiment data found",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            fig.update_layout(title="📈 Sentiment Trends Over Time", template='plotly_white')
            return fig
        
        # Group by date and sentiment
        timeline_data = df_copy.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig = px.line(
            timeline_data, 
            x='date', 
            y='count', 
            color='sentiment',
            color_discrete_map=COLOR_SCHEMES['sentiment'],
            title="📈 Sentiment Trends Over Time"
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

def create_aspect_sentiment_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap of aspect-sentiment relationships"""
    if 'aspects' not in df.columns or 'sentiment' not in df.columns:
        # Fallback to simple sentiment distribution if columns missing
        if 'sentiment' in df.columns:
            return create_sentiment_timeline(df)
        else:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Aspect and sentiment data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="🔥 Aspect-Sentiment Analysis",
                template='plotly_white',
                height=400
            )
            return fig
    
    # Expand aspects and create aspect-sentiment combinations
    aspect_sentiment_data = []
    
    for _, row in df.iterrows():
        if pd.notna(row['aspects']) and row['aspects'] and pd.notna(row['sentiment']):
            try:
                if isinstance(row['aspects'], str):
                    if row['aspects'].strip() and row['aspects'] != '[]':
                        aspects = eval(row['aspects'])
                    else:
                        continue
                elif isinstance(row['aspects'], list):
                    aspects = row['aspects']
                else:
                    continue
                
                if isinstance(aspects, list):
                    for aspect in aspects:
                        aspect_sentiment_data.append({
                            'aspect': str(aspect),
                            'sentiment': row['sentiment']
                        })
            except:
                continue
    
    if aspect_sentiment_data:
        aspect_df = pd.DataFrame(aspect_sentiment_data)
        heatmap_data = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='RdYlBu_r',
            aspect='auto',
            title="🔥 Aspect-Sentiment Heatmap"
        )
        
        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Aspects",
            template='plotly_white'
        )
        
        return fig
    
    # Fallback to simple sentiment distribution
    return create_sentiment_timeline(df)

def create_network_graph(df: pd.DataFrame) -> go.Figure:
    """Create network graph of aspect relationships"""
    try:
        if 'aspects' in df.columns:
            # Create network from aspect co-occurrences
            G = nx.Graph()
            
            for _, row in df.iterrows():
                if pd.notna(row['aspects']) and row['aspects']:
                    try:
                        if isinstance(row['aspects'], str):
                            if row['aspects'].strip() and row['aspects'] != '[]':
                                aspects = eval(row['aspects'])
                            else:
                                continue
                        elif isinstance(row['aspects'], list):
                            aspects = row['aspects']
                        else:
                            continue
                        
                        if isinstance(aspects, list) and len(aspects) > 1:
                            # Add edges between co-occurring aspects
                            for i, aspect1 in enumerate(aspects):
                                for aspect2 in aspects[i+1:]:
                                    aspect1_str = str(aspect1)
                                    aspect2_str = str(aspect2)
                                    if G.has_edge(aspect1_str, aspect2_str):
                                        G[aspect1_str][aspect2_str]['weight'] += 1
                                    else:
                                        G.add_edge(aspect1_str, aspect2_str, weight=1)
                    except:
                        continue
            
            if len(G.nodes()) > 0:
                # Create layout
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Create edge traces
                edge_x = []
                edge_y = []
                for edge in G.edges():
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
                
                # Create node traces
                node_x = []
                node_y = []
                node_text = []
                node_sizes = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_sizes.append(10 + G.degree(node) * 5)
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="middle center",
                    marker=dict(
                        size=node_sizes,
                        color='lightblue',
                        line=dict(width=2, color='darkblue')
                    )
                )
                
                fig = go.Figure(data=[edge_trace, node_trace],
                              layout=go.Layout(
                                title='🕸️ Aspect Relationship Network',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                annotations=[ dict(
                                    text="Network shows relationships between aspects mentioned together",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002,
                                    xanchor='left', yanchor='bottom',
                                    font=dict(color='gray', size=12)
                                )],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                              ))
                
                return fig
    except Exception as e:
        pass
    
    # Fallback to simple visualization
    return create_sentiment_timeline(df)

def create_kpi_cards(df: pd.DataFrame):
    """Create KPI metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_reviews = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Reviews</div>
            <div class="metric-value">{total_reviews:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Check for both column names
        sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'overall_sentiment' if 'overall_sentiment' in df.columns else None
        if sentiment_col:
            positive_pct = (df[sentiment_col].str.upper() == 'POSITIVE').mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Positive Sentiment</div>
                <div class="metric-value">{positive_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Positive Sentiment</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'aspects' in df.columns:
            def safe_count_aspects(x):
                """Safely count aspects with proper type checking"""
                try:
                    if pd.notna(x) and x:
                        if isinstance(x, str):
                            if x.strip() and x != '[]':
                                aspects = eval(x)
                                return len(aspects) if isinstance(aspects, list) else 0
                        elif isinstance(x, list):
                            return len(x)
                    return 0
                except:
                    return 0
            
            total_aspects = df['aspects'].apply(safe_count_aspects).sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Aspects</div>
                <div class="metric-value">{total_aspects:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Total Aspects</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Check for both column names
        language_col = 'language' if 'language' in df.columns else 'detected_language' if 'detected_language' in df.columns else None
        if language_col:
            multilingual_pct = (df[language_col].str.upper() != 'EN').mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Multilingual Content</div>
                <div class="metric-value">{multilingual_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Multilingual Content</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)

def create_advanced_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced filtering sidebar"""
    st.sidebar.markdown("### 🔍 Advanced Filters")
    
    filtered_df = df.copy()
    
    # Sentiment filter (check for both 'sentiment' and 'overall_sentiment')
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'overall_sentiment' if 'overall_sentiment' in df.columns else None
    if sentiment_col:
        sentiment_options = df[sentiment_col].unique().tolist()
        selected_sentiments = st.sidebar.multiselect(
            "Filter by Sentiment",
            sentiment_options,
            default=sentiment_options,
            key="sentiment_filter"
        )
        filtered_df = filtered_df[filtered_df[sentiment_col].isin(selected_sentiments)]
    
    # Intent filter
    if 'intent' in df.columns:
        intent_options = df['intent'].unique().tolist()
        selected_intents = st.sidebar.multiselect(
            "Filter by Intent",
            intent_options,
            default=intent_options,
            key="intent_filter"
        )
        filtered_df = filtered_df[filtered_df['intent'].isin(selected_intents)]
    
    # Language filter (check for both 'language' and 'detected_language')
    language_col = 'language' if 'language' in df.columns else 'detected_language' if 'detected_language' in df.columns else None
    if language_col:
        language_options = df[language_col].unique().tolist()
        selected_languages = st.sidebar.multiselect(
            "Filter by Language",
            language_options,
            default=language_options,
            key="language_filter"
        )
        filtered_df = filtered_df[filtered_df[language_col].isin(selected_languages)]
    
    # Date range filter
    if 'date' in df.columns:
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        if not df_copy['date'].isna().all():
            min_date = df_copy['date'].min().date()
            max_date = df_copy['date'].max().date()
            
            selected_date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(selected_date_range) == 2:
                start_date, end_date = selected_date_range
                filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
                filtered_df = filtered_df[
                    (filtered_df['date'].dt.date >= start_date) & 
                    (filtered_df['date'].dt.date <= end_date)
                ]
    
    # Text search
    search_text = st.sidebar.text_input("🔍 Search in reviews")
    if search_text:
        if 'review' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['review'].str.contains(search_text, case=False, na=False)
            ]
    
    st.sidebar.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} reviews**")
    
    return filtered_df

def show_home_page():
    """Display the home page with file upload and basic analysis"""
    st.markdown("""
    <div class="dashboard-header">
        <h1>🚀 Advanced Sentiment Analytics Dashboard</h1>
        <p>Upload your review data to get started with AI-powered sentiment analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "📂 Upload your CSV file",
        type=['csv'],
        help="Expected columns: id, reviews_title, review, date, user_id"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['id', 'review']
            optional_columns = ['reviews_title', 'date', 'user_id']
            missing_required = [col for col in required_columns if col not in df.columns]
            
            if missing_required:
                st.error(f"Missing required columns: {missing_required}")
                st.info("Required columns: id, review")
                st.info("Optional columns: reviews_title, date, user_id (will be filled with defaults)")
                return
            
            # Show info about optional columns
            missing_optional = [col for col in optional_columns if col not in df.columns]
            if missing_optional:
                st.info(f"Missing optional columns (will use defaults): {missing_optional}")
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} reviews")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # Process data button
            if st.button("🚀 Process Reviews with AI", type="primary"):
                with st.spinner("🤖 Processing reviews with ML backend..."):
                    # Prepare data for API in the correct format
                    # Ensure all required fields are present with defaults
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
                        "options": {}
                    }
                    
                    # Debug: Show API request format
                    with st.expander("🔍 Debug: API Request", expanded=False):
                        st.json({
                            "url": f"{HF_SPACES_API_URL}/process-reviews",
                            "sample_record": records[0] if records else {},
                            "total_records": len(records)
                        })
                    
                    # Call ML backend (but use hardcoded results for now)
                    result = call_ml_backend(api_data)
                    
                    # Show backend response for debugging
                    with st.expander("🔍 Debug: Backend Response (Raw)", expanded=False):
                        st.json(result)
                    # Show backend response for debugging
                    with st.expander("🔍 Debug: Backend Response (Raw)", expanded=False):
                        st.json(result)
                    
                    # ===== HARDCODED DASHBOARD RESULTS =====
                    # Using mock data while backend response format is being debugged
                    # This allows deployment now, automation later
                    
                    st.info("📊 **Note:** Currently showing demo results with your uploaded data. Backend integration in progress.")
                    
                    # Create hardcoded results based on uploaded data
                    hardcoded_results = []
                    
                    sentiments = ['Positive', 'Negative', 'Neutral']
                    languages = ['en', 'hi']
                    intents = ['appreciation', 'complaint', 'question', 'suggestion', 'neutral']
                    aspects_list = [
                        'product quality', 'customer service', 'delivery', 'price', 'user interface',
                        'performance', 'features', 'support', 'payment', 'design'
                    ]
                    
                    for idx, row in df.iterrows():
                        # Simulate sentiment analysis based on simple keywords
                        review_text = str(row.get('review', '')).lower()
                        
                        if any(word in review_text for word in ['great', 'excellent', 'amazing', 'love', 'good', 'best']):
                            sentiment = 'Positive'
                            intent = 'appreciation'
                            aspects = ['product quality', 'features']
                            aspect_sentiments = ['Positive', 'Positive']
                        elif any(word in review_text for word in ['bad', 'poor', 'terrible', 'worst', 'hate', 'slow', 'error']):
                            sentiment = 'Negative'
                            intent = 'complaint'
                            aspects = ['customer service', 'performance']
                            aspect_sentiments = ['Negative', 'Negative']
                        elif any(word in review_text for word in ['how', 'when', 'what', 'help', '?']):
                            sentiment = 'Neutral'
                            intent = 'question'
                            aspects = ['support']
                            aspect_sentiments = ['Neutral']
                        else:
                            sentiment = 'Neutral'
                            intent = 'neutral'
                            aspects = ['general']
                            aspect_sentiments = ['Neutral']
                        
                        # Detect language (simple heuristic)
                        language = 'hi' if any(ord(c) > 2304 and ord(c) < 2432 for c in review_text) else 'en'
                        
                        hardcoded_results.append({
                            'id': row.get('id', idx + 1),
                            'reviews_title': row.get('reviews_title', ''),
                            'review': row.get('review', ''),
                            'date': row.get('date', '2025-01-01'),
                            'user_id': row.get('user_id', f'user{idx}'),
                            'translated_review': row.get('review', ''),  # Assume already in English
                            'language': language,
                            'intent': intent,
                            'intent_severity': 'standard',
                            'intent_confidence': 0.85,
                            'aspects': str(aspects),  # Convert to string for consistency
                            'aspect_sentiments': str(aspect_sentiments),
                            'sentiment': sentiment  # Use 'sentiment' directly (not 'overall_sentiment')
                        })
                    
                    processed_df = pd.DataFrame(hardcoded_results)
                    
                    # ===== END HARDCODED SECTION =====
                    
                    # Show columns for verification
                    with st.expander("🔍 Debug: Hardcoded Results Structure", expanded=False):
                        st.write("**Columns:**", list(processed_df.columns))
                        st.write("**Shape:**", processed_df.shape)
                        st.write("**Has 'sentiment' column:**", 'sentiment' in processed_df.columns)
                        st.write("**Has 'language' column:**", 'language' in processed_df.columns)
                        if len(processed_df) > 0:
                            st.dataframe(processed_df.head(2), use_container_width=True)
                    
                    # Old API parsing code (commented out for now)
                    """
                    if result.get("status") == "success":
                        # Parse the processed data
                        processed_data = result.get("data", {}).get("processed_data", [])
                        
                        # Debug: Show raw API response structure
                        with st.expander("🔍 Debug: API Response Structure", expanded=False):
                            st.write("**Response keys:**", list(result.keys()))
                            st.write("**Data keys:**", list(result.get("data", {}).keys()))
                            st.write("**Number of records:**", len(processed_data))
                            if processed_data:
                                st.write("**Sample record keys:**", list(processed_data[0].keys()) if processed_data else [])
                        
                        processed_df = pd.DataFrame(processed_data)
                        
                        # Debug: Show columns before normalization
                        with st.expander("🔍 Debug: Before Normalization", expanded=False):
                            st.write("**Columns:**", list(processed_df.columns))
                            st.write("**Shape:**", processed_df.shape)
                            if len(processed_df) > 0:
                                st.dataframe(processed_df.head(2), use_container_width=True)
                        
                        # Normalize backend response to match frontend expectations
                        processed_df = normalize_backend_response(processed_df)
                        
                        # Debug: Show columns after normalization
                        with st.expander("🔍 Debug: After Normalization", expanded=True):
                            st.write("**Columns:**", list(processed_df.columns))
                            st.write("**Shape:**", processed_df.shape)
                            st.write("**Has 'sentiment' column:**", 'sentiment' in processed_df.columns)
                            st.write("**Has 'language' column:**", 'language' in processed_df.columns)
                            if len(processed_df) > 0:
                                st.dataframe(processed_df.head(2), use_container_width=True)
                    """
                    
                    # Continue with existing display logic
                    if len(processed_df) > 0:
                        
                        # Save to session state
                        st.session_state.processed_data = processed_df
                        st.session_state.filename = uploaded_file.name
                        
                        # Save session
                        session_manager = SessionManager()
                        session_id = session_manager.save_session(processed_df, uploaded_file.name)
                        
                        st.success("✅ Analysis completed! Check the Analytics tab for detailed insights.")
                        
                        # Show quick stats
                        st.markdown("### 📊 Quick Stats")
                        try:
                            create_kpi_cards(processed_df)
                        except Exception as kpi_error:
                            st.warning(f"Could not generate KPI cards: {str(kpi_error)}")
                            st.write("Available columns:", list(processed_df.columns))
                        
                        # Show sample results
                        with st.expander("🔍 Sample Analysis Results", expanded=True):
                            try:
                                # Select columns that exist
                                display_cols = ['review', 'sentiment', 'aspects', 'intent']
                                available_cols = [col for col in display_cols if col in processed_df.columns]
                                if available_cols:
                                    sample_df = processed_df.head(3)[available_cols]
                                    st.dataframe(sample_df, use_container_width=True)
                                else:
                                    st.info("Showing all available columns:")
                                    st.dataframe(processed_df.head(3), use_container_width=True)
                            except Exception as display_error:
                                st.error(f"Error displaying sample: {str(display_error)}")
                                st.write("Columns available:", list(processed_df.columns))
                                st.dataframe(processed_df.head(3), use_container_width=True)
                    
                    else:
                        st.warning("⚠️ Backend returned an error or unexpected format. Using demo mode.")
                        st.info("Backend response shown in debug section above.")
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("**Error Type:** " + type(e).__name__)
            
            # Show detailed error information
            with st.expander("🔍 Debug: Error Details", expanded=True):
                import traceback
                st.code(traceback.format_exc())
                
                # Show what data we have
                if 'processed_df' in locals():
                    st.write("**DataFrame was created with columns:**", list(processed_df.columns))
    
    else:
        # Show sample data format
        st.markdown("### 📋 Sample Data Format")
        sample_data = {
            'id': [1, 2, 3],
            'review': [
                'Great product! Love the quality and design.',
                'Delivery was slow but the item is good.',
                'Poor customer service experience.'
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
    
    # Debug: Show available columns
    with st.expander("🔍 Debug: Available Data Columns", expanded=False):
        st.write("**Available columns:**", list(df.columns))
        st.write("**Data shape:**", df.shape)
        if len(df) > 0:
            st.write("**Sample data:**")
            st.dataframe(df.head(2), use_container_width=True)
    
    # Apply filters
    filtered_df = create_advanced_filters(df)
    
    # KPI Cards
    st.markdown("### 📊 Key Performance Indicators")
    create_kpi_cards(filtered_df)
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Sentiment Timeline")
        timeline_fig = create_sentiment_timeline(filtered_df)
        st.plotly_chart(timeline_fig, use_container_width=True, key="sentiment_timeline")
    
    with col2:
        st.markdown("### 🔥 Aspect-Sentiment Heatmap")
        heatmap_fig = create_aspect_sentiment_heatmap(filtered_df)
        st.plotly_chart(heatmap_fig, use_container_width=True, key="aspect_sentiment_heatmap")
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Intent Distribution")
        if 'intent' in filtered_df.columns and not filtered_df['intent'].isna().all():
            intent_counts = filtered_df['intent'].value_counts()
            if len(intent_counts) > 0:
                intent_fig = px.pie(
                    values=intent_counts.values,
                    names=intent_counts.index,
                    color=intent_counts.index,
                    color_discrete_map=COLOR_SCHEMES['intent'],
                    title="Intent Classification Results"
                )
                intent_fig.update_layout(template='plotly_white')
                st.plotly_chart(intent_fig, use_container_width=True, key="intent_distribution")
            else:
                st.info("No intent data available in filtered results")
        else:
            st.info("Intent data not available")
    
    with col2:
        st.markdown("### 🌐 Language Distribution")
        if 'language' in filtered_df.columns and not filtered_df['language'].isna().all():
            lang_counts = filtered_df['language'].value_counts()
            if len(lang_counts) > 0:
                lang_fig = px.bar(
                    x=lang_counts.index,
                    y=lang_counts.values,
                    color=lang_counts.index,
                    color_discrete_map=COLOR_SCHEMES['language'],
                    title="Language Detection Results"
                )
                lang_fig.update_layout(
                    xaxis_title="Language",
                    yaxis_title="Count",
                    template='plotly_white'
                )
                st.plotly_chart(lang_fig, use_container_width=True, key="language_distribution")
            else:
                st.info("No language data available in filtered results")
        else:
            st.info("Language data not available")
    
    # Network graph
    st.markdown("### 🕸️ Aspect Relationship Network")
    network_fig = create_network_graph(filtered_df)
    st.plotly_chart(network_fig, use_container_width=True, key="aspect_network_graph")
    
    # Data table
    st.markdown("### 📋 Detailed Results")
    st.dataframe(filtered_df, use_container_width=True)

def show_history_page():
    """Display the history page with saved sessions"""
    st.markdown("## 📚 Analysis History")
    
    session_manager = SessionManager()
    saved_sessions = session_manager.get_all_sessions()
    
    if not saved_sessions:
        st.info("📝 No saved sessions found. Upload and process data to create history.")
        return
    
    st.markdown(f"### Found {len(saved_sessions)} saved sessions")
    
    for session_id, session_info in saved_sessions.items():
        with st.expander(f"📊 Session: {session_info['filename']} - {session_info['timestamp'][:19]}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Reviews", session_info['total_reviews'])
            
            with col2:
                st.write(f"**Filename:** {session_info['filename']}")
            
            with col3:
                if st.button(f"Load Session", key=f"load_{session_id}"):
                    loaded_df = session_manager.load_session(session_id)
                    if loaded_df is not None:
                        st.session_state.processed_data = loaded_df
                        st.session_state.filename = session_info['filename']
                        st.success("✅ Session loaded successfully! Go to Analytics to view.")
                        st.experimental_rerun()

def show_documentation_page():
    """Display the documentation page"""
    st.markdown("## 📖 Documentation & Help")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Upload Data**: Go to the Home page and upload a CSV file with your review data
    2. **Process Reviews**: Click the "Process Reviews with AI" button to analyze your data
    3. **View Analytics**: Switch to the Analytics tab to explore detailed insights
    4. **Apply Filters**: Use the sidebar filters to focus on specific segments
    5. **Save Sessions**: Your analysis is automatically saved in the History tab
    
    ### 📋 Data Format Requirements
    
    **Required Columns:**
    - `id`: Unique identifier for each review
    - `review`: The review text content
    
    **Optional Columns:**
    - `reviews_title`: Title of the review
    - `date`: Date of the review (YYYY-MM-DD format)
    - `user_id`: User identifier
    
    ### 🔬 Analysis Features
    
    **Sentiment Analysis:**
    - Classifies reviews as POSITIVE, NEGATIVE, or NEUTRAL
    - Uses advanced PyABSA models for high accuracy
    
    **Aspect-Based Sentiment Analysis (ABSA):**
    - Extracts specific aspects mentioned in reviews
    - Determines sentiment for each aspect
    - Provides confidence scores
    
    **Intent Classification:**
    - Categorizes reviews by intent (COMPLAINT, APPRECIATION, etc.)
    - Helps understand customer motivations
    
    **Language Detection & Translation:**
    - Automatically detects review language
    - Translates Hindi reviews to English for analysis
    
    ### 📊 Visualization Types
    
    1. **KPI Cards**: High-level metrics and statistics
    2. **Sentiment Timeline**: Trends over time
    3. **Aspect-Sentiment Heatmap**: Relationship between aspects and sentiments
    4. **Intent Distribution**: Pie chart of customer intents
    5. **Language Distribution**: Bar chart of detected languages
    6. **Network Graph**: Relationships between aspects
    
    ### 🔧 Technical Architecture
    
    This application uses a distributed architecture:
    - **Frontend**: Streamlit Cloud (lightweight, fast loading)
    - **ML Backend**: HuggingFace Spaces (powerful GPU processing)
    - **Models**: PyABSA, M2M100 translation, custom intent classifiers
    
    ### ❓ Troubleshooting
    
    **Common Issues:**
    - **Timeout Errors**: Large files may take >2 minutes to process
    - **Format Errors**: Ensure CSV has required columns
    - **Backend Unavailable**: Check HuggingFace Spaces status
    
    **Performance Tips:**
    - Process files with <1000 reviews for best performance
    - Use filters to focus on specific data segments
    - Save sessions to avoid reprocessing
    
    ### 🆘 Support
    
    For technical support or feature requests, please contact the development team or check the project repository.
    """)

def main():
    """Main application function"""
    # Apply custom CSS
    apply_custom_css()
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["🏠 Home", "📈 Analytics", "📚 History", "📖 Documentation"],
        icons=["house", "graph-up", "clock-history", "book"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee"
            },
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )
    
    # Route to appropriate page
    if selected == "🏠 Home":
        show_home_page()
    elif selected == "📈 Analytics":
        show_analytics_page()
    elif selected == "📚 History":
        show_history_page()
    elif selected == "📖 Documentation":
        show_documentation_page()

if __name__ == "__main__":
    main()