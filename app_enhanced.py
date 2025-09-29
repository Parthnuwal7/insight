"""
Enhanced Streamlit application for Advanced Sentiment Analysis Dashboard.
Production-level application with advanced analytics, network graphs, and comprehensive insights.
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.data_processor import DataProcessor
from src.components.visualizations import (
    AdvancedVisualizationEngine, 
    KPIEngine, 
    FilterEngine, 
    ExportEngine
)
from src.utils.data_management import DataManager, TestDataGenerator, SessionManager, ConfigManager
from streamlit_option_menu import option_menu
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, Any
import networkx as nx

# Page configuration
st.set_page_config(
    page_title="Advanced Sentiment Analysis Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize managers and processors
@st.cache_resource
def initialize_components():
    """Initialize all application components."""
    data_manager = DataManager()
    session_manager = SessionManager()
    config_manager = ConfigManager()
    data_processor = DataProcessor()
    viz_engine = AdvancedVisualizationEngine()
    kpi_engine = KPIEngine()
    export_engine = ExportEngine()
    return (data_manager, session_manager, config_manager, 
            data_processor, viz_engine, kpi_engine, export_engine)

(data_manager, session_manager, config_manager, 
 data_processor, viz_engine, kpi_engine, export_engine) = initialize_components()

def display_advanced_kpi_header(processed_results: Dict[str, Any]):
    """Display the enhanced KPI header section."""
    if 'areas_of_improvement' in processed_results and 'strength_anchors' in processed_results:
        kpis = kpi_engine.calculate_kpis(
            processed_results['processed_data'],
            processed_results['areas_of_improvement'],
            processed_results['strength_anchors']
        )
        kpi_engine.create_kpi_header(kpis)
        return kpis
    return {}

def display_dual_panels(processed_results: Dict[str, Any]):
    """Display Areas of Improvement and Strength Anchors panels."""
    st.header("🎯 Strategic Analysis")
    viz_engine.create_dual_ranking_tables(
        processed_results.get('areas_of_improvement', pd.DataFrame()),
        processed_results.get('strength_anchors', pd.DataFrame())
    )

def display_network_analysis(processed_results: Dict[str, Any]):
    """Display aspect network graph."""
    st.header("🕸️ Aspect Relationship Network")
    if 'aspect_network' in processed_results:
        network_fig = viz_engine.create_aspect_network_graph(processed_results['aspect_network'])
        st.plotly_chart(network_fig, use_container_width=True)
    else:
        st.info("Network analysis not available.")

def display_sankey_flow(processed_results: Dict[str, Any]):
    """Display Intent-Aspect-Sentiment Sankey diagram."""
    st.header("🔄 Intent → Aspect → Sentiment Flow")
    sankey_fig = viz_engine.create_intent_aspect_sankey(processed_results['processed_data'])
    st.plotly_chart(sankey_fig, use_container_width=True)

def display_trend_analysis(processed_results: Dict[str, Any]):
    """Display enhanced trend analysis with annotations."""
    st.header("📈 Trend Analysis")
    
    # Sample annotations (in production, these would come from user input or external events)
    # Convert dates to datetime objects for proper plotting
    import pandas as pd
    sample_annotations = [
        {'date': '2025-09-28', 'text': 'Product Launch'},
        # Add more annotations as needed
    ]
    
    timeline_fig = viz_engine.create_enhanced_timeline_chart(
        processed_results['processed_data'], 
        annotations=sample_annotations
    )
    st.plotly_chart(timeline_fig, use_container_width=True)

def display_regional_analysis(processed_results: Dict[str, Any]):
    """Display regional/language analysis."""
    st.header("🌍 Regional & Language Insights")
    regional_fig = viz_engine.create_regional_language_analysis(processed_results['processed_data'])
    st.plotly_chart(regional_fig, use_container_width=True)

def display_alerts_and_simulation(processed_results: Dict[str, Any]):
    """Display alerts and impact simulation."""
    col1, col2 = st.columns(2)
    
    with col1:
        viz_engine.create_alert_dashboard(processed_results.get('sentiment_alerts', []))
    
    with col2:
        viz_engine.create_impact_simulation_tool(processed_results['processed_data'])

def display_summaries(processed_results: Dict[str, Any]):
    """Display macro and micro summaries."""
    st.header("📋 AI-Powered Insights")
    viz_engine.create_summary_sections(
        processed_results.get('macro_summary', {}),
        processed_results.get('micro_summaries', {})
    )

def display_export_options(processed_results: Dict[str, Any], kpis: Dict[str, Any]):
    """Display export functionality."""
    st.header("📄 Export Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download Excel Report", type="primary"):
            excel_data = export_engine.generate_excel_export(
                processed_results['processed_data'],
                processed_results.get('areas_of_improvement', pd.DataFrame()),
                processed_results.get('strength_anchors', pd.DataFrame())
            )
            st.download_button(
                label="📥 Download Excel File",
                data=excel_data,
                file_name=f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Download PDF Report", type="secondary"):
            pdf_data = export_engine.generate_pdf_report(
                processed_results['processed_data'],
                kpis,
                processed_results.get('areas_of_improvement', pd.DataFrame()),
                processed_results.get('strength_anchors', pd.DataFrame())
            )
            st.download_button(
                label="📥 Download PDF File",
                data=pdf_data,
                file_name=f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

def create_advanced_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Create enhanced filtering interface."""
    st.sidebar.header("🔍 Advanced Filters")
    
    filter_options = FilterEngine.get_filter_options(df)
    filters = {}
    
    # Date range filter
    if filter_options['date_range'][0] != filter_options['date_range'][1]:
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(filter_options['date_range'][0], filter_options['date_range'][1]),
            min_value=filter_options['date_range'][0],
            max_value=filter_options['date_range'][1]
        )
        filters['date_range'] = date_range
    
    # Sentiment filter
    sentiment_filter = st.sidebar.selectbox(
        "Filter by Sentiment",
        filter_options['sentiments']
    )
    if sentiment_filter != 'All':
        filters['sentiment'] = sentiment_filter
    
    # Intent filter
    intent_filter = st.sidebar.selectbox(
        "Filter by Intent",
        filter_options['intents']
    )
    if intent_filter != 'All':
        filters['intent'] = intent_filter
    
    # Language filter
    language_filter = st.sidebar.selectbox(
        "Filter by Language",
        filter_options['languages']
    )
    if language_filter != 'All':
        filters['language'] = language_filter
    
    # Aspect filter
    aspect_filter = st.sidebar.selectbox(
        "Filter by Aspect",
        filter_options['aspects']
    )
    if aspect_filter != 'All':
        filters['aspect'] = aspect_filter
    
    return filters

def home_page():
    """Enhanced home page with data upload and processing."""
    st.header("🏠 Welcome to Advanced Sentiment Analysis")
    
    # File upload section
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your CSV file with reviews",
        type=['csv'],
        help="CSV should contain columns: id, reviews_title, review, date, user_id"
    )
    
    # Sample data option
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 **CSV Format Required:** id, reviews_title, review, date, user_id")
    with col2:
        if st.button("🎲 Generate Test Data"):
            test_data = TestDataGenerator.generate_complex_test_data(rows=50)
            st.session_state.uploaded_data = test_data
            st.success("✅ Test data generated successfully!")
            st.rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            
            # Display basic info
            st.success(f"✅ File uploaded successfully! {len(df)} reviews loaded.")
            
            # Show preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return
    
    # Process data if available
    if 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing data with advanced analytics..."):
                try:
                    # Process the data with enhanced pipeline
                    processed_results = data_processor.process_uploaded_data(df)
                    
                    if 'error' in processed_results:
                        st.error(f"❌ Processing failed: {processed_results['error']}")
                        return
                    
                    # Store results in session state
                    st.session_state.processed_results = processed_results
                    
                    # Display success message with enhanced metrics
                    summary = processed_results['summary']
                    st.success("✅ **Analysis Complete!**")
                    
                    # Quick overview
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Reviews", summary['total_reviews'])
                    with col2:
                        st.metric("Problem Areas", summary['top_problem_areas'])
                    with col3:
                        st.metric("Strength Anchors", summary['top_strength_anchors'])
                    with col4:
                        st.metric("Active Alerts", summary['active_alerts'])
                    
                    st.info("🎯 Navigate to **Analytics** tab to explore detailed insights!")
                    
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    st.exception(e)

def analytics_page():
    """Enhanced analytics page with comprehensive visualizations."""
    st.header("📊 Advanced Analytics Dashboard")
    
    if 'processed_results' not in st.session_state:
        st.warning("⚠️ Please upload and process data first from the Home page.")
        return
    
    processed_results = st.session_state.processed_results
    df = processed_results['processed_data']
    
    # Create filters
    filters = create_advanced_filters(df)
    
    # Apply filters
    if filters:
        filtered_df = FilterEngine.apply_filters(df, filters)
        # Update processed_results with filtered data for visualizations
        processed_results_filtered = processed_results.copy()
        processed_results_filtered['processed_data'] = filtered_df
        st.info(f"🔍 Showing {len(filtered_df)} reviews (filtered from {len(df)} total)")
    else:
        processed_results_filtered = processed_results
        st.info(f"📊 Showing all {len(df)} reviews")
    
    # 1. KPI Header
    kpis = display_advanced_kpi_header(processed_results_filtered)
    
    st.divider()
    
    # 2. Dual Strategic Panels
    display_dual_panels(processed_results_filtered)
    
    st.divider()
    
    # 3. Network Analysis
    display_network_analysis(processed_results_filtered)
    
    st.divider()
    
    # 4. Sankey Flow Diagram
    display_sankey_flow(processed_results_filtered)
    
    st.divider()
    
    # 5. Trend Analysis
    display_trend_analysis(processed_results_filtered)
    
    st.divider()
    
    # 6. Regional Analysis
    display_regional_analysis(processed_results_filtered)
    
    st.divider()
    
    # 7. Alerts and Simulation
    display_alerts_and_simulation(processed_results_filtered)
    
    st.divider()
    
    # 8. AI-Powered Summaries
    display_summaries(processed_results_filtered)
    
    st.divider()
    
    # 9. Export Options
    display_export_options(processed_results_filtered, kpis)

def history_page():
    """Enhanced history page with advanced session management."""
    st.header("🕐 Analysis History")
    
    # Get all saved sessions
    saved_sessions = data_manager.get_all_sessions()
    
    if not saved_sessions:
        st.info("📝 No previous analysis sessions found.")
        return
    
    # Display sessions
    for session_id, session_data in saved_sessions.items():
        with st.expander(f"📊 Session: {session_data.get('timestamp', 'Unknown')} - {session_data.get('total_reviews', 0)} reviews"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Reviews:** {session_data.get('total_reviews', 0)}")
                st.write(f"**Languages:** {', '.join(session_data.get('languages', []))}")
                st.write(f"**Problem Areas:** {session_data.get('problem_areas', 0)}")
                st.write(f"**Strengths:** {session_data.get('strengths', 0)}")
            
            with col2:
                if st.button(f"🔄 Restore Session", key=f"restore_{session_id}"):
                    if data_manager.load_session(session_id):
                        st.success("✅ Session restored successfully!")
                        st.rerun()
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{session_id}", type="secondary"):
                    if data_manager.delete_session(session_id):
                        st.success("✅ Session deleted!")
                        st.rerun()

def documentation_page():
    """Enhanced documentation page."""
    st.header("📚 Documentation")
    
    # Enhanced documentation sections
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Getting Started", "📊 Features", "🔧 Technical", "🤖 AI Models"])
    
    with tab1:
        st.markdown("""
        ## 🚀 Getting Started
        
        Welcome to the **Advanced Sentiment Analysis Dashboard** - a production-ready application for comprehensive review analysis.
        
        ### Quick Start:
        1. **Upload Data**: Go to Home page and upload your CSV file or generate test data
        2. **Process**: Click "Start Analysis" to run the advanced processing pipeline
        3. **Explore**: Navigate to Analytics for comprehensive insights
        4. **Export**: Download PDF/Excel reports of your findings
        
        ### CSV Format Required:
        ```
        id,reviews_title,review,date,user_id
        review_001,Product Review,Great product with excellent quality,2025-09-28,user_123
        ```
        """)
    
    with tab2:
        st.markdown("""
        ## 📊 Advanced Features
        
        ### 🎯 Strategic Analysis
        - **Areas of Improvement**: Prioritized negative aspects with severity scoring
        - **Strength Anchors**: Top positive aspects for leveraging
        
        ### 🕸️ Network Analysis
        - **Aspect Co-occurrence**: Interactive network showing aspect relationships
        - **Sentiment Mapping**: Color-coded nodes showing sentiment distribution
        
        ### 🔄 Flow Analysis
        - **Intent → Aspect → Sentiment**: Sankey diagrams showing review flow
        - **Multi-dimensional filtering**: Complex filtering across all dimensions
        
        ### 📈 Trend Analysis
        - **Timeline Charts**: Sentiment trends over time
        - **Event Annotations**: Mark important dates and events
        - **Spike Detection**: Automated alerts for sentiment changes
        
        ### 🌍 Regional Insights
        - **Language Analysis**: Distribution and sentiment by language
        - **Multilingual Support**: Hindi-to-English translation
        
        ### 🤖 AI-Powered Insights
        - **Macro Summaries**: High-level executive insights
        - **Micro Analysis**: Detailed aspect-specific summaries
        - **What-If Simulation**: Impact analysis for improvements
        """)
    
    with tab3:
        st.markdown("""
        ## 🔧 Technical Architecture
        
        ### Data Processing Pipeline:
        1. **Data Validation**: CSV format and content validation
        2. **Language Detection**: Automatic language identification
        3. **Translation**: Hindi-to-English using M2M100
        4. **Intent Classification**: Enhanced with severity scoring
        5. **ABSA Processing**: Aspect-based sentiment analysis
        6. **Advanced Analytics**: Priority scoring and co-occurrence analysis
        
        ### Model Components:
        - **Translation**: Facebook M2M100 (418M parameters)
        - **ABSA**: pyABSA multilingual checkpoint
        - **Network Analysis**: NetworkX for graph computations
        - **Visualization**: Plotly for interactive charts
        
        ### Performance Features:
        - **Caching**: Streamlit caching for model loading
        - **Batch Processing**: Efficient handling of large datasets
        - **Progressive Loading**: Real-time progress indicators
        """)
    
    with tab4:
        st.markdown("""
        ## 🤖 AI Models & Algorithms
        
        ### Intent Classification with Severity:
        - **Complaint Severity**: High/Medium/Low severity detection
        - **Praise Levels**: Positive sentiment intensity analysis
        - **Multi-class Support**: Questions, suggestions, comparisons
        
        ### Aspect Analytics:
        - **Priority Scoring**: `negativity_ratio × frequency × (1 + severity_weight)`
        - **Strength Scoring**: `positivity_ratio × frequency × (1 + positivity_ratio × 2)`
        - **Co-occurrence Analysis**: Graph-based relationship mapping
        
        ### Sentiment Spike Detection:
        - **Rolling Window Analysis**: Week-over-week comparison
        - **Threshold Detection**: >50% increase with minimum volume
        - **Alert Severity**: Automated high/medium classification
        
        ### Summary Generation:
        - **Macro Insights**: Executive-level strategic summaries
        - **Micro Analysis**: Aspect-specific detailed insights
        - **Impact Simulation**: What-if scenario modeling
        """)

def main():
    """Enhanced main application function."""
    
    # Initialize session manager
    session_manager = SessionManager()
    
    # Enhanced CSS styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .filter-section {
        background-color: #f1f5f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    .success-banner {
        background: linear-gradient(90deg, #4ade80, #22c55e);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown('<h1 class="main-header">🚀 Advanced Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Enhanced navigation menu
    selected = option_menu(
        menu_title=None,
        options=["🏠 Home", "📊 Analytics", "🕐 History", "📚 Documentation"],
        icons=["house-fill", "graph-up", "clock-history", "book-fill"],
        menu_icon="rocket-takeoff",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#ffffff", "border-radius": "10px"},
            "icon": {"color": "#3b82f6", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "10px",
                "border-radius": "10px",
                "--hover-color": "#e2e8f0"
            },
            "nav-link-selected": {"background-color": "#3b82f6", "color": "white"},
        }
    )
    
    # Route to appropriate page
    if selected == "🏠 Home":
        home_page()
    elif selected == "📊 Analytics":
        analytics_page()
    elif selected == "🕐 History":
        history_page()
    elif selected == "📚 Documentation":
        documentation_page()

if __name__ == "__main__":
    main()