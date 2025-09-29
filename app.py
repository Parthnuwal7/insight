"""
Main Streamlit application for Sentiment Analysis Dashboard.
Entry point for the production-level sentiment analysis application.
"""

import streamlit as st
import pandas as pd
import sys
import os


# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.data_processor import DataProcessor
from src.components.visualizations import VisualizationEngine, FilterEngine
from src.utils.data_management import DataManager, TestDataGenerator, SessionManager, ConfigManager
from streamlit_option_menu import option_menu
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize managers
@st.cache_resource
def initialize_managers():
    """Initialize application managers."""
    data_manager = DataManager()
    session_manager = SessionManager()
    config_manager = ConfigManager()
    return data_manager, session_manager, config_manager

data_manager, session_manager, config_manager = initialize_managers()

# Initialize processors
@st.cache_resource
def initialize_processors():
    """Initialize data processing components."""
    data_processor = DataProcessor()
    viz_engine = VisualizationEngine()
    return data_processor, viz_engine

data_processor, viz_engine = initialize_processors()

def main():
    """Main application function."""
    
    # Initialize session manager first to ensure session state is set up
    session_manager = SessionManager()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .filter-section {
        background-color: #f1f5f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Analytics", "History", "Documentation"],
        icons=["house", "graph-up", "clock-history", "book"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#3b82f6", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#e2e8f0",
            },
            "nav-link-selected": {"background-color": "#3b82f6"},
        }
    )
    
    # Route to appropriate page
    if selected == "Home":
        home_page()
    elif selected == "Analytics":
        analytics_page()
    elif selected == "History":
        history_page()
    elif selected == "Documentation":
        documentation_page()

def home_page():
    """Home page with file upload and initial processing."""
    
    st.markdown("## 🏠 Welcome to the Sentiment Analysis Dashboard")
    st.markdown("Upload your CSV file to start analyzing product reviews with advanced sentiment and aspect extraction.")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📁 Upload Dataset")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with columns: id, reviews_title, review, date, user_id"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"File uploaded successfully! Found {len(df)} reviews.")
                
                # Show data preview
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Process data button
                if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    process_data(df, uploaded_file.name)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        # Generate test data
        st.markdown("#### 🧪 Generate Test Data")
        
        col_a, col_b = st.columns(2)
        with col_a:
            test_reviews = st.number_input("Number of reviews", min_value=10, max_value=500, value=100)
        with col_b:
            complexity = st.selectbox("Complexity", ["Simple", "Complex"])
        
        if st.button("Generate Test Dataset", use_container_width=True):
            generate_test_data(test_reviews, complexity)
        
        # Quick stats if data exists
        if session_manager.get_processed_data():
            st.markdown("#### 📊 Current Session")
            data = session_manager.get_processed_data()
            df_processed = data['processed_data']
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Reviews", len(df_processed))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            sentiment_dist = df_processed['overall_sentiment'].value_counts()
            st.metric("Most Common Sentiment", sentiment_dist.index[0], sentiment_dist.values[0])
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("View Analytics", use_container_width=True):
                st.switch_page("pages/analytics.py")

def process_data(df: pd.DataFrame, filename: str):
    """Process uploaded data through the complete pipeline."""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Processing data through pipeline...")
        progress_bar.progress(20)
        
        # Process data
        result = data_processor.process_uploaded_data(df)
        progress_bar.progress(80)
        
        if 'error' in result:
            st.error(f"Data validation failed: {', '.join(result['error'])}")
            return
        
        # Save processed data
        session_id = session_manager.get_session_id()
        data_path = data_manager.save_processed_data(result, session_id)
        progress_bar.progress(90)
        
        # Store in session
        session_manager.set_processed_data(result)
        progress_bar.progress(100)
        
        status_text.text("Processing complete!")
        
        # Show summary
        st.success("🎉 Data processed successfully!")
        
        summary = result['summary']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", summary['total_reviews'])
        with col2:
            st.metric("Languages", len(summary['languages_detected']))
        with col3:
            st.metric("Unique Intents", len(summary['intents_distribution']))
        with col4:
            st.metric("Sentiment Types", len(summary['sentiment_distribution']))
        
        # Show quick charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            sentiment_data = list(summary['sentiment_distribution'].items())
            if sentiment_data:
                fig = px.pie(
                    values=[item[1] for item in sentiment_data],
                    names=[item[0] for item in sentiment_data],
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Intent distribution
            intent_data = list(summary['intents_distribution'].items())
            if intent_data:
                fig = px.bar(
                    x=[item[0] for item in intent_data],
                    y=[item[1] for item in intent_data],
                    title="Intent Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Navigate to analytics
        st.info("✨ Go to the Analytics page to view detailed insights!")
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def generate_test_data(num_reviews: int, complexity: str):
    """Generate test dataset."""
    
    with st.spinner("Generating test data..."):
        if complexity == "Simple":
            test_df = TestDataGenerator.generate_sample_dataset(num_reviews)
        else:
            test_df = TestDataGenerator.generate_complex_reviews(num_reviews)
        
        st.success(f"Generated {len(test_df)} test reviews!")
        
        # Show preview
        st.markdown("#### Preview of Generated Data")
        st.dataframe(test_df.head(), use_container_width=True)
        
        # Download option
        csv = test_df.to_csv(index=False)
        st.download_button(
            label="Download Test Dataset",
            data=csv,
            file_name=f"test_reviews_{num_reviews}_{complexity.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Option to process immediately
        if st.button("Process This Test Data", type="primary", use_container_width=True):
            process_data(test_df, f"test_data_{complexity.lower()}")

def analytics_page():
    """Analytics dashboard page."""
    
    st.markdown("## 📈 Analytics Dashboard")
    
    # Check if data exists
    processed_data = session_manager.get_processed_data()
    if not processed_data:
        st.warning("⚠️ No data available. Please upload and process data first.")
        if st.button("Go to Home Page"):
            st.switch_page("app.py")
        return
    
    df_processed = processed_data['processed_data']
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")
    
    # Get filter options
    filter_options = FilterEngine.get_filter_options(df_processed)
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(filter_options['date_range'][0], filter_options['date_range'][1]),
        min_value=filter_options['date_range'][0],
        max_value=filter_options['date_range'][1]
    )
    
    # Other filters
    sentiment_filter = st.sidebar.selectbox("Sentiment", filter_options['sentiments'])
    intent_filter = st.sidebar.selectbox("Intent", filter_options['intents'])
    language_filter = st.sidebar.selectbox("Language", filter_options['languages'])
    aspect_filter = st.sidebar.selectbox("Aspect", filter_options['aspects'])
    
    # Apply filters
    filters = {
        'date_range': date_range,
        'sentiment': sentiment_filter,
        'intent': intent_filter,
        'language': language_filter,
        'aspect': aspect_filter
    }
    
    filtered_df = FilterEngine.apply_filters(df_processed, filters)
    
    # Store current filters
    session_manager.set_filters(filters)
    
    # Show filtered data info
    st.info(f"📊 Showing {len(filtered_df)} reviews (filtered from {len(df_processed)} total)")
    
    # Main dashboard
    if len(filtered_df) > 0:
        create_dashboard(filtered_df)
    else:
        st.warning("No data matches the current filters. Please adjust your filter criteria.")
    
    # Save dashboard button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💾 Save Dashboard", type="primary", use_container_width=True):
            save_dashboard_state(filtered_df, filters)

def create_dashboard(df: pd.DataFrame):
    """Create the main analytics dashboard."""
    
    # Summary metrics
    st.markdown("### 📊 Summary Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Reviews", len(df))
    
    with col2:
        avg_sentiment = df['overall_sentiment'].value_counts().index[0]
        st.metric("Dominant Sentiment", avg_sentiment)
    
    with col3:
        unique_aspects = set()
        for aspects in df['aspects']:
            if isinstance(aspects, list):
                unique_aspects.update(aspects)
        st.metric("Unique Aspects", len(unique_aspects))
    
    with col4:
        languages = len(df['detected_language'].unique())
        st.metric("Languages", languages)
    
    with col5:
        intents = len(df['intent'].unique())
        st.metric("Intent Types", intents)
    
    # Charts section
    st.markdown("### 📈 Visualizations")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeline chart
        timeline_fig = viz_engine.create_timeline_chart(df)
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    with col2:
        # Sentiment distribution
        sentiment_fig = viz_engine.create_sentiment_distribution(df)
        st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Intent distribution
        intent_fig = viz_engine.create_intent_distribution(df)
        st.plotly_chart(intent_fig, use_container_width=True)
    
    with col2:
        # Language distribution
        lang_fig = viz_engine.create_language_distribution(df)
        st.plotly_chart(lang_fig, use_container_width=True)
    
    # Third row - wider charts
    st.markdown("### 🔍 Detailed Analysis")
    
    # Aspect analysis
    col1, col2 = st.columns(2)
    
    with col1:
        aspect_freq_fig = viz_engine.create_aspect_frequency_chart(df)
        st.plotly_chart(aspect_freq_fig, use_container_width=True)
    
    with col2:
        heatmap_fig = viz_engine.create_aspect_sentiment_heatmap(df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Word cloud section
    st.markdown("### ☁️ Word Clouds")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (sentiment, col) in enumerate(zip(['Positive', 'Negative', 'Neutral'], [col1, col2, col3])):
        with col:
            st.markdown(f"#### {sentiment} Reviews")
            wordcloud_b64 = viz_engine.create_wordcloud(df, sentiment)
            if wordcloud_b64:
                st.markdown(
                    f'<img src="data:image/png;base64,{wordcloud_b64}" width="100%">',
                    unsafe_allow_html=True
                )
            else:
                st.info(f"No {sentiment.lower()} reviews found")
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    corr_fig = viz_engine.create_correlation_matrix(df)
    st.plotly_chart(corr_fig, use_container_width=True)
    
    # Sample reviews for debugging
    st.markdown("### 🔍 Sample Review Analysis")
    
    if len(df) > 0:
        sample_idx = st.selectbox("Select a review to analyze:", range(min(10, len(df))))
        sample_row = df.iloc[sample_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Review")
            st.write(f"**Title:** {sample_row['reviews_title']}")
            st.write(f"**Review:** {sample_row['review']}")
            st.write(f"**Date:** {sample_row['date']}")
            st.write(f"**User ID:** {sample_row['user_id']}")
        
        with col2:
            st.markdown("#### Analysis Results")
            st.write(f"**Detected Language:** {sample_row['detected_language']}")
            st.write(f"**Translated Review:** {sample_row['translated_review']}")
            st.write(f"**Intent:** {sample_row['intent']}")
            st.write(f"**Overall Sentiment:** {sample_row['overall_sentiment']}")
            st.write(f"**Extracted Aspects:** {', '.join(sample_row['aspects']) if sample_row['aspects'] else 'None'}")
            st.write(f"**Aspect Sentiments:** {', '.join(sample_row['aspect_sentiments']) if sample_row['aspect_sentiments'] else 'None'}")

def save_dashboard_state(df: pd.DataFrame, filters: Dict):
    """Save current dashboard state to history."""
    
    # Convert date objects to strings for JSON serialization
    serializable_filters = {}
    for key, value in filters.items():
        if key == 'date_range' and value:
            # Convert date objects to ISO format strings
            if isinstance(value, (list, tuple)) and len(value) == 2:
                serializable_filters[key] = [
                    value[0].isoformat() if hasattr(value[0], 'isoformat') else str(value[0]),
                    value[1].isoformat() if hasattr(value[1], 'isoformat') else str(value[1])
                ]
            else:
                serializable_filters[key] = str(value)
        else:
            serializable_filters[key] = value
    
    dashboard_config = {
        'filters_applied': serializable_filters,
        'total_reviews': len(df),
        'data_file': 'processed_data',
        'charts_generated': [
            'timeline_chart', 'sentiment_distribution', 'intent_distribution',
            'language_distribution', 'aspect_frequency', 'aspect_heatmap',
            'wordclouds', 'correlation_matrix'
        ],
        'timestamp': datetime.now().isoformat(),
        'session_id': session_manager.get_session_id()
    }
    
    history_id = data_manager.save_dashboard_state(
        session_manager.get_session_id(),
        dashboard_config
    )
    
    st.success(f"✅ Dashboard saved to history! ID: {history_id}")

def history_page():
    """History page to view saved dashboards."""
    
    st.markdown("## 🕐 Dashboard History")
    
    # Get history list
    history_list = data_manager.get_history_list()
    
    if not history_list:
        st.info("No saved dashboards found.")
        return
    
    st.markdown(f"Found {len(history_list)} saved dashboard(s)")
    
    # Display history
    for history_item in history_list:
        with st.expander(f"Dashboard saved on {history_item['timestamp'][:16]} - {history_item['total_reviews']} reviews"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Total Reviews:** {history_item['total_reviews']}")
                st.write(f"**Data File:** {history_item['data_file']}")
                st.write(f"**Timestamp:** {history_item['timestamp']}")
            
            with col2:
                if st.button(f"Load", key=f"load_{history_item['id']}"):
                    load_dashboard_history(history_item['id'])
            
            with col3:
                if st.button(f"Delete", key=f"delete_{history_item['id']}", type="secondary"):
                    if data_manager.delete_history_entry(history_item['id']):
                        st.success("Deleted successfully!")
                        st.rerun()

def load_dashboard_history(history_id: str):
    """Load a specific dashboard from history."""
    
    history_data = data_manager.load_dashboard_history(history_id)
    if history_data:
        st.success(f"Loaded dashboard: {history_id}")
        # You would implement loading the actual data here
        st.info("Dashboard loaded! (Implementation depends on data storage strategy)")
    else:
        st.error("Failed to load dashboard history")

def documentation_page():
    """Documentation page."""
    
    st.markdown("## 📚 Documentation")
    
    st.markdown("""
    ### 🎯 Overview
    
    This Sentiment Analysis Dashboard provides comprehensive analysis of product reviews using advanced NLP techniques:
    
    - **Multi-language Support**: Automatic Hindi-to-English translation using Facebook M2M100
    - **Aspect-Based Sentiment Analysis**: Extract aspects and their sentiments using pyABSA
    - **Intent Classification**: Classify review intents (complaint, praise, question, etc.)
    - **Interactive Visualizations**: Timeline charts, heatmaps, word clouds, and more
    - **Advanced Filtering**: Filter by sentiment, intent, language, aspects, and date range
    
    ### 📋 Required CSV Format
    
    Your CSV file must contain the following columns:
    
    | Column | Description | Example |
    |--------|-------------|---------|
    | `id` | Unique review identifier | review_001 |
    | `reviews_title` | Review title or summary | "Great Product" |
    | `review` | Full review text | "This product is amazing..." |
    | `date` | Review date | 2024-01-15 |
    | `user_id` | User identifier | user_123 |
    
    ### 🚀 Getting Started
    
    1. **Upload Data**: Go to Home page and upload your CSV file
    2. **Process Data**: Click "Start Processing" to run the analysis pipeline
    3. **View Analytics**: Navigate to Analytics page for detailed insights
    4. **Apply Filters**: Use sidebar filters to focus on specific data
    5. **Save Dashboard**: Save your current view to history
    
    ### 🔧 Features
    
    #### Translation
    - Automatic language detection
    - Hindi-to-English translation using M2M100
    - Preserves original text for comparison
    
    #### Sentiment Analysis
    - Aspect-based sentiment extraction
    - Overall sentiment classification
    - Confidence scores for predictions
    
    #### Intent Classification
    - Rule-based intent detection
    - Categories: complaint, praise, question, suggestion, comparison, neutral
    
    #### Visualizations
    - **Timeline Chart**: Sentiment trends over time
    - **Distribution Charts**: Sentiment, intent, and language distributions
    - **Heatmaps**: Aspect-sentiment correlations
    - **Word Clouds**: Visual representation of text content
    - **Frequency Charts**: Most common aspects and terms
    
    ### 📊 Understanding the Results
    
    #### Sentiment Scores
    - **Positive**: Generally favorable reviews
    - **Negative**: Critical or unfavorable reviews  
    - **Neutral**: Balanced or factual reviews
    
    #### Intent Categories
    - **Complaint**: Issues or problems reported
    - **Praise**: Positive feedback and compliments
    - **Question**: Inquiries or requests for information
    - **Suggestion**: Recommendations for improvement
    - **Comparison**: Comparisons with other products
    - **Neutral**: General observations or facts
    
    ### 🛠️ Troubleshooting
    
    #### Common Issues
    
    **File Upload Errors**
    - Check CSV format and column names
    - Ensure file size is under 100MB
    - Verify date format (YYYY-MM-DD preferred)
    
    **Processing Errors**
    - Large datasets may take longer to process
    - Check for empty review entries
    - Ensure sufficient memory for model loading
    
    **Visualization Issues**
    - Some charts may be empty if no data matches filters
    - Word clouds require sufficient text content
    - Heatmaps need extracted aspects to display
    
    ### 📈 Best Practices
    
    1. **Data Quality**: Clean your data before upload
    2. **File Size**: Start with smaller datasets for testing
    3. **Filters**: Use filters to focus on specific insights
    4. **Save States**: Save important dashboard configurations
    5. **Regular Cleanup**: Remove old data files periodically
    
    ### 🔍 Advanced Usage
    
    #### Custom Filtering
    - Combine multiple filters for targeted analysis
    - Use date ranges to analyze trends
    - Filter by specific aspects for detailed insights
    
    #### Data Export
    - Download processed data for further analysis
    - Export visualizations as images
    - Save dashboard configurations for reuse
    
    ### 📞 Support
    
    For technical issues or questions:
    - Check the troubleshooting section above
    - Review error messages for specific guidance
    - Ensure all dependencies are properly installed
    
    ### 🔄 Updates
    
    This application supports:
    - Model updates for improved accuracy
    - New visualization types
    - Additional language support
    - Enhanced filtering options
    """)

if __name__ == "__main__":
    main()