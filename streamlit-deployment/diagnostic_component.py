"""
Diagnostic component to check aspect data format in your dashboard
Add this temporarily to see what's happening with the aspects
"""

import streamlit as st
import pandas as pd
from dashboard_components import extract_aspects_list, get_top_aspects_by_frequency

def show_aspect_diagnostics(df: pd.DataFrame):
    """Show diagnostic information about aspects in the dataset"""
    
    st.markdown("---")
    st.markdown("### 🔍 Aspect Data Diagnostics")
    
    with st.expander("📊 Click to view aspect extraction details", expanded=False):
        
        if 'aspects' not in df.columns:
            st.error("No 'aspects' column found in the data!")
            return
        
        # Sample 5 random rows
        sample_df = df.sample(min(5, len(df)))
        
        st.markdown("#### Sample Aspect Data (5 random reviews)")
        
        for idx, row in sample_df.iterrows():
            aspects_raw = row.get('aspects')
            aspects_extracted = extract_aspects_list(aspects_raw)
            
            st.markdown(f"**Review #{idx}:**")
            st.code(f"Raw value: {repr(aspects_raw)}")
            st.code(f"Type: {type(aspects_raw)}")
            st.code(f"Extracted: {aspects_extracted}")
            st.code(f"Count: {len(aspects_extracted)}")
            st.markdown("---")
        
        # Statistics
        st.markdown("#### Overall Statistics")
        
        total_reviews = len(df)
        reviews_with_aspects = 0
        reviews_with_multiple = 0
        total_aspects = 0
        
        for idx, row in df.iterrows():
            aspects = extract_aspects_list(row.get('aspects'))
            if aspects:
                reviews_with_aspects += 1
                total_aspects += len(aspects)
                if len(aspects) >= 2:
                    reviews_with_multiple += 1
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", total_reviews)
        
        with col2:
            st.metric("Reviews with Aspects", reviews_with_aspects)
        
        with col3:
            st.metric("Reviews with 2+ Aspects", reviews_with_multiple)
        
        with col4:
            avg_aspects = total_aspects / total_reviews if total_reviews > 0 else 0
            st.metric("Avg Aspects/Review", f"{avg_aspects:.2f}")
        
        # Top aspects
        st.markdown("#### Top 15 Most Frequent Aspects")
        top_aspects = get_top_aspects_by_frequency(df, top_n=15)
        
        if top_aspects:
            # Count frequency for each
            aspect_counts = {}
            for aspects_value in df['aspects']:
                aspects = extract_aspects_list(aspects_value)
                for aspect in aspects:
                    aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
            
            for i, aspect in enumerate(top_aspects, 1):
                count = aspect_counts.get(aspect, 0)
                st.text(f"{i:2}. {aspect:30} ({count} occurrences)")
        else:
            st.warning("No aspects found!")
        
        # Co-occurrence preview
        st.markdown("#### Co-occurrence Sample")
        st.text("Showing first 10 reviews with 2+ aspects:")
        
        found = 0
        for idx, row in df.iterrows():
            if found >= 10:
                break
            
            aspects = extract_aspects_list(row.get('aspects'))
            if len(aspects) >= 2:
                found += 1
                review_text = row.get('review', 'N/A')[:80]
                st.text(f"{found}. {aspects} - '{review_text}...'")
        
        if found == 0:
            st.warning("⚠️ No reviews found with 2+ aspects! This explains the all-zero co-occurrence matrix.")
            st.info("Possible reasons:\n"
                   "- PyABSA is extracting only 1 aspect per review\n"
                   "- Aspect data format is incorrect\n"
                   "- Data processing issue in backend")
