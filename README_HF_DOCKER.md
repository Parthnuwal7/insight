---
title: Enhanced Sentiment Analysis Dashboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🚀 Enhanced Sentiment Analysis Dashboard

A production-ready Streamlit application for **Abstract Sentiment Analysis of Product Reviews** with advanced NLP capabilities, multilingual support, and enterprise-level analytics.

## ✨ Features

### 🔍 Core Analytics
- **Multilingual Support**: Automatic language detection and Hindi-to-English translation
- **Aspect-Based Sentiment Analysis (ABSA)**: Extract detailed aspects and their sentiments
- **Intent Classification**: Categorize reviews as praise, complaints, inquiries, etc.
- **Priority Scoring**: Intelligent scoring system for issue prioritization

### 📊 Advanced Visualizations
- **Network Analysis**: Aspect co-occurrence graphs showing relationships
- **Sankey Diagrams**: Intent-to-aspect flow analysis
- **Timeline Charts**: Sentiment trends over time with event annotations
- **KPI Dashboard**: Comprehensive metrics and performance indicators
- **Regional Analysis**: Language-wise sentiment distribution
- **Interactive Heatmaps**: Aspect-sentiment correlation matrices

### 🎯 Business Intelligence
- **Areas of Improvement**: AI-powered identification of problem areas
- **Strength Anchors**: Recognition of positive aspects to leverage
- **Alert System**: Automated sentiment spike detection
- **Impact Simulation**: What-if analysis for business decisions
- **Export Functionality**: PDF reports and Excel data export

## 📝 Data Format

Your CSV file should include these columns:
- `id`: Unique identifier for each review
- `reviews_title`: Title of the review
- `review`: The actual review text
- `date`: Review date (YYYY-MM-DD format)
- `user_id`: Identifier for the reviewer

## 🚀 Usage

### For End Users:
1. **Upload your CSV file** with review data (see format below)
2. **Process the data** using our advanced NLP pipeline (~2-3 minutes for 1000 reviews)
3. **Explore insights** through interactive visualizations
4. **Export results** as PDF reports or Excel files

### For Developers:
#### Docker Deployment (Hugging Face Spaces)
This app is optimized for Docker deployment with:
- `Dockerfile`: Production-ready container setup
- `requirements-docker.txt`: Optimized dependencies with version pinning
- `.dockerignore`: Efficient build context
- Health checks and proper port configuration (7860)

#### Local Development
```bash
git clone <your-repo>
cd insights
pip install -r requirements.txt
streamlit run app_enhanced.py
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit with interactive components
- **NLP**: pyABSA for aspect-based sentiment analysis
- **Translation**: Facebook M2M100 for multilingual support
- **Visualization**: Plotly for interactive charts and graphs
- **Network Analysis**: NetworkX for aspect relationship graphs

## 📊 Sample Output

The dashboard provides:
- Comprehensive sentiment analysis
- Aspect extraction and sentiment mapping
- Intent classification with confidence scores
- Interactive network graphs of aspect relationships
- Time-series analysis of sentiment trends
- Exportable business intelligence reports

---

**Status**: ✅ **Production Ready** - Enterprise-level sentiment analysis with advanced NLP capabilities.