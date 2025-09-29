# � Enhanced Sentiment Analysis Dashboard

A production-ready Streamlit application for **Abstract Sentiment Analysis of Product Reviews** with advanced NLP capabilities, multilingual support, and enterprise-level analytics including network analysis, Sankey diagrams, AI summaries, and comprehensive business intelligence features.

## 🌟 Features

- **🌍 Multi-language Support**: Automatic Hindi-to-English translation using Facebook M2M100
- **🎯 Aspect-Based Sentiment Analysis**: Extract aspects and their sentiments using pyABSA
- **🧠 Intent Classification**: Classify review intents (complaint, praise, question, etc.)
- **📈 Interactive Visualizations**: Timeline charts, heatmaps, word clouds, correlation analysis
- **🔍 Advanced Filtering**: Filter by sentiment, intent, language, aspects, and date range
- **💾 Dashboard History**: Save and reload dashboard configurations
- **🧪 Test Data Generation**: Generate synthetic datasets for testing
- **📚 Comprehensive Documentation**: Built-in help and guidance

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (for ML models)
- Internet connection (for initial model downloads)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd insights
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Start by uploading a CSV file or generating test data

## 📋 Data Format

Your CSV file must contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Unique review identifier | review_001 |
| `reviews_title` | Review title | "Great Product" |
| `review` | Full review text | "This product is amazing..." |
| `date` | Review date | 2024-01-15 |
| `user_id` | User identifier | user_123 |

## 🏗️ Architecture

### Project Structure
```
insights/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── src/
│   ├── components/
│   │   ├── visualizations.py  # Chart and plot generation
│   │   └── __init__.py
│   ├── utils/
│   │   ├── data_processor.py  # Main processing pipeline
│   │   ├── data_management.py # Storage and utilities
│   │   └── __init__.py
│   └── __init__.py
├── data/
│   ├── uploads/               # User uploaded files
│   ├── processed/             # Processed data cache
│   └── history/               # Dashboard histories
└── .github/
    └── copilot-instructions.md # AI agent guidance
```

### Data Processing Pipeline

1. **Validation**: CSV format and content validation
2. **Language Detection**: Automatic language identification
3. **Translation**: Hindi-to-English using M2M100
4. **Intent Classification**: Rule-based intent detection
5. **ABSA**: Aspect extraction and sentiment analysis using pyABSA
6. **Visualization**: Interactive dashboard generation

## 🔧 Core Components

### Data Processor (`src/utils/data_processor.py`)
- **DataValidator**: Validates CSV format and content
- **TranslationService**: M2M100-based Hindi translation
- **ABSAProcessor**: pyABSA integration for aspect extraction
- **IntentClassifier**: Rule-based intent classification
- **DataProcessor**: Main pipeline coordinator

### Visualization Engine (`src/components/visualizations.py`)
- **Timeline Charts**: Sentiment trends over time
- **Distribution Charts**: Sentiment, intent, language breakdowns
- **Heatmaps**: Aspect-sentiment correlations
- **Word Clouds**: Visual text representation
- **Correlation Matrix**: Feature relationship analysis

### Data Management (`src/utils/data_management.py`)
- **DataManager**: File storage and history tracking
- **TestDataGenerator**: Synthetic dataset creation
- **SessionManager**: State management across pages
- **ConfigManager**: Application configuration

## 📊 Dashboard Features

### Main Analytics
- **Summary Metrics**: Key statistics and counts
- **Timeline Analysis**: Sentiment trends over time
- **Distribution Charts**: Breakdown by sentiment, intent, language
- **Aspect Analysis**: Most frequent aspects and their sentiments
- **Word Clouds**: Visual representation by sentiment
- **Correlation Analysis**: Feature relationships

### Advanced Filtering
- **Date Range**: Filter by specific time periods
- **Sentiment**: Focus on positive, negative, or neutral reviews
- **Intent**: Filter by complaint, praise, question, etc.
- **Language**: Filter by detected languages
- **Aspects**: Focus on specific product aspects

### Sample Analysis
- **Debug View**: Detailed processing results for individual reviews
- **Processing Logs**: Step-by-step analysis breakdown
- **Confidence Scores**: Model prediction confidence

## 🛠️ Technical Details

### Model Integration

**pyABSA Configuration**:
```python
from pyabsa import ATEPCCheckpointManager

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
    checkpoint='multilingual',
    auto_device=True,
    task_code='ATEPC'
)
```

**M2M100 Translation**:
```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
```

### Performance Optimizations
- **Model Caching**: `@st.cache_resource` for expensive model loading
- **Batch Processing**: Efficient handling of multiple reviews
- **Progress Indicators**: User feedback during processing
- **Error Handling**: Graceful degradation and user guidance

### Data Persistence
- **Session State**: Maintains data across page navigation
- **File Storage**: Processed data and dashboard configurations
- **History Tracking**: Save and reload dashboard states

## 🎯 Usage Examples

### Basic Workflow
1. **Upload Data**: Use the file uploader on the Home page
2. **Process**: Click "Start Processing" to run the analysis pipeline
3. **Analyze**: Navigate to Analytics page for detailed insights
4. **Filter**: Use sidebar filters to focus on specific data
5. **Save**: Save dashboard configuration to history

### Generate Test Data
1. Go to Home page
2. Select number of reviews and complexity
3. Click "Generate Test Dataset"
4. Download or process the generated data immediately

### Advanced Analysis
1. Apply multiple filters for targeted insights
2. Use correlation matrix to identify relationships
3. Examine word clouds for content themes
4. Review sample analysis for debugging

## 🔍 Troubleshooting

### Common Issues

**Model Loading Errors**:
- Ensure sufficient RAM (4GB+)
- Check internet connection for downloads
- Restart application if models fail to load

**File Upload Problems**:
- Verify CSV format and column names
- Check file size limits (100MB max)
- Ensure proper date formatting

**Processing Failures**:
- Check for empty review entries
- Verify data quality and format
- Monitor system resources during processing

### Performance Tips
- Start with smaller datasets for testing
- Use filters to reduce processing load
- Clear browser cache if experiencing issues
- Monitor memory usage with large datasets

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Make changes and test thoroughly
5. Submit pull request with detailed description

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Include docstrings for functions and classes
- Add error handling and logging

## 📈 Future Enhancements

- **Real-time Processing**: Live data ingestion and analysis
- **Custom Models**: Train domain-specific sentiment models
- **API Integration**: REST API for programmatic access
- **Advanced Visualizations**: 3D plots and interactive maps
- **Multi-language Support**: Additional translation pairs
- **Export Features**: PDF reports and data export options

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **pyABSA**: Aspect-based sentiment analysis framework
- **Hugging Face**: Transformers and model hosting
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review error messages for guidance
- Ensure all dependencies are properly installed
- Check system requirements and resources

---

**Built with ❤️ for comprehensive sentiment analysis**