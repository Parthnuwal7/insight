# Copilot Instructions for AI Coding Agents

This is a production-level Streamlit application for **Abstract Sentiment Analysis of Product Reviews** with advanced NLP capabilities.

## Project Architecture

### Core Components
- **`app.py`**: Main Streamlit application with navigation and page routing
- **`src/utils/data_processor.py`**: Complete data processing pipeline (translation, ABSA, intent classification)
- **`src/components/visualizations.py`**: Interactive dashboard visualizations using Plotly
- **`src/utils/data_management.py`**: Data persistence, history tracking, and test data generation

### Key Technologies
- **pyABSA**: Aspect-based sentiment analysis with multilingual support
- **Facebook M2M100**: Hindi-to-English translation
- **Streamlit**: Production web interface with option menu navigation
- **Plotly**: Interactive visualizations and charts

## Data Flow Architecture
1. **Upload**: CSV with columns: `id, reviews_title, review, date, user_id`
2. **Processing Pipeline**: Language detection → Translation → Intent classification → ABSA extraction
3. **Analytics**: Timeline charts, heatmaps, word clouds, correlation analysis
4. **Filtering**: Multi-dimensional filtering (sentiment, intent, language, aspects, date range)
5. **Persistence**: Dashboard state saving and history management

## Development Patterns

### Model Loading
- Use `@st.cache_resource` for expensive model initialization (M2M100, pyABSA)
- Models are loaded once per session and shared across components
- Error handling for model loading failures with user-friendly messages

### Data Processing
- Validation before processing with detailed error reporting
- Progress indicators for long-running operations
- Batch processing for translation and ABSA tasks

### Session Management
- `SessionManager` class handles state persistence across page navigation
- Data stored in `st.session_state` with proper cleanup
- Unique session IDs for tracking and data organization

### Visualization Engine
- Modular chart creation with consistent color schemes
- Responsive design using Streamlit columns and containers
- Base64 encoding for complex visualizations (word clouds)

## Critical Dependencies
```
streamlit>=1.28.0
pyabsa>=2.4.0
transformers>=4.30.0
plotly>=5.15.0
streamlit-option-menu>=0.3.6
wordcloud>=1.9.0
```

## Directory Structure
```
src/
├── components/
│   ├── visualizations.py      # Chart and plot generation
│   └── __init__.py
├── utils/
│   ├── data_processor.py      # Main processing pipeline
│   ├── data_management.py     # Storage and utilities
│   └── __init__.py
└── __init__.py
data/
├── uploads/                   # User uploaded files
├── processed/                 # Processed data cache
└── history/                   # Dashboard histories
```

## Key Implementation Notes

### ABSA Integration
- Uses multilingual checkpoint from pyABSA
- Extracts aspects, sentiments, confidence scores, and IOB tags
- Results processed into structured format for visualization

### Translation Service
- Automatic language detection using `langdetect`
- M2M100 model for Hindi-to-English translation
- Preserves original text alongside translations

### Performance Considerations
- Model caching to prevent reloading
- Batch processing for multiple reviews
- Progress indicators for user feedback during processing

### Error Handling
- Comprehensive validation for CSV format and content
- Graceful degradation when models fail to load
- User-friendly error messages with actionable guidance

## Debugging and Monitoring
- Sample review analysis section shows processing details
- Logging throughout the pipeline for troubleshooting
- Data preview and validation feedback for uploads

## Future Extension Points
- Additional translation language pairs
- Custom intent classification models
- Enhanced aspect extraction rules
- Real-time data processing capabilities

---
_This architecture supports scalable sentiment analysis with production-ready error handling and user experience._
