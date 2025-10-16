# 📊 Advanced Sentiment Analytics Dashboard

A lightweight frontend for sentiment analysis powered by PyABSA backend.

## 🏗️ Architecture

- **Frontend**: Streamlit Cloud (this repository)
- **Backend**: HuggingFace Spaces with PyABSA + FastAPI
- **Communication**: REST API integration

## 🚀 Features

### 📱 Multi-Page Dashboard
- **🏠 Home**: File upload and processing
- **📈 Analytics**: Advanced visualizations and KPI dashboard  
- **📚 History**: Session management and data persistence
- **📖 Documentation**: Complete user guide

### 📊 Visualizations
- Sentiment timeline charts
- Aspect-sentiment heatmaps
- Intent distribution charts
- Language distribution analysis
- Aspect relationship networks
- Interactive KPI cards

### 🔍 Advanced Features
- Multi-dimensional filtering
- Real-time data processing
- Session state management
- Responsive design
- Error handling and fallbacks

## 🛠️ Technology Stack

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **NetworkX**: Network graph analysis
- **Pandas**: Data manipulation
- **Requests**: API communication

## 📦 Dependencies

All dependencies are lightweight UI libraries (~60MB total):
- No ML models or heavy libraries
- Fast loading and deployment
- Perfect for Streamlit Cloud free tier

## 🔗 Backend Integration

Connects to HuggingFace Spaces backend for:
- PyABSA sentiment analysis
- M2M100 translation
- Intent classification
- Aspect extraction

## 🚀 Deployment

This app is designed for Streamlit Cloud deployment:

1. Connect your GitHub repository to Streamlit Cloud
2. Set main file path: `app.py`
3. Deploy automatically from main branch

## 📝 Usage

1. **Upload**: CSV file with review data
2. **Process**: AI analysis via backend API
3. **Analyze**: Explore insights in Analytics tab
4. **Save**: Sessions automatically saved in History

## 🔧 Configuration

Backend API endpoint configured in `app.py`:
```python
HF_SPACES_API_URL = "https://parthnuwal7-absa.hf.space"
```

## 📋 Data Format

**Required columns:**
- `id`: Unique identifier
- `review`: Review text content

**Optional columns:**
- `reviews_title`: Review title
- `date`: Review date
- `user_id`: User identifier

## 🆘 Support

For issues or questions, please check the Documentation tab in the application or contact the development team.