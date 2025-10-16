# 🚀 Deployment Checklist

## ✅ Pre-Deployment Verification

### 1. Backend API Configuration
- [ ] Backend deployed on HuggingFace Spaces: `https://parthnuwal7-absa.hf.space`
- [ ] API endpoint `/process-reviews` is accessible
- [ ] Test API with sample data returns expected columns

### 2. Column Mapping Verified
- [ ] Backend returns: `overall_sentiment`, `detected_language`, `aspects`
- [ ] Frontend maps to: `sentiment`, `language`, `aspects`
- [ ] All 13 expected columns are handled

### 3. Data Normalization
- [ ] `normalize_backend_response()` function implemented
- [ ] Column mapping tested with sample data
- [ ] Default values added for missing columns

### 4. Visualization Compatibility
- [ ] Sentiment charts use `sentiment` column
- [ ] Language charts handle `en`, `hi` codes (not full names)
- [ ] KPI cards calculate correctly with new columns
- [ ] Filters detect both old and new column names

### 5. Error Handling
- [ ] Missing columns show informative messages
- [ ] Empty data displays gracefully
- [ ] Debug section shows available columns
- [ ] All plotly charts have unique keys

## 📦 Files to Deploy

### Required Files:
```
streamlit-deployment/
├── app.py                          # Main application ✅
├── requirements.txt                # Dependencies ✅
├── .streamlit/config.toml          # Configuration ✅
├── README.md                       # Documentation ✅
└── .gitignore                      # Git ignore ✅
```

### Optional Files (for reference):
```
├── BACKEND_INTEGRATION.md          # Integration guide
└── test_backend_integration.py     # Test script
```

## 🔧 Configuration Updates

### API Endpoint (app.py line ~37)
```python
HF_SPACES_API_URL = "https://parthnuwal7-absa.hf.space"
```

### Column Mappings (app.py ~175)
```python
column_mapping = {
    'overall_sentiment': 'sentiment',
    'detected_language': 'language'
}
```

## 🧪 Testing Steps

### Local Testing:
```bash
# 1. Test normalization function
python test_backend_integration.py

# 2. Run app locally
streamlit run app.py

# 3. Upload sample CSV and process
# 4. Verify all visualizations display correctly
# 5. Check debug section shows correct columns
```

### Expected Behavior:
- ✅ File upload accepts CSV with required columns
- ✅ Processing sends correct API request format
- ✅ Response normalized to frontend format
- ✅ All 4 pages display without errors
- ✅ Visualizations show correct data
- ✅ Filters work with normalized columns
- ✅ KPI cards calculate accurate percentages

## 📊 Sample Test Data

Use this CSV to test:
```csv
id,reviews_title,review,date,user_id
1,Great Service,Excellent customer support!,2025-09-20,u101
2,Slow Process,The application process took too long.,2025-09-21,u102
```

Expected backend response should include:
- `overall_sentiment`: "Positive", "Negative"
- `detected_language`: "en"
- `aspects`: extracted aspects
- `intent`: classified intent

## 🌐 Streamlit Cloud Deployment

### Steps:
1. **Create GitHub Repository**
   - Name: `sentiment-analytics-frontend`
   - Visibility: Public (for free tier)

2. **Upload Files**
   - Push all files from `streamlit-deployment/` folder
   - Exclude `test_backend_integration.py` (optional)
   - Include `BACKEND_INTEGRATION.md` (optional documentation)

3. **Deploy on Streamlit Cloud**
   - Go to: https://share.streamlit.io
   - Connect GitHub account
   - Select repository: `sentiment-analytics-frontend`
   - Main file: `app.py`
   - Branch: `main`
   - Click "Deploy"

4. **Verify Deployment**
   - Wait for build to complete (~2-3 minutes)
   - Test file upload functionality
   - Verify API connection works
   - Check all visualizations display
   - Test filters and navigation

## 🔍 Post-Deployment Verification

### Functionality Checks:
- [ ] Home page loads without errors
- [ ] File upload accepts CSV
- [ ] API processing completes successfully
- [ ] Debug section shows 13 columns from backend
- [ ] Analytics page displays all charts
- [ ] Sentiment timeline shows data
- [ ] Aspect heatmap renders correctly
- [ ] Intent distribution shows percentages
- [ ] Language distribution displays
- [ ] Network graph creates
- [ ] Filters work correctly
- [ ] History page saves sessions
- [ ] Documentation page displays

### Performance Checks:
- [ ] Page loads in < 5 seconds
- [ ] File processing completes in < 2 minutes (depends on backend)
- [ ] Charts render smoothly
- [ ] No memory errors (within 1GB limit)

## 🐛 Common Issues & Solutions

### Issue: "KeyError: 'sentiment'"
**Solution**: Normalization function maps `overall_sentiment` → `sentiment`

### Issue: "No data to display"
**Solution**: Check debug section for actual column names returned by API

### Issue: "API Error 422"
**Solution**: Verify request format matches backend expectations (ProcessRequest model)

### Issue: "Charts not displaying"
**Solution**: Each chart has unique key to prevent duplicate IDs

### Issue: "Wrong language percentages"
**Solution**: Now compares against 'EN' code, not 'ENGLISH'

## 📞 Support

- **Backend Issues**: Check HuggingFace Spaces logs
- **Frontend Issues**: Check Streamlit Cloud logs
- **API Integration**: Review `BACKEND_INTEGRATION.md`
- **Testing**: Run `test_backend_integration.py`

## ✅ Final Checklist

Before deploying to production:
- [ ] All local tests pass
- [ ] Backend API is stable and responsive
- [ ] Sample data processes correctly
- [ ] All visualizations display properly
- [ ] Error messages are user-friendly
- [ ] Documentation is complete
- [ ] Repository is organized and clean
- [ ] .gitignore excludes sensitive files
- [ ] README has deployment instructions

## 🎉 Ready to Deploy!

Once all items are checked, your application is ready for Streamlit Cloud deployment!
