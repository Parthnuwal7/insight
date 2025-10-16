# Local Testing Guide

This guide walks you through testing the Streamlit app locally before deploying to Streamlit Cloud.

## 🎯 Testing Approaches

### 1. Quick Test with Existing Script

Run the provided test script to verify column normalization:

```bash
cd streamlit-deployment
python test_backend_integration.py
```

**Expected Output:**
```
📊 Original Backend Response:
Columns: ['id', 'reviews_title', 'review', 'date', 'user_id', 'translated_review', 
          'detected_language', 'intent', 'intent_severity', 'intent_confidence', 
          'aspects', 'aspect_sentiments', 'overall_sentiment']

✅ After Normalization:
Columns: [..., 'sentiment', 'language']
✓ 'sentiment' column: ['Positive', 'Negative']
✓ 'language' column: ['en', 'en']

✅ All tests passed!
```

---

### 2. Full Local Streamlit Test

#### Step 1: Install Dependencies
```bash
cd streamlit-deployment
pip install -r requirements.txt
```

#### Step 2: Run Streamlit Locally
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

#### Step 3: Test Full Workflow

**A. Upload Test File**
1. Navigate to **Home** page
2. Upload a CSV with these columns: `id, reviews_title, review, date, user_id`
3. Click "Process Reviews with AI"

**B. Check Debug Output**
Look for these debug sections (they'll appear after processing):

1. **"Debug: API Response Structure"**
   - Shows backend response format
   - Verify it has `processed_data` key

2. **"Debug: Before Normalization"**
   - Should show columns: `overall_sentiment`, `detected_language`
   - Verify data is present

3. **"Debug: After Normalization"** (expanded by default)
   - **CRITICAL CHECK:** Look for:
     - `Has 'sentiment' column: True`
     - `Has 'language' column: True`
   - If False, normalization failed

**C. Test Visualizations**
1. Check if KPI cards show correct metrics
2. Navigate to **Analytics** page
3. Verify all charts load without errors
4. Test filters (sentiment, intent, language)

---

### 3. API Integration Test

Test the backend API connection independently:

```bash
cd streamlit-deployment
python test_api_connection.py
```

(I'll create this script below)

---

### 4. Test With Sample Data

#### Option A: Use Provided Test Data
Create `test_reviews.csv`:

```csv
id,reviews_title,review,date,user_id
1,Smooth Filing,The new filing system is really smooth and quick.,2025-09-20,u101
2,Payment Issue,I faced multiple payment gateway errors while paying.,2025-09-21,u102
3,Great Service,बहुत अच्छी सेवा है। मुझे पसंद आया।,2025-09-22,u103
4,Slow Response,Response time is too slow. Need improvement.,2025-09-23,u104
5,Excellent,Amazing experience! Highly recommend.,2025-09-24,u105
```

#### Option B: Generate Test Data
The app has a "Generate Test Data" button on the Home page that creates sample reviews.

---

## 🔍 What to Check During Testing

### ✅ Critical Checkpoints

1. **Backend Connection**
   - Health check succeeds
   - API returns 200 status
   - Response has `success: true`

2. **Column Normalization**
   - `overall_sentiment` → `sentiment`
   - `detected_language` → `language`
   - Both columns exist in normalized DataFrame

3. **Data Processing**
   - Aspects extracted correctly
   - Sentiments assigned to aspects
   - Intent classification works
   - Translation works for Hindi reviews

4. **Visualizations**
   - No KeyError for 'sentiment' or 'language'
   - All charts render properly
   - Filters work without errors
   - KPI metrics calculate correctly

5. **Session Management**
   - Can save dashboard state
   - Can load previous sessions
   - History page shows saved sessions

---

## 🐛 Common Issues and Fixes

### Issue 1: KeyError: 'sentiment'
**Symptom:** Error when trying to access sentiment column  
**Debug:** Check "Debug: After Normalization" section  
**Fix:** If `Has 'sentiment' column: False`, the normalization function isn't working

**Solution:**
```python
# Check if normalize_backend_response is being called
# Add print statement before normalization:
print(f"Before: {list(df.columns)}")
df = normalize_backend_response(df)
print(f"After: {list(df.columns)}")
```

### Issue 2: API Connection Failed
**Symptom:** "Backend health check failed"  
**Fix:** Verify HF Spaces URL is correct and backend is running
```python
# Test manually:
import requests
response = requests.get("https://parthnuwal7-absa.hf.space/health")
print(response.json())
```

### Issue 3: No Data After Processing
**Symptom:** Processing completes but no results shown  
**Debug:** Check if `processed_data` is empty  
**Fix:** Verify CSV format matches expected columns

### Issue 4: Translation Not Working
**Symptom:** Hindi reviews not translated  
**Debug:** Check `detected_language` and `translated_review` columns  
**Fix:** Backend M2M100 model might be loading slowly (first request takes time)

---

## 📊 Testing Scenarios

### Scenario 1: English Only Reviews
- Upload CSV with only English reviews
- Verify: `language` column shows 'en'
- Check: No translation overhead

### Scenario 2: Mixed Language Reviews
- Include Hindi and English reviews
- Verify: Hindi reviews have translated_review
- Check: `language` column shows 'hi' for Hindi

### Scenario 3: Edge Cases
- Empty reviews
- Very long reviews (>500 words)
- Special characters
- Malformed dates

### Scenario 4: Large Dataset
- Upload 100+ reviews
- Check: Progress bar shows during processing
- Verify: No timeout errors
- Check: All visualizations render

---

## 🚀 Pre-Deployment Checklist

Before pushing to GitHub and deploying:

- [ ] `test_backend_integration.py` passes all tests
- [ ] Local Streamlit app runs without errors
- [ ] Can upload and process sample CSV
- [ ] All debug sections show correct data
- [ ] `sentiment` and `language` columns exist after normalization
- [ ] All 4 pages (Home, Analytics, History, Documentation) work
- [ ] Filters apply correctly
- [ ] KPI cards show accurate metrics
- [ ] Charts render without errors
- [ ] Session save/load works
- [ ] No console errors in browser (F12 developer tools)

---

## 🎓 Advanced Testing

### Test with Mock Backend Response

If you want to test without hitting the API:

```python
# In app.py, temporarily replace API call:
def process_reviews_mock(file_content):
    # Return mock data matching backend format
    return {
        "success": True,
        "processed_data": [
            {
                "id": 1,
                "review": "Test review",
                "overall_sentiment": "Positive",
                "detected_language": "en",
                "aspects": "test aspect",
                "aspect_sentiments": "Positive",
                # ... other fields
            }
        ]
    }
```

### Performance Testing

Test with large datasets:
```python
# Generate 1000 reviews
import pandas as pd
df = pd.DataFrame({
    'id': range(1000),
    'reviews_title': ['Test'] * 1000,
    'review': ['Good product'] * 1000,
    'date': ['2025-01-01'] * 1000,
    'user_id': [f'u{i}' for i in range(1000)]
})
df.to_csv('large_test.csv', index=False)
```

---

## 📝 Next Steps After Testing

Once all tests pass locally:

1. **Commit changes:**
   ```bash
   git add .
   git commit -m "Fix: Enhanced debugging and normalization"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to your Streamlit Cloud dashboard
   - Click "Reboot" on your app
   - Or deploy fresh from GitHub

3. **Monitor deployed app:**
   - Check logs in Streamlit Cloud dashboard
   - Test with same CSV file
   - Verify debug sections show correct data

4. **Share results:**
   - If issues persist, share debug output from deployed app
   - Include all three debug sections
   - Share any error tracebacks

---

## 🆘 Getting Help

If tests fail:

1. **Run test_backend_integration.py** - Shows column normalization
2. **Check debug expanders** in the Streamlit app - Shows actual data flow
3. **Review DEBUGGING_GUIDE.md** - Detailed troubleshooting scenarios
4. **Check logs** - Streamlit shows errors in the app and in Cloud dashboard

**What to share if asking for help:**
- Output from `test_backend_integration.py`
- Screenshot of "Debug: After Normalization" section
- Full error traceback
- CSV file used for testing
