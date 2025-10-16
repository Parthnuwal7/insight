# 🐛 Debugging Guide: "['sentiment'] not in index" Error

## 🔍 Issue Analysis

The error "['sentiment'] not in index" means the DataFrame doesn't have a 'sentiment' column when the code tries to access it.

## 📊 What We've Added

### 1. Enhanced Debug Information
After uploading and processing a file, you'll now see multiple debug sections:

- **Debug: API Response Structure** - Shows what the backend returned
- **Debug: Before Normalization** - Shows columns from backend
- **Debug: After Normalization** - Shows columns after mapping (should include 'sentiment')

### 2. Improved Error Handling
- Better error messages with full traceback
- Shows actual column names available
- Indicates where the error occurred

## 🔧 How to Debug

### Step 1: Upload Your File
Upload your CSV file as normal on the Home page.

### Step 2: Check Debug Sections
After clicking "Process Reviews with AI", expand these sections:

#### A. "Debug: API Response Structure"
Look for:
```
Response keys: ['status', 'data']
Data keys: ['processed_data', 'summary', ...]
Number of records: 10
Sample record keys: ['id', 'review', 'overall_sentiment', 'detected_language', ...]
```

#### B. "Debug: Before Normalization"
Should show:
```
Columns: ['id', 'reviews_title', 'review', 'date', 'user_id', 'translated_review', 
          'detected_language', 'intent', 'intent_severity', 'intent_confidence', 
          'aspects', 'aspect_sentiments', 'overall_sentiment']
Shape: (10, 13)
```

#### C. "Debug: After Normalization" ⭐ MOST IMPORTANT
Should show:
```
Columns: [...all previous columns..., 'sentiment', 'language']
Has 'sentiment' column: True
Has 'language' column: True
```

## ❓ Common Scenarios

### Scenario 1: Backend Returns Different Structure
**Symptoms:**
- "Debug: Before Normalization" shows unexpected columns
- No `overall_sentiment` or `detected_language` columns

**Solution:**
- Check the "Sample record keys" in "API Response Structure"
- Note the actual column names
- Update `normalize_backend_response()` function with correct mapping

### Scenario 2: Normalization Not Applied
**Symptoms:**
- "Debug: After Normalization" shows `Has 'sentiment' column: False`
- Normalization function didn't create the column

**Solution:**
- Check if `overall_sentiment` exists in "Before Normalization"
- If it exists but 'sentiment' wasn't created, there's a bug in normalization
- Share the debug output for further assistance

### Scenario 3: Data Structure Issue
**Symptoms:**
- `processed_data` is empty or not a list
- DataFrame has 0 rows

**Solution:**
- Check "Number of records" in API Response Structure
- Verify backend is actually processing the data
- Check backend logs on HuggingFace Spaces

## 🛠️ Quick Fixes to Try

### Fix 1: Clear Cache and Reboot
On Streamlit Cloud:
1. Go to Settings (⚙️) → "Reboot app"
2. Or: Settings → "Clear cache" → "Reboot app"

### Fix 2: Check Backend API
Test your backend directly:
```python
import requests
import json

# Test data
test_data = {
    "data": [{
        "id": 1,
        "reviews_title": "Test",
        "review": "Great product!",
        "date": "2025-01-01",
        "user_id": "u101"
    }],
    "options": {}
}

# Call API
response = requests.post(
    "https://parthnuwal7-absa.hf.space/process-reviews",
    json=test_data,
    timeout=120
)

# Check response
print("Status:", response.status_code)
print("Response:", response.json())
```

### Fix 3: Verify Column Names
If backend returns different column names:

1. Look at "Debug: Before Normalization"
2. Note the actual column names
3. Update mapping in `app.py`:

```python
column_mapping = {
    'your_backend_sentiment_column': 'sentiment',
    'your_backend_language_column': 'language'
}
```

## 📝 What to Share for Help

If the issue persists, share these details:

1. **API Response Structure:**
   - Response keys
   - Data keys
   - Sample record keys

2. **Before Normalization:**
   - All column names
   - DataFrame shape

3. **After Normalization:**
   - All column names
   - Has 'sentiment' column: True/False
   - Has 'language' column: True/False

4. **Error Traceback:**
   - Full error message
   - Line number where error occurred

## ✅ Expected Behavior

When working correctly, you should see:

1. ✅ API returns 13 columns including `overall_sentiment` and `detected_language`
2. ✅ Normalization creates `sentiment` and `language` columns
3. ✅ DataFrame has both original and new columns
4. ✅ KPI cards display without errors
5. ✅ Sample results show data in all columns
6. ✅ Analytics page displays all visualizations

## 🚀 Next Steps

1. **Reboot the Streamlit app** on Streamlit Cloud
2. **Upload your CSV file** again
3. **Click "Process Reviews with AI"**
4. **Expand ALL debug sections** and check each one
5. **Take screenshots** of debug output if issue persists
6. **Check the error details** expander for full traceback

The enhanced debugging will show exactly where the issue is occurring and what data is actually available at each step.
