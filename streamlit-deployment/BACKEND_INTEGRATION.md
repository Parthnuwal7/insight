# Backend API Integration Guide

## 📊 Backend Response Format

Your HuggingFace Spaces backend returns data with the following structure:

### Response Columns (13 total):
```
1.  id                    - Unique identifier
2.  reviews_title         - Review title
3.  review                - Original review text
4.  date                  - Review date (ISO format)
5.  user_id               - User identifier
6.  translated_review     - Translated text (if applicable)
7.  detected_language     - Language code (e.g., 'en', 'hi')
8.  intent                - Customer intent classification
9.  intent_severity       - Severity level of intent
10. intent_confidence     - Confidence score for intent
11. aspects               - Extracted aspects (comma-separated or list)
12. aspect_sentiments     - Sentiment for each aspect
13. overall_sentiment     - Overall sentiment (Positive/Negative/Neutral)
```

## 🔄 Column Mapping

The frontend normalizes backend columns to match expected names:

| Backend Column        | Frontend Column | Used For                    |
|----------------------|-----------------|----------------------------|
| `overall_sentiment`  | `sentiment`     | Sentiment visualizations   |
| `detected_language`  | `language`      | Language distribution      |
| `intent`             | `intent`        | Intent classification      |
| `aspects`            | `aspects`       | Aspect analysis            |
| `aspect_sentiments`  | `aspect_sentiments` | Aspect-sentiment heatmap |

## 🛠️ Data Transformation

The `normalize_backend_response()` function handles:

1. **Column Renaming**: Maps backend names to frontend expectations
2. **Default Values**: Adds missing columns with safe defaults
3. **Data Validation**: Ensures all required columns exist

### Example Transformation:

**Backend Response:**
```json
{
  "overall_sentiment": "Positive",
  "detected_language": "en",
  "aspects": "payment gateway",
  "aspect_sentiments": "Negative"
}
```

**After Normalization:**
```json
{
  "sentiment": "Positive",
  "language": "en", 
  "overall_sentiment": "Positive",
  "detected_language": "en",
  "aspects": "payment gateway",
  "aspect_sentiments": "Negative"
}
```

## 📈 Visualization Compatibility

### Sentiment Analysis
- Uses: `sentiment` or `overall_sentiment`
- Format: "Positive", "Negative", "Neutral" (case-insensitive)
- Charts: Timeline, Distribution, KPI cards

### Language Detection
- Uses: `language` or `detected_language`
- Format: Language code (e.g., "en", "hi")
- Charts: Language distribution, Multilingual percentage

### Intent Classification
- Uses: `intent`
- Format: Intent category string
- Charts: Intent distribution pie chart

### Aspect Analysis
- Uses: `aspects`, `aspect_sentiments`
- Format: String or list of aspects
- Charts: Heatmap, Network graph

## 🔍 Filtering

Filters automatically detect and use the correct column names:

```python
# Sentiment filter checks both columns
sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'overall_sentiment'

# Language filter checks both columns  
language_col = 'language' if 'language' in df.columns else 'detected_language'
```

## ✅ Validation

The frontend includes:

1. **Column Validation**: Checks for required columns before visualization
2. **Data Type Checking**: Handles strings, lists, and null values safely
3. **Debug Information**: Shows available columns in Analytics page
4. **Graceful Fallbacks**: Displays informative messages when data is missing

## 🚀 API Endpoint

Currently configured:
```python
HF_SPACES_API_URL = "https://parthnuwal7-absa.hf.space"
```

## 📝 Sample Data Processing

**Input (CSV):**
```csv
id,reviews_title,review,date,user_id
1,Great Product,Love this product!,2025-01-01,u101
```

**Backend Processing:**
- Language detection: `en`
- Sentiment analysis: `Positive`
- Aspect extraction: `product`
- Intent classification: `appreciation`

**Frontend Display:**
- Timeline chart with sentiment trends
- KPI cards with metrics
- Aspect relationship network
- Interactive filters

## 🔧 Troubleshooting

### Issue: Missing Columns
**Solution**: `normalize_backend_response()` adds defaults

### Issue: KeyError on 'sentiment'
**Solution**: Frontend checks both `sentiment` and `overall_sentiment`

### Issue: Wrong Language Percentage
**Solution**: Compares against 'EN' (uppercase) instead of 'ENGLISH'

### Issue: Empty Visualizations
**Solution**: Debug expander shows available columns for troubleshooting

## 📞 Support

For backend API changes, update the column mapping in:
```python
def normalize_backend_response(df: pd.DataFrame):
    column_mapping = {
        'backend_col': 'frontend_col',
        # Add new mappings here
    }
```
