# AI Insights Setup Guide

## OpenRouter API Configuration

The dashboard now includes **AI-Powered Insights & Recommendations** using the **nvidia/nemotron-3-nano-30b-a3b:free** model via OpenRouter.

### Setup Steps

1. **Get OpenRouter API Key**
   - Visit: https://openrouter.ai/keys
   - Sign up or log in
   - Create a new API key
   - Copy the key (starts with `sk-or-...`)

2. **Configure Environment Variable**
   
   Create a `.env` file in the `streamlit-deployment/` directory:
   
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
   ```

3. **Run the Dashboard**
   
   ```bash
   cd streamlit-deployment
   streamlit run app_a.py
   ```

### Features

The AI Insights section provides:

✅ **Key Findings** - Most important patterns in your data
✅ **Strengths** - Aspects performing well  
✅ **Areas for Improvement** - Aspects needing attention
✅ **Actionable Recommendations** - Specific steps to take

### Model Details

- **Model**: `nvidia/nemotron-3-nano-30b-a3b:free`
- **Provider**: OpenRouter
- **Cost**: FREE tier (rate-limited)
- **Max Tokens**: 800 per request
- **Temperature**: 0.7 (balanced creativity)

### Fallback Behavior

If the API key is missing or the API fails:
- The dashboard will show **pattern-based insights** instead
- No errors will interrupt the user experience
- A friendly message indicates AI insights are unavailable

### Privacy & Security

- API calls are made server-side only
- Your review data is sent to OpenRouter for analysis
- OpenRouter's privacy policy: https://openrouter.ai/privacy
- Keep your `.env` file private (already in `.gitignore`)

### Troubleshooting

**Problem**: "AI insights are currently unavailable"
- **Solution**: Check that `OPENROUTER_API_KEY` is set in `.env` file
- Verify the key is valid (test at https://openrouter.ai/docs)

**Problem**: API errors or timeout
- **Solution**: OpenRouter free tier has rate limits - wait a minute and retry
- Consider upgrading to paid tier for higher limits

**Problem**: Generic/irrelevant insights
- **Solution**: This is a data issue - ensure you have processed data with aspect-level analysis
- Re-process your reviews to get `aspect_level_data`

### Example Output

```
🤖 AI Analysis

**Key Findings:**
1. Quality aspects show strong positive sentiment (85%) but are often 
   mentioned alongside negative delivery feedback
2. 23% of reviews have mixed sentiments, indicating nuanced customer experiences

**Strengths:**
- Performance: 92% positive mentions
- Quality: Consistently praised when standalone

**Areas for Improvement:**
- Delivery: 68% negative sentiment, top complaint aspect
- Price: Mentioned negatively when paired with Quality

**Actionable Recommendations:**
1. Investigate delivery process - root cause of delays
2. Consider competitive pricing analysis for premium products
3. Leverage strong quality feedback in marketing materials
```

---

**Questions?** Check OpenRouter docs: https://openrouter.ai/docs
