# Co-occurrence Matrix Debugging Guide

## Problem
The aspect co-occurrence matrix is showing all zeros, even though reviews have multiple aspects.

## Fixes Applied

### 1. **Created `get_top_aspects_by_frequency()` function**
- Previously used alphabetically sorted aspects
- Now uses the 15 most frequently occurring aspects
- Located in: `dashboard_components.py`

### 2. **Enhanced co-occurrence calculation**
- Only counts reviews with 2+ aspects from the top 15
- Added diagnostic statistics in chart title
- Shows warning if no co-occurrences found

### 3. **Added diagnostic component**
- Shows sample aspect data (raw format + extracted)
- Displays statistics: reviews with aspects, reviews with 2+ aspects
- Lists top 15 aspects with frequency counts
- Shows example reviews with multiple aspects

## How to Debug

### Step 1: View Diagnostics in Dashboard
1. Reload your Streamlit app
2. Navigate to Advanced Analytics page
3. Scroll down to the **"🔍 Aspect Data Diagnostics"** section
4. Click to expand it

### Step 2: Check What the Diagnostics Show

**Look for these key metrics:**

```
Total Reviews: X
Reviews with Aspects: Y
Reviews with 2+ Aspects: Z  ← This is critical!
Avg Aspects/Review: A.B
```

**If "Reviews with 2+ Aspects" is 0 or very low:**
- That's why the co-occurrence matrix is all zeros
- Co-occurrence requires at least 2 aspects per review

### Step 3: Examine Sample Data

The diagnostics will show 5 random reviews like:

```
Review #123:
Raw value: "['Design', 'Quality', 'Price']"
Type: <class 'str'>
Extracted: ['Design', 'Quality', 'Price']
Count: 3
```

**Check:**
- Is `Type` showing `str` or `list`?
- Are aspects being extracted correctly?
- Do you see multiple aspects per review?

### Step 4: Check Co-occurrence Sample

The diagnostics show the first 10 reviews with 2+ aspects:

```
1. ['Design', 'Quality'] - 'Great design but quality issues...'
2. ['Price', 'Performance'] - 'Good performance for the price...'
```

**If this section is empty:**
- No reviews have multiple aspects → explains zero matrix
- Need to investigate why PyABSA isn't extracting multiple aspects

## Common Issues & Solutions

### Issue 1: Aspects stored as string, not parsed
**Symptoms:** Raw value shows `"['Design', 'Quality']"` (with quotes), but extracted is empty

**Solution:** Check if aspects need JSON parsing instead of `eval()`

### Issue 2: Aspects separated per review
**Symptoms:** Each row has only 1 aspect, even for complex reviews

**Solution:** Check backend API - PyABSA should extract multiple aspects per review

### Issue 3: Top 15 aspects don't overlap in reviews
**Symptoms:** Reviews have multiple aspects, but they're all different

**Solution:** Use more reviews or reduce top N from 15 to 10

### Issue 4: Filtered data has too few reviews
**Symptoms:** Co-occurrence works with full data, fails with filters

**Solution:** Reduce filters or expect lower co-occurrence with small datasets

## Expected Behavior

### Good Co-occurrence Matrix:
```
              Design  Quality  Price  Performance
Design           0      12      8       5
Quality         12       0      15      7
Price            8      15      0       10
Performance      5       7      10      0
```

### Zero Matrix (Current Issue):
```
              Design  Quality  Price  Performance
Design           0       0      0       0
Quality          0       0      0       0
Price            0       0      0       0
Performance      0       0      0       0
```

## Next Steps

1. **Run the dashboard and check diagnostics**
   ```bash
   cd c:\Users\Lenovo\insights\streamlit-deployment
   streamlit run app_a.py
   ```

2. **Share diagnostic output** with me:
   - Reviews with 2+ Aspects count
   - Sample raw aspect values
   - Sample extracted aspects
   - Co-occurrence sample section

3. **Check your backend API response**:
   - Do reviews in the API response have multiple aspects?
   - What format are they in?

4. **Verify data processing**:
   - Are aspects being split/joined incorrectly somewhere?
   - Is each aspect stored in a separate row instead of one list per review?

## Remove Diagnostic Component Later

Once debugging is complete, remove the diagnostic section from `app_a.py`:

```python
# Remove these lines (around line 1530):
from diagnostic_component import show_aspect_diagnostics
show_aspect_diagnostics(filtered_df)
```

## Files Modified

1. ✅ `dashboard_components.py` - Fixed co-occurrence calculation
2. ✅ `diagnostic_component.py` - NEW diagnostic tool
3. ✅ `app_a.py` - Added diagnostic section temporarily
4. ✅ `CO_OCCURRENCE_DEBUG.md` - This guide

---

**Status:** Awaiting diagnostic results from your actual data
**Next:** Share what the diagnostics show and we'll fix the root cause
