# Dashboard Layout Visual Guide

## Complete Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED ANALYTICS DASHBOARD                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  TOP ROW: ENHANCED KPI CARDS (Full Width - 5 Columns)                   │
├──────────┬──────────┬──────────┬──────────────┬──────────────────────────┤
│ 📝 Total │ 😊 Pos % │ 😞 Neg % │ 🔴 Top       │ 🎯 Dominant             │
│ Reviews  │          │          │ Complaint    │ Intent                  │
│  1,234   │  45.2%   │  23.8%   │  Design      │  Complaint              │
└──────────┴──────────┴──────────┴──────────────┴──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  FILTER BAR: Multi-Select + Date Range                                  │
├────────────────────┬────────────────────┬────────────────────┬───────────┤
│ Sentiment          │ Intent             │ Language           │ Aspects   │
│ ☑ Positive         │ ☑ complaint        │ ☑ en               │ ☑ Design  │
│ ☑ Neutral          │ ☑ praise           │ ☑ hi               │ ☑ Price   │
│ ☑ Negative         │ ☑ question         │                    │ ☑ Quality │
├──────────────────────────────────────────┴────────────────────┴───────────┤
│ From Date: 2024-01-01     To Date: 2024-12-31                           │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROW 1: OVERVIEW
┌────────────────────────────────────┬────────────────────────────────────┐
│ 📊 Sentiment Distribution          │ 🎯 Intent × Aspect Heatmap         │
│                                    │                                    │
│     Donut chart showing:           │  Matrix showing which intents      │
│     - Positive: 45.2%              │  appear with which aspects         │
│     - Neutral: 31.0%               │                                    │
│     - Negative: 23.8%              │  Aspects (rows) × Intents (cols)   │
│                                    │  Top 15 aspects                    │
│     Colorblind-safe palette        │  YlOrRd colorscale                 │
└────────────────────────────────────┴────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROW 2: SENTIMENT PATTERNS
┌────────────────────────────────────┬────────────────────────────────────┐
│ 💭 Aspect × Sentiment Heatmap      │ 📈 Reviews Timeline                │
│                                    │                                    │
│  Matrix showing sentiment for      │  Daily review volume with          │
│  each aspect                       │  sentiment breakdown               │
│                                    │                                    │
│  Aspects (rows) ×                  │  Filled area chart with            │
│  Sentiments (columns)              │  multi-line overlay:               │
│                                    │  - Total volume (filled)           │
│  RdYlGn diverging scale            │  - Positive (green line)           │
│  Top 15 aspects                    │  - Neutral (blue line)             │
│                                    │  - Negative (red line)             │
└────────────────────────────────────┴────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROW 3: PRIORITY INSIGHTS
┌────────────────────────────────────┬────────────────────────────────────┐
│ 🚨 Priority Aspects Leaderboard    │ 📊 Aspect Sentiment Trends         │
│                                    │                                    │
│  Horizontal bar chart:             │  [COMING SOON]                     │
│  Priority Score Algorithm:         │                                    │
│  neg_ratio × log(1 + count)        │  Multi-line time-series showing    │
│                                    │  sentiment evolution for           │
│  Design          ████████ 3.45     │  top 5-8 aspects over time         │
│  Price           ██████ 2.89       │                                    │
│  Durability      ████ 2.12         │                                    │
│  Quality         ███ 1.87          │                                    │
│  ...                               │                                    │
│                                    │                                    │
│  Color-coded by severity:          │                                    │
│  🔴 Red: >50% negative             │                                    │
│  🟠 Orange: 30-50% negative        │                                    │
│  🟡 Yellow: <30% negative          │                                    │
└────────────────────────────────────┴────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROW 4: CORRELATIONS & INSIGHTS
┌────────────────────────────────────┬────────────────────────────────────┐
│ 🔗 Aspect Co-occurrence Matrix     │ 💡 AI-Generated Insights           │
│                                    │                                    │
│  Symmetric heatmap showing         │  [COMING SOON - LLM Integration]   │
│  which aspects appear together     │                                    │
│                                    │  ┌──────────────────────────────┐  │
│  15×15 matrix with Blues scale     │  │ 🔍 Top Priority:             │  │
│                                    │  │ Design needs immediate       │  │
│  Useful for finding aspect         │  │ attention (65% negative)     │  │
│  relationships and patterns        │  └──────────────────────────────┘  │
│                                    │                                    │
│                                    │  ┌──────────────────────────────┐  │
│                                    │  │ ✅ Strength Area:            │  │
│                                    │  │ Performance consistently     │  │
│                                    │  │ praised (78% positive)       │  │
│                                    │  └──────────────────────────────┘  │
└────────────────────────────────────┴────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROW 5: DEEP DIVE ANALYSIS
┌────────────────────────────────────────────────────┬───────────────────┐
│ 🔍 Review Details Explorer (2/3 width)             │ 📊 Confidence     │
│                                                    │    Funnel         │
│ ▼ Click to explore individual reviews             │   (1/3 width)     │
│                                                    │                   │
│  Sort by: Date (Newest) ▼  | Reviews/page: 25 ▼  │   High (>0.9)     │
│  Page: 1 of 50                                    │   ████████        │
│                                                    │   65%             │
│  ┌────────────────────────────────────────────┐   │                   │
│  │ Review: "Great product but design is..."   │   │   Good            │
│  │ Sentiment: Positive | Intent: praise       │   │   ██████          │
│  │ Confidence: 92% | Aspects: Design, Quality│   │   20%             │
│  └────────────────────────────────────────────┘   │                   │
│                                                    │   Medium          │
│  ┌────────────────────────────────────────────┐   │   ███             │
│  │ Review: "Poor quality control, many..."    │   │   10%             │
│  │ Sentiment: Negative | Intent: complaint    │   │                   │
│  │ Confidence: 87% | Aspects: Quality, Price  │   │   Low (<0.6)      │
│  └────────────────────────────────────────────┘   │   █               │
│                                                    │   5%              │
│  [📥 Export Drill-Down Data]                      │                   │
└────────────────────────────────────────────────────┴───────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEGACY COMPONENTS (Retained)
┌─────────────────────────────────────────────────────────────────────────┐
│  ☁️ Word Clouds                                                          │
├─────────────────────────┬─────────────────────────┬─────────────────────┤
│ 😊 Positive Reviews     │ 😐 Neutral Reviews      │ 😞 Negative Reviews │
│                         │                         │                     │
│  [Word Cloud Image]     │  [Word Cloud Image]     │  [Word Cloud Image] │
│                         │                         │                     │
└─────────────────────────┴─────────────────────────┴─────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  🕸️ Aspect Network                                                       │
│                                                                          │
│  Interactive network graph showing aspect relationships from backend    │
│                                                                          │
│  [Network Visualization]                                                │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOOTER: EXPORT & UTILITIES
┌─────────────────────────┬─────────────────────────┬─────────────────────┐
│ 📥 Download Full        │ 📄 Generate PDF Report  │ 🔔 Setup Alerts     │
│    Dataset (CSV)        │    [Coming Soon]        │    [Coming Soon]    │
│    [Active]             │                         │                     │
└─────────────────────────┴─────────────────────────┴─────────────────────┘
```

## Component Details

### Enhanced KPI Cards
- **Total Reviews:** Simple count
- **Positive %:** Percentage of positive sentiment reviews
- **Negative %:** Percentage of negative sentiment reviews
- **Top Complaint:** Most mentioned aspect in negative reviews
- **Dominant Intent:** Most common intent across all reviews

### Multi-Select Filters
- **Behavior:** All options selected by default
- **Logic:** AND operation between filter types, OR within each type
- **Example:** (Sentiment IN [Positive, Neutral]) AND (Intent IN [praise]) AND (Aspects CONTAINS ANY [Design, Quality])

### Priority Score Formula
```
priority_score = (negative_count / total_count) × log(1 + total_count)

Where:
- negative_count: Number of negative mentions of aspect
- total_count: Total mentions of aspect
- log(1 + x): Logarithmic scaling prevents rare aspects from dominating
```

**Interpretation:**
- High negative ratio + high frequency = High priority (urgent)
- High negative ratio + low frequency = Medium priority (monitor)
- Low negative ratio = Low priority (even if high frequency)

### Confidence Funnel Buckets
- **High (>0.9):** Model is very confident, trust strongly
- **Good (0.8-0.9):** Reliable predictions, safe to use
- **Medium (0.6-0.8):** Moderate confidence, verify for important decisions
- **Low (<0.6):** Model uncertain, requires human review

### Color Schemes (Accessibility)

**Sentiment Colors (Colorblind-Safe):**
- Positive: `#2E7D32` (Dark Green)
- Neutral: `#1976D2` (Blue)
- Negative: `#D32F2F` (Red)

**Intent Colors:**
- complaint: `#D32F2F` (Red)
- praise: `#2E7D32` (Green)
- question: `#1976D2` (Blue)
- suggestion: `#7B1FA2` (Purple)
- comparison: `#F57C00` (Orange)
- neutral: `#757575` (Gray)

**Priority Severity:**
- High (>50% neg): `#D32F2F` (Red)
- Medium (30-50%): `#F57C00` (Orange)
- Low (<30%): `#FDD835` (Yellow)

## User Workflows

### Workflow 1: Identify Urgent Issues
1. View **Priority Leaderboard** (ROW 3, Left)
2. Click top aspect → drill-down panel filters to that aspect
3. Review individual complaints in **Drill-Down Panel**
4. Export filtered reviews for deeper analysis
5. Check **Confidence Funnel** to validate model reliability

### Workflow 2: Track Sentiment Over Time
1. Set **Date Range Filter** to specific period
2. View **Reviews Timeline** (ROW 2, Right) for trends
3. Check **Aspect Sentiment Trends** (ROW 3, Right) for specific aspects
4. Export time-series data for forecasting

### Workflow 3: Understand Aspect Relationships
1. View **Co-occurrence Heatmap** (ROW 4, Left)
2. Identify frequently co-mentioned aspects
3. Use **Multi-Select Aspect Filter** to analyze those combinations
4. Review **Intent-Aspect Heatmap** (ROW 1, Right) to see intent patterns

### Workflow 4: Validate Model Performance
1. Check **Confidence Funnel** (ROW 5, Right)
2. If >20% in Low bucket, investigate:
   - **Drill-Down Panel** → Sort by Confidence (Low)
   - Review low-confidence predictions
   - Consider retraining or adjusting thresholds
3. Export low-confidence reviews for labeling

## Technical Notes

### Performance Optimizations
- **Heatmaps:** Limited to top 15 aspects (225 cells max)
- **Co-occurrence:** Uses `itertools.combinations()` for efficiency
- **Pagination:** Drill-down loads 10-100 reviews at a time
- **Timeline:** Aggregates to daily granularity
- **Aspect extraction:** Cached after first filter application

### Browser Compatibility
- **Tested:** Chrome 120+, Firefox 120+, Safari 17+, Edge 120+
- **Minimum:** Any browser supporting Plotly.js 2.0+
- **Mobile:** Responsive 2-column → 1-column on <768px

### Data Requirements
**Minimum:**
- `review` column (text)
- `sentiment` column (Positive/Neutral/Negative)

**Recommended:**
- `date` column (datetime or parseable string)
- `intent` column (complaint/praise/question/suggestion/comparison/neutral)
- `aspects` column (JSON string or list)
- `confidence` column (float 0-1)

**Optional:**
- `language` column (for language filter)
- `user_id` column (for user-level analysis)

---

**Dashboard Version:** 2.0  
**Last Updated:** 2024-01-XX  
**Status:** Production-Ready with Phase 2 Placeholders
