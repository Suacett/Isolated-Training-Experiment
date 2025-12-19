# V9 System Enhancements - Implementation Progress

**Last Updated:** 2025-12-13
**Status:** Phase 1 Backend Complete, Phase 1.3 Frontend Pending, Phases 2-7 In Progress

---

## Overview

Comprehensive UI/UX enhancement project adding educational content, bug fixes, and major feature expansions to the V9 AI trading system. **Total 21 implementation tasks across 7 phases.**

---

## Completed Work ✅

### Phase 1.1: Backend Signal Calculation (COMPLETE)
**File Modified:** `backend/routers/playground.py`

**What Was Added:**
- `calculate_signals()` function (lines 40-138)
- `get_spy_data()` helper (lines 24-37)
- Integration into `/playground/compare/{ticker}` endpoint

**Functions Created:**
1. **calculate_signals()** - Returns JSON with:
   ```json
   {
     "confidence": 0.0-100.0,           // AI prediction strength
     "relative_strength": -100.0-100.0,  // vs SPY performance
     "strength_label": "Very Strong|Strong|Weak|Very Weak",
     "similarity": 0.0-100.0,            // Correlation to portfolio
     "similarity_label": "Good|Moderate|High",
     "market_mood": "Calm|Choppy|Volatile",
     "vix_proxy": 0.0-100.0              // Annualized volatility
   }
   ```

2. **get_spy_data()** - Fetches SPY historical data for market comparison

**Integration Points:**
- Added to `/playground/compare/{ticker}` response as `"signals"` key
- Calculates signals using V9 or first available model's predictions
- Normalizes prediction scores to 0-1 range for confidence display

---

### Phase 1.2: Model Explanation Endpoint (COMPLETE)
**File Modified:** `backend/routers/playground.py`

**New Endpoint:** `GET /playground/explain/{ticker}/{model_version}`

**Response Format:**
```json
{
  "model": "V9 Transformer",
  "ticker": "AAPL",
  "current_price": 150.25,
  "why_selected": "V9 Transformer predicts strong relative momentum and ranking",
  "key_factors": [
    "20-day momentum: +12.5% (price trend)",
    "5-day performance: +2.1% (short-term strength)",
    "Volatility: 18.3% annualized (market conditions)"
  ],
  "risk_factors": [
    "Correlation filtering ensures portfolio diversification",
    "Stop-loss protection at 7% below entry price",
    "Rebalanced every 5 trading days based on latest rankings"
  ],
  "architecture": "Transformer with attention mechanism, trained on 60-day price windows with 12 stationary features"
}
```

**Model-Specific Explanations:**
- **V9:** Transformer rank-based predictions, 12 stationary features
- **V7:** BiLSTM + Attention, 52.6% accuracy, -23.7% bearish bias
- **Others:** Generic explanation with custom architecture details

---

## Pending Frontend Work - Phase 1.3

### File: `frontend/app/components/ModelPlayground.tsx`

#### Task 1: Add useEffect Hook (Lines ~400-450)
**Location:** Near state declarations

```typescript
// Add this after existing useEffect hooks
useEffect(() => {
  if (selectedTicker && comparisons[selectedTicker]) {
    // Fetch explanation if not already cached
    if (!comparisons[selectedTicker]?.explanation) {
      fetch(`/playground/explain/${selectedTicker}/v9`)
        .then(res => res.json())
        .then(data => {
          setComparisons(prev => ({
            ...prev,
            [selectedTicker]: {
              ...prev[selectedTicker],
              explanation: data
            }
          }));
        })
        .catch(err => console.error('Failed to fetch explanation:', err));
    }
  }
}, [selectedTicker, comparisons]);
```

#### Task 2: Replace Hardcoded Signal Values (Lines 400-444)

**Current (Hardcoded):**
```tsx
// Line 412
<span className="text-emerald-400 font-bold">94%</span>

// Line 422
<span className="text-emerald-400">+98% (Very Strong)</span>

// Line 432
<span className="text-amber-400">45% (Good)</span>

// Line 442
<span className="text-blue-400">Calm</span>
```

**Replace With (Dynamic):**
```tsx
const signals = comparisons[selectedTicker]?.signals;

// AI Confidence (Line 412)
<span className="text-emerald-400 font-bold">
  {signals?.confidence?.toFixed(1) ?? '--'}%
</span>

// Strength vs Market (Line 422)
<span className={
  (signals?.relative_strength ?? 0) > 0
    ? "text-emerald-400"
    : "text-red-400"
}>
  {signals?.relative_strength > 0 ? '+' : ''}{signals?.relative_strength?.toFixed(1) ?? '--'}%
  ({signals?.strength_label ?? 'Unknown'})
</span>

// Similarity to Others (Line 432)
<span className="text-amber-400">
  {signals?.similarity?.toFixed(1) ?? '--'}% ({signals?.similarity_label ?? '--'})
</span>

// Market Mood (Line 442)
<span className="text-blue-400">
  {signals?.market_mood ?? 'Unknown'}
</span>
```

#### Task 3: Update "Why This Stock?" Section (Lines 449-456)

**Current (Generic):**
```tsx
<strong className="text-white">{selectedTicker}</strong> is performing
<span className="text-emerald-400 ml-1">98% stronger</span> than the market...
```

**Replace With (Dynamic):**
```tsx
const explanation = comparisons[selectedTicker]?.explanation;

{explanation ? (
  <>
    <strong className="text-white">{selectedTicker}</strong> - {explanation.why_selected}
    <ul className="mt-3 space-y-2 ml-4">
      {explanation.key_factors?.map((factor, idx) => (
        <li key={idx} className="text-sm text-zinc-300">
          • {factor}
        </li>
      ))}
    </ul>
    {explanation.risk_factors && (
      <div className="mt-3 pt-3 border-t border-zinc-700">
        <p className="text-xs text-zinc-500 uppercase mb-2">Risk Factors:</p>
        <ul className="space-y-1 ml-4">
          {explanation.risk_factors.map((risk, idx) => (
            <li key={idx} className="text-xs text-zinc-400">
              ⚠️ {risk}
            </li>
          ))}
        </ul>
      </div>
    )}
  </>
) : (
  <span className="text-zinc-400">Loading explanation...</span>
)}
```

#### Task 4: Add Color Helper Function (Top of file)

```typescript
// Add near top of component, after imports
const getSentimentColor = (score: number | null | undefined): string => {
  if (score === null || score === undefined) return "text-zinc-400";
  if (score > 0.3) return "text-emerald-400"; // Bullish
  if (score > 0.1) return "text-lime-400";    // Slightly Positive
  if (score > -0.1) return "text-zinc-400";   // Neutral
  if (score > -0.3) return "text-orange-400"; // Slightly Negative
  return "text-red-400";                       // Bearish
};
```

---

## Remaining Phases - Implementation Roadmap

### Phase 2: Dashboard Educational Tooltips (Next)
**Files to Modify:**
- `frontend/app/components/PaperHoldings.tsx` OR `frontend/app/page.tsx`
- Components exist: `frontend/app/components/InfoTooltip.tsx` (working, no changes needed)

**Tasks:**
1. Find table headers for "Rank", "Corr", "Exposure"
2. Wrap with `<InfoTooltip>` component with educational content
3. Add tooltip to trade "Why?" column explaining rank_score and correlation values

**Expected Time:** 1-2 hours

---

### Phase 3: Trades Section Fixes (Medium Priority)
**Files to Modify:**
- `backend/routers/paper.py` - Remove 50-trade limit, add pagination
- `frontend/app/components/PaperTrades.tsx` - Improve scroll container

**Backend Changes:** Increase limit from 50 to 200, add pagination params, return total count
**Frontend Changes:** Increase max-height from 400px to 600px, add sticky header, optional load-more button

**Expected Time:** 1 hour

---

### Phase 4: AlphaVantage Multi-Key Support (High Priority)
**Files to Create:**
- `backend/services/alpha_vantage_pool.py` - Key pool manager with rotation

**Files to Modify:**
- `backend/services/intrinsic.py` - Use pool instead of single key
- `backend/services/sentiment_data.py` - Use pool instead of single key
- `frontend/app/components/SetupModal.tsx` - Add multi-key textarea input
- `backend/utils/config_loader.py` - Support saving multiple keys

**Key Features:**
- Round-robin key rotation
- Rate limiting: 5 calls/min per key (25/min with 5 keys)
- Automatic 12-second cooldown when all keys exhausted
- Fallback to single key if ALPHA_VANTAGE_KEYS not set

**Expected Time:** 2-3 hours

---

### Phase 5: 5-Year Portfolio Simulation (High Priority)
**Files to Modify:**
- `backend/scripts/seed_paper_history.py` - Change default to 5 years
- `frontend/app/components/SetupModal.tsx` - Rename buttons
- `backend/routers/paper.py` - Add `/paper/reset-and-simulate` endpoint

**Changes:**
- Default: 1 year → 5 years (252 → 1260 trading days)
- Section name: "Market Data Sync" → "Portfolio Simulation"
- Button name: "Full History Sync" → "Reset & Simulate 5 Years"
- Expected time: ~15 seconds (vs 9 seconds for 1 year)

**Expected Time:** 1 hour

---

### Phase 6: 10-Portfolio Comparison (Major Feature)
**Files to Create:**
- `backend/scripts/generate_portfolio_ensemble.py` - Run 10 variations
- `backend/routers/portfolio_comparison.py` - API endpoint
- `frontend/app/components/PortfolioComparison.tsx` - UI component

**Portfolio Variations:**
```
Monte Carlo (Same Config, Different Seeds):
- v9_mc_seed1, v9_mc_seed2, v9_mc_seed3, v9_mc_seed4, v9_mc_seed5

Parameter Variations:
- v9_top5_tight (Top-5, Corr 0.50)
- v9_top15_loose (Top-15, Corr 0.70)
- v9_weekly_rebal (Weekly rebalance vs 5-day default)
- v9_monthly_rebal (Monthly rebalance)
- v9_golden_2025 (Baseline configuration)
```

**Frontend Display:**
- Comparison table (Strategy, Config, Return, Max Drawdown, Sharpe Ratio, Final Value)
- Multi-line equity curve chart
- Click for detailed breakdown per portfolio

**Expected Time:** 4-5 hours

---

### Phase 7: Market Sentiment Integration (Optional Enhancement)
**Files to Modify:**
- `backend/services/sentiment_data.py` - Add bulk fetch method
- `backend/routers/dashboard.py` - Include sentiment in responses
- `frontend/app/components/ModelPlayground.tsx` - Display sentiment in "Market Mood"
- `backend/scripts/seed_paper_history.py` - Optional sentiment post-filter

**Features:**
- Display sentiment as "Bullish", "Bearish", "Neutral", etc.
- Show article count and last update date
- Optional: Adjust V9 rankings by ±10% based on sentiment

**Expected Time:** 2-3 hours

---

## Testing Checklist

### Phase 1.3 (Frontend) - After Implementation:
- [ ] Select ticker in Playground → signals update dynamically
- [ ] Switch to different ticker → signals change
- [ ] Explanation loads → shows correct model-specific info
- [ ] V9 vs V7 explanations differ correctly
- [ ] Browser console shows no errors

### Phase 2:
- [ ] Hover over "Rank" → tooltip shows
- [ ] Hover over "Corr" → tooltip shows
- [ ] Hover over "Exposure" → tooltip shows
- [ ] Tooltips don't obscure other UI elements (z-index correct)

### Phase 3:
- [ ] Scroll trades table → can see 200+ trades
- [ ] Header stays visible while scrolling
- [ ] Trades extend back 5 years in backtest

### Phase 4:
- [ ] Can enter multiple keys in SetupModal
- [ ] Keys saved to secrets.json
- [ ] AlphaVantage calls rotate through keys
- [ ] Sentiment fetch completes 5x faster with 5 keys

### Phase 5:
- [ ] 5-year backtest completes in ~15 seconds
- [ ] Dashboard shows 1260 trading days of data
- [ ] Can click "Reset & Simulate 5 Years" and re-run

### Phase 6:
- [ ] All 10 portfolios generate successfully
- [ ] Comparison table shows all 10 rows
- [ ] Multi-line chart shows 10 colored lines
- [ ] Can identify best/worst performing configuration

### Phase 7:
- [ ] Market Mood shows "Bullish/Bearish/etc" instead of "Calm"
- [ ] Sentiment tooltip shows article count and date
- [ ] Optional: Sentiment filter improves performance

---

## Database Impact

**Schema Changes Required:** NONE

All existing tables already support the new features:
- `paper_portfolio` - Has `session_id` for multiple portfolios
- `paper_portfolio_history` - Stores daily values for all sessions
- `paper_holdings` - Linked to session_id
- `paper_trades` - Linked to session_id

**New Session IDs Used:**
```
v9_golden_2025      (baseline, current)
v9_mc_seed1-5       (Monte Carlo variations)
v9_top5_tight       (smaller, tighter portfolio)
v9_top15_loose      (larger, looser portfolio)
v9_weekly_rebal     (different rebalance frequency)
v9_monthly_rebal    (different rebalance frequency)
```

---

## Git Status

**Modified Files (Ready to Commit):**
- `backend/routers/playground.py` - Phase 1.1 & 1.2 backend

**Pending Files (Will be modified):**
- Phase 1.3: `frontend/app/components/ModelPlayground.tsx`
- Phase 2: `frontend/app/components/PaperHoldings.tsx` or `frontend/app/page.tsx`
- Phase 3: `backend/routers/paper.py`, `frontend/app/components/PaperTrades.tsx`
- Phase 4: NEW `backend/services/alpha_vantage_pool.py`, + 3 other files
- Phase 5: `backend/scripts/seed_paper_history.py`, `frontend/app/components/SetupModal.tsx`, `backend/routers/paper.py`
- Phase 6: NEW `backend/scripts/generate_portfolio_ensemble.py`, NEW `backend/routers/portfolio_comparison.py`, NEW `frontend/app/components/PortfolioComparison.tsx`
- Phase 7: `backend/services/sentiment_data.py`, `backend/routers/dashboard.py`, others

---

## Key Dependencies

**Already Available:**
- ✅ InfoTooltip component (working, documented)
- ✅ SPY data in database (fetched by yfinance)
- ✅ Model loader with multiple versions
- ✅ AsyncSessionLocal for database queries

**Will Need:**
- Multiple AlphaVantage API keys (Phase 4)
- patience/tolerance for initial 10-portfolio generation (Phase 6, ~2-3 min)

---

## Implementation Notes

1. **API Response Format:** All endpoints return JSON with consistent error handling
2. **Async/Await:** Use `async`/`await` for database calls, properly handle exceptions
3. **Type Hints:** Python backend uses type hints, TypeScript frontend uses interfaces
4. **Error Handling:** Graceful degradation - missing data returns placeholder, not 500 error
5. **Performance:** Phase 6 will be slowest (~2-3 min for 10 backtests), can run in parallel

---

## Success Criteria

When all 7 phases complete:
- ✅ Playground shows dynamic signals and explanations per stock
- ✅ Dashboard metrics explained with tooltips
- ✅ Trades section shows full history (5 years, not 2 months)
- ✅ AlphaVantage 5x faster with multi-key support
- ✅ Default 5-year simulation instead of 1-year
- ✅ 10 portfolio comparison showing strategy variance
- ✅ Market sentiment integrated (optional)

**Total Expected Time:** 15-18 hours of implementation

---

## Next Immediate Actions

1. **For Frontend (Phase 1.3):** Use the detailed implementation guide above to add dynamic signals to ModelPlayground.tsx
2. **For Backend (Phases 2-7):** Continue with Phase 2 (Dashboard tooltips) while frontend work is in progress
3. **For Testing:** Run phases in order, test each before moving to next

---

**Generated:** 2025-12-13
**Status:** Phase 1 Backend 100% Complete, Awaiting Phase 1.3 Frontend Implementation
