# LSTM Stock Predictor - Training and Usage Guide

This guide explains how to train and use the full 39-feature LSTM model with legacy training data.

## Overview

### Model Hierarchy & Evolution

The system maintains multiple models for different trading strategies:

1.  **V9 (Transformer-Rank)**: **PRIMARY / STATE OF THE ART**. Uses a Transformer architecture to predict relative performance across the S&P 500. Best for long/short portfolio construction.
2.  **V7 (BiLSTM-Attention)**: **STABLE / DIRECTIONAL**. Uses temporal attention to predict price targets with high directional accuracy. Best for individual stock entry/exit timing.
3.  **V6-V2 (Legacy LSTM)**: Baseline models primarily used for regression testing and comparative analysis.

**4 Prediction Horizons (V7 & Legacy):**
- `Target_1d`: 1 day ahead
- `Target_1w`: 5 days (1 week) ahead
- `Target_1m`: 21 days (1 month) ahead
- `Target_6m`: 126 days (6 months) ahead

## Training the Model

### Step 1: Install Dependencies

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Or in Docker
docker compose exec backend pip install -r requirements.txt
```

### Step 2: Run Training

**V9 Training (STATE OF THE ART):**
```bash
# 1. Fetch Full S&P 500 Data (Required for Ranking)
docker exec proxmox_stock_backend python -m scripts.fetch_training_data --sp500

# 2. Train Transformer Model (Uses Sliding Window + Lazy Loading)
docker exec proxmox_stock_backend python -m scripts.train_model_v9

# 3. Backtest the Ranking Strategy
docker exec proxmox_stock_backend python -m scripts.backtest_v9_portfolio
```

**V7 Training (Stable):**
```bash
# Fetch training data (59 tickers: indices, sectors, top stocks)
docker exec proxmox_stock_backend python -m scripts.fetch_training_data

# Train V7 model (BiLSTM + Attention, ~2 hours)
docker exec proxmox_stock_backend python -m scripts.train_model_v7

# Activate V7 as production model
docker exec proxmox_stock_backend cp /app/models/lstm_model_v7.pth /app/models/lstm_model_v2.pth
docker exec proxmox_stock_backend cp /app/models/scaler_v7.pkl /app/models/scaler_v2.pkl
docker compose restart backend
```

**Legacy V6 Training:**
```bash
docker exec proxmox_stock_backend python -m scripts.train_model_v6
```

This will:
1. Load all CSV files from `legacy/LSTM_AI_Stock_Predictor/TrainingData/indicators_data/raw/stocksData/`
2. Process 39 features for each stock
3. Fit a StandardScaler on training data
4. Train the LSTM model with:
   - Time-based weighting (recent data weighted higher)
   - Early stopping (patience=25 epochs)
   - Validation monitoring
5. Save:
   - Trained model: `backend/models/lstm_model.pth`
   - Fitted scaler: `backend/models/scaler.pkl`

**Training Parameters:**
- Window size: 60 timesteps
- Batch size: 128
- Max epochs: 200
- Learning rate: 0.001
- Train/Val split: 80%/16%

### Step 3: Verify Training

Check the log file:
```bash
tail -f backend/training.log
```

Look for:
- "Training Complete!"
- Final validation loss
- Model saved confirmation

### Production Selection
By default, `bootstrap.py` auto-resolves the highest versioned model found in `/app/models`. If `transformer_v9.pth` and `scaler_v9.pkl` exist, the system boots in **V9 (Ranking) mode**. If only V7 files exist, it uses **V7 (Directional) mode**.

## Using the Trained Model

### 1. Model Initialization

The model loads automatically on backend startup from `backend/main.py`:

```python
model = LSTMModel(input_dim=39, window_size=60)
model.load("backend/models/lstm_model.pth")

scaler = FeatureScaler()
scaler.load("backend/models/scaler.pkl")
```

### 2. Making Predictions

To generate predictions, you need:

**A. Historical OHLCV Data (60 days)**
- Fetched from Alpaca or Alpha Vantage
- Stored in TimescaleDB

**B. Insider Trading Data (Optional but Recommended)**
```python
from services.insider_data import insider_collector

insider_df = await insider_collector.get_daily_insider_summary(
    ticker="AAPL",
    start_date=datetime.now() - timedelta(days=90)
)
```

**C. Sentiment Data (Optional but Recommended)**
```python
from services.sentiment_data import sentiment_collector

sentiment_df = await sentiment_collector.get_recent_sentiment(
    ticker="AAPL",
    days=90
)
```

**D. Process Features**
```python
from services.feature_engineering import process_stock_data

# Combine all data
processed_df = process_stock_data(
    df=ohlcv_df,
    insider_df=insider_df,
    sentiment_df=sentiment_df,
    create_targets=False  # No targets for inference
)

# Get last 60 days
recent_data = processed_df.tail(60)

# Scale using fitted scaler
feature_cols = get_model_input_features()
X = scaler.transform(recent_data[feature_cols].values)

# Reshape for model: (1, 60, 39)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

# Predict
predictions = model(X_tensor)  # Returns (1, 4) - four horizons
```

## Database Schema Updates

Run database migration to create new tables:

```sql
-- Predictions table (enhanced)
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    target_date TIMESTAMP NOT NULL,
    horizon VARCHAR NOT NULL,
    predicted_value FLOAT NOT NULL,
    actual_value FLOAT,
    current_price FLOAT NOT NULL,
    is_correct BOOLEAN,
    confidence FLOAT
);

-- Insider trades table
CREATE TABLE insider_trades (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    date TIMESTAMP NOT NULL,
    shares FLOAT NOT NULL,
    amount FLOAT NOT NULL,
    buy_flag INTEGER NOT NULL
);

-- Sentiment data table
CREATE TABLE sentiment_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    date TIMESTAMP NOT NULL,
    sentiment FLOAT NOT NULL,
    num_articles INTEGER NOT NULL
);
```

## Prediction Tracking

### Saving Predictions

```python
from services.db import save_prediction
from datetime import datetime, timedelta

await save_prediction(
    ticker="AAPL",
    prediction_date=datetime.now(),
    target_date=datetime.now() + timedelta(days=1),
    horizon="1d",
    predicted_value=0.02,  # 2% predicted return
    current_price=180.50,
    confidence=0.75
)
```

### Validating Predictions

Create a background task to check past predictions:

```python
from services.db import get_pending_predictions, update_prediction_actual

# Get predictions where target_date has passed
pending = await get_pending_predictions()

for pred in pending:
    # Fetch actual price at target_date
    actual_price = await fetch_historical_price(pred.ticker, pred.target_date)

    if actual_price:
        actual_return = np.log(actual_price / pred.current_price)
        is_correct = (
            (pred.predicted_value > 0 and actual_return > 0) or
            (pred.predicted_value < 0 and actual_return < 0)
        )

        await update_prediction_actual(
            pred.id,
            actual_value=actual_return,
            is_correct=is_correct
        )
```

## Frontend Integration

### Display Past Predictions

Add to `frontend/app/components/DetailDrawer.tsx`:

```typescript
interface Prediction {
  id: number;
  prediction_date: string;
  target_date: string;
  horizon: string;
  predicted_value: number;
  actual_value?: number;
  is_correct?: boolean;
  confidence?: number;
}

// Fetch predictions
const predictions = await fetch(`/api/predictions/${ticker}`).then(r => r.json());

// Display
<div className="prediction-history">
  <h3>Past Predictions</h3>
  {predictions.map(pred => (
    <div key={pred.id} className={pred.is_correct ? 'correct' : 'incorrect'}>
      <span>{pred.horizon}</span>
      <span>Predicted: {(pred.predicted_value * 100).toFixed(2)}%</span>
      {pred.actual_value && (
        <span>Actual: {(pred.actual_value * 100).toFixed(2)}%</span>
      )}
      <span>{pred.is_correct ? '✓' : '✗'}</span>
    </div>
  ))}
</div>
```

## Key Files Created

**Feature Engineering:**
- `backend/services/feature_engineering.py` - 39 features computation
- `backend/services/insider_data.py` - SEC Form 4 scraper
- `backend/services/sentiment_data.py` - Alpha Vantage news sentiment

**Training:**
- `backend/services/training.py` - Complete training pipeline
- `backend/services/scaler.py` - StandardScaler wrapper
- `backend/scripts/train_model.py` - Training script

**Database:**
- Updated `backend/services/db.py` - New models and queries

**Model:**
- Updated `backend/services/lstm_model.py` - Now accepts 39 features

## Troubleshooting

### Issue: "Scaler not fitted"
**Solution:** Train the model first to generate `backend/models/scaler.pkl`

### Issue: "Feature dimension mismatch"
**Solution:** Ensure all 39 features are computed. Check for missing insider/sentiment data.

### Issue: "Model weights not found"
**Solution:** Run `python backend/scripts/train_model.py` to train and save weights

### Issue: "SEC EDGAR rate limiting"
**Solution:** Insider data collection is slow (SEC rate limits). Cache results in database.


## Accuracy Benchmarks

| Model | Training | SPY Accuracy | Bias | Notes |
|-------|----------|--------------|------|-------|
| Model | Training | Accuracy/IC | Notes |
|-------|----------|-------------|-------|
| **V9** | S&P 500, Transformer | **IC ~0.09** | **Rank Prediction** | Best for portfolio selection |
| **V7** | 59 tickers, BiLSTM | **52.6% Acc** | -23.7% Bias | Directional prediction |
| V6 | All stocks, clean data | 54.0% Acc | +8% Bias | Baseline |
| V3 | 100 stocks | ~54% | N/A | Legacy |

**Note:** V7 is more defensive/bearish - better at vetoing bad trades.
## Performance Notes

**Training Time:**
- ~31 stocks from legacy data
- ~200 epochs with early stopping
- Expect 1-3 hours on CPU, 15-30 minutes on GPU

**Inference Time:**
- Single prediction: <100ms
- Batch predictions (100 stocks): ~2-5 seconds

**Data Collection:**
- Insider data: ~30-60 seconds per stock (SEC rate limits)
- Sentiment data: ~5-10 minutes for 90 days (Alpha Vantage limits)
- OHLCV data: <1 second (Alpaca is fast)

## Next Steps

1. **Train the model**: `python backend/scripts/train_model.py`
2. **Update database schema**: Run migrations for new tables
3. **Implement prediction validation**: Background task to check accuracy
4. **Update frontend**: Display prediction history and accuracy
5. **Add retraining schedule**: Weekly/monthly retraining on new data

## References

- Legacy implementation: `legacy/LSTM_AI_Stock_Predictor/forecast.ipynb`
- Feature engineering: `legacy/LSTM_AI_Stock_Predictor/TrainingData/processor.py`
- Backtesting: `legacy/LSTM_AI_Stock_Predictor/forecasting_backtest_Predictor_v2.py`
