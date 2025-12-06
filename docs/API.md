# API Reference

The Proxmox AI Stock Predictor exposes a REST API on port 8000.

Base URL: `http://localhost:8000`

## Status Endpoints

### GET /status
Get system status and configuration.

**Response:**
```json
{
    "configured": true,
    "data_source": "yahoo_finance",
    "yahoo_finance_active": true,
    "alpha_vantage_configured": true,
    "ai_model_loaded": true,
    "scaler_loaded": true,
    "device": "cuda",
    "model_version": "v6"
}
```

### GET /status/ai
Get detailed AI model status.

**Response:**
```json
{
    "model_loaded": true,
    "model_version": "v6",
    "model_info": {
        "type": "LSTM",
        "input_features": 37,
        "hidden_dim": 64,
        "num_layers": 2,
        "window_size": 60,
        "horizons": ["1d", "1w", "1m", "6m"]
    },
    "device": "cuda",
    "ready_for_predictions": true
}
```

---

## Stock Management

### GET /stocks/
List all tickers in the watchlist.

**Response:**
```json
["AAPL", "MSFT", "SPY", "BTC-USD"]
```

### POST /stocks/
Add a ticker to the watchlist.

**Request Body:**
```json
{
    "ticker": "NVDA",
    "is_favorite": true
}
```

**Response:**
```json
{
    "message": "Added NVDA to watchlist.",
    "source": "hybrid",
    "is_favorite": true
}
```

### DELETE /stocks/{ticker}
Remove a ticker from the watchlist.

**Query Parameters:**
- `cascade` (bool): If true, also delete price data and predictions

**Response:**
```json
{
    "message": "Removed NVDA from watchlist."
}
```

### GET /stocks/browse
Get categorized list of popular stocks for browsing.

**Response:**
```json
{
    "Popular": [
        {"ticker": "AAPL", "in_watchlist": true},
        {"ticker": "MSFT", "in_watchlist": false}
    ],
    "Tech": [...],
    "Finance": [...]
}
```

---

## Dashboard

### GET /dashboard
Get summary for all watchlist items.

**Response:**
```json
[
    {
        "ticker": "AAPL",
        "current_price": 189.50,
        "prediction": 192.30,
        "intrinsic_value": 175.00,
        "signal": "BUY",
        "is_favorite": true
    }
]
```

### GET /dashboard/{ticker}
Get detailed data for a single ticker including historical prices and predictions.

**Response:**
```json
{
    "history": [
        {
            "date": "2024-01-15",
            "open": 185.20,
            "high": 190.00,
            "low": 184.50,
            "close": 189.50,
            "predicted_close": 188.75,
            "intrinsic_value": 175.00
        }
    ],
    "last_sync_date": "2024-01-15"
}
```

---

## Predictions

### GET /forecasts/{ticker}
Get multi-horizon AI forecasts.

**Response:**
```json
{
    "ticker": "AAPL",
    "current_price": 189.50,
    "confidence": 0.72,
    "forecasts": {
        "1d": {
            "label": "Tomorrow",
            "price": 191.20,
            "change_pct": 0.89,
            "log_return": 0.0089
        },
        "1w": {
            "label": "Next Week",
            "price": 194.50,
            "change_pct": 2.64
        },
        "1m": {...},
        "6m": {...}
    }
}
```

### GET /predictions/{ticker}
Get prediction history for a ticker.

**Query Parameters:**
- `limit` (int): Maximum predictions to return (default: 50)

**Response:**
```json
[
    {
        "id": 1,
        "ticker": "AAPL",
        "prediction_date": "2024-01-14T10:00:00",
        "target_date": "2024-01-15T00:00:00",
        "horizon": "1d",
        "predicted_value": 0.0089,
        "predicted_price": 191.20,
        "actual_value": 0.0075,
        "current_price": 189.50,
        "is_correct": true,
        "confidence": 0.72
    }
]
```

### GET /predictions/{ticker}/stats
Get prediction accuracy statistics.

**Response:**
```json
{
    "ticker": "AAPL",
    "total_predictions": 100,
    "validated_predictions": 90,
    "correct_predictions": 49,
    "accuracy": 54.4,
    "by_horizon": {
        "1d": {"total": 90, "validated": 85, "correct": 46, "accuracy": 54.1},
        "1w": {"total": 10, "validated": 5, "correct": 3, "accuracy": 60.0}
    }
}
```

### POST /predictions/{ticker}/backfill
Generate historical predictions for backtesting.

**Query Parameters:**
- `days` (int): Number of days to backfill (default: 90)

**Response:** List of backfill data points with predicted vs actual comparison.

---

## Data Ingestion

### POST /ingest/{ticker}
Ingest data for a single ticker.

**Query Parameters:**
- `mode` (str): "full" or "daily" (default: "full")

**Response:**
```json
{
    "message": "Ingestion complete for AAPL",
    "source": "hybrid",
    "ingestion": {"rows": 1000},
    "prediction": {"ticker": "AAPL", "prediction": 191.20},
    "backfill": null
}
```

### POST /ingest/all
Ingest data for all watchlist tickers.

**Query Parameters:**
- `mode` (str): "daily" or "full" (default: "daily")

**Response:**
```json
{
    "message": "Ingestion and prediction complete",
    "source": "yahoo_finance",
    "ingestion": {...},
    "predictions": [...],
    "errors": null
}
```

---

## Settings

### POST /settings/keys
Save API keys.

**Request Body:**
```json
{
    "ALPHA_VANTAGE_KEY": "your_key_here"
}
```

### DELETE /settings/keys
Reset all API keys.

---

## Model Playground

### GET /models
List available AI models.

### GET /settings/playground
Get playground mode status.

### POST /settings/playground
Enable/disable model playground.

**Query Parameters:**
- `enabled` (bool): Enable or disable

### GET /playground/compare/{ticker}
Compare predictions from all loaded models.

**Query Parameters:**
- `accuracy_days` (int): Days for accuracy calculation (default: 10)

---

## Error Responses

All endpoints may return error responses:

```json
{
    "detail": "Error message describing the issue"
}
```

Common HTTP status codes:
- `400`: Bad request (invalid parameters)
- `404`: Resource not found (unknown ticker)
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)
