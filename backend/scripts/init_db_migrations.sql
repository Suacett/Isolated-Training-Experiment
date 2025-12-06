-- Enhanced Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
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

-- Insider Trades Table  
CREATE TABLE IF NOT EXISTS insider_trades (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    date TIMESTAMP NOT NULL,
    shares FLOAT NOT NULL,
    amount FLOAT NOT NULL,
    buy_flag INTEGER NOT NULL
);

-- Sentiment Data Table
CREATE TABLE IF NOT EXISTS sentiment_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    date TIMESTAMP NOT NULL,
    sentiment FLOAT NOT NULL,
    num_articles INTEGER NOT NULL
);

-- Add indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_insider_ticker_date ON insider_trades(ticker, date);
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_data(ticker, date);
