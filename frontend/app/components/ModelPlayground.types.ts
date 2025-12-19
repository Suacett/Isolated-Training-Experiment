// Shared types and constants for ModelPlayground

export const HARDCODED_MODELS: ModelInfo[] = [
    {
        version: "v9",
        display_name: "V9 Transformer Rank",
        description: "Predicts Relative Strength. Succeeds by diversifying & ranking momentum.",
        reasoning: "The Transformer architecture captures long-range dependencies in 60-day price sequences. It learns to rank stocks by relative strength vs SPY, not absolute price movement.",
        training_notes: "Trained on 500+ S&P stocks. 60-day window. 12 stationary features.",
        is_available: true,
        status: "active",
    },
    {
        version: "v8",
        display_name: "V8 Bi-LSTM Regime",
        description: "Predicts Volatility. Failed by buying 'Crash Volatility' (Falling Knives).",
        reasoning: "V8 learned to detect high-volatility regimes but couldn't distinguish between 'opportunity volatility' (bounce) and 'crisis volatility' (crash). It bought 2-sigma dips that kept dipping.",
        training_notes: "Binary classification. High buy signal recall, low precision.",
        is_available: false, // is_available indicates whether the model is available for selection/use (false for retired)
        status: "retired",
    },
    {
        version: "v7",
        display_name: "V7 LSTM Regression",
        description: "Predicts exact price. Failed due to 'Lazy Learner' bias (predicting t+1 = t).",
        reasoning: "V7 learned to minimize MSE by predicting tomorrow's price ≈ today's price. The model had no incentive to predict direction, only minimize error.",
        training_notes: "LSTM 128 hidden units. 41 features. DirectionalLoss didn't help.",
        is_available: false,
        status: "retired",
    },
];

export interface ModelInfo {
    version: string;
    display_name: string;
    description: string;
    reasoning: string;
    training_notes: string;
    is_available: boolean;
    status?: "active" | "retired";
}

export interface Prediction {
    price: number;
    change_pct: number;
    log_return: number;
    direction: "bullish" | "bearish";
}

export interface ModelPrediction {
    display_name: string;
    device: string;
    predictions: {
        "1d"?: Prediction;
        "1w"?: Prediction;
        "1m"?: Prediction;
        "6m"?: Prediction;
    } | null;
    prediction_today: {
        "1d"?: Prediction;
    } | null;
    accuracy: number | null;
    error: string | null;
}

export interface Signals {
    confidence: number;
    relative_strength: number;
    strength_label: string;
    similarity: number;
    similarity_label: string;
    market_mood: string;
    vix_proxy: number;
}

export interface Explanation {
    model: string;
    ticker: string;
    current_price: number | null;
    why_selected: string;
    key_factors: string[];
    risk_factors: string[];
    architecture: string;
}

export interface TickerComparison {
    ticker: string;
    current_price: number;
    timestamp: string;
    models: Record<string, ModelPrediction>;
    signals?: Signals;
    explanation?: Explanation;
}
