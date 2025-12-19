// Shared types for DetailDrawer and sub-components

export interface HistoryItem {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    predicted_close: number | null;
    intrinsic_value: number | null;
    spy_close: number | null;
}

export interface IntrinsicBreakdown {
    eps: number;
    growth_rate: number;
    bond_yield: number;
    intrinsic_value: number;
    is_estimated: boolean;
}

export interface ChartDataPoint {
    date: string;
    fullDate: Date;
    close: number;
    prediction: number | null;
    intrinsic: number | null;
    spyClose: number | null;
    isCorrect: boolean;
    errorPct: number;
    closePct: number;
    predictionPct: number | null;
    spyPct: number | null;
    intrinsicPct: number | null;
}

export interface Forecast {
    label: string;
    price: number;
    change_pct: number;
    log_return: number;
}

export interface ForecastsData {
    ticker: string;
    current_price: number;
    confidence: number;
    forecasts: {
        "1d": Forecast;
        "1w": Forecast;
        "1m": Forecast;
        "6m": Forecast;
    };
}

export const TIME_RANGES = [
    { label: "1W", days: 7 },
    { label: "1M", days: 30 },
    { label: "3M", days: 90 },
    { label: "6M", days: 180 },
    { label: "1Y", days: 365 },
    { label: "5Y", days: 1825 },
    { label: "ALL", days: Infinity },
];

export const MAX_CHART_POINTS = 500;

export const getConfidenceExplanation = (confidence: number): string => {
    if (confidence >= 0.8) return "High - Strong directional signal. All model features align.";
    if (confidence >= 0.6) return "Moderate - Signal clear but some uncertainty in features.";
    if (confidence >= 0.4) return "Low - High market volatility or conflicting signals.";
    return "Very Low - Consider as noise. Model is unsure.";
};

export function downsampleData(data: ChartDataPoint[], maxPoints: number): ChartDataPoint[] {
    if (data.length <= maxPoints) return data;

    const bucketSize = Math.floor(data.length / maxPoints);
    const downsampled: ChartDataPoint[] = [data[0]];

    for (let i = 1; i < maxPoints - 1; i++) {
        const bucketStart = Math.floor((i * (data.length - 1)) / (maxPoints - 1));
        const bucketEnd = Math.floor(((i + 1) * (data.length - 1)) / (maxPoints - 1));

        let maxVariation = -1;
        let selectedPoint = data[bucketStart];

        for (let j = bucketStart; j < bucketEnd; j++) {
            // Prioritize points with predictions; for null predictions, use close value volatility
            const comparisonValue =
                data[j].prediction ?? (j > 0 ? data[j - 1].close : data[j].close);
            const variation = Math.abs(data[j].close - (comparisonValue as number));

            const hasPrediction = data[j].prediction !== null;
            const hasNoPrediction = selectedPoint.prediction === null;

            // Prefer points with predictions; fallback to variation in close price
            if ((hasPrediction && hasNoPrediction) || (hasPrediction === hasNoPrediction && variation > maxVariation)) {
                maxVariation = variation;
                selectedPoint = data[j];
            }
        }
        downsampled.push(selectedPoint);
    }

    downsampled.push(data[data.length - 1]);
    return downsampled;
}
