"use client";

import React, { useState, useEffect, useMemo } from "react";
import { X, Loader2, Info, CheckCircle, XCircle, TrendingUp, TrendingDown, BarChart3, Table as TableIcon, Eye, EyeOff, AlertTriangle } from "lucide-react";
import {
    ComposedChart,
    Line,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Brush,
    Legend,
    ReferenceLine,
} from "recharts";
import { Asset } from "./AssetTable";

const API_BASE = "http://localhost:8000";

// Confidence explanation helper
const getConfidenceExplanation = (confidence: number): string => {
    if (confidence >= 0.8) return "High - Strong directional signal. All model features align.";
    if (confidence >= 0.6) return "Moderate - Signal clear but some uncertainty in features.";
    if (confidence >= 0.4) return "Low - High market volatility or conflicting signals.";
    return "Very Low - Consider as noise. Model is unsure.";
};

interface DetailDrawerProps {
    asset: Asset | null;
    onClose: () => void;
}

interface HistoryItem {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    predicted_close: number | null;
    intrinsic_value: number | null;
    spy_close: number | null;
}

interface IntrinsicBreakdown {
    eps: number;
    growth_rate: number;
    bond_yield: number;
    intrinsic_value: number;
    is_estimated: boolean;
}

interface ChartDataPoint {
    date: string;
    fullDate: Date;
    close: number;
    prediction: number | null;
    intrinsic: number | null;
    spyClose: number | null;
    isCorrect: boolean;
    errorPct: number;
    // Percentage change from first data point
    closePct: number;
    predictionPct: number | null;
    spyPct: number | null;
    intrinsicPct: number | null;
}

interface Forecast {
    label: string;
    price: number;
    change_pct: number;
    log_return: number;
}

interface ForecastsData {
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

// Time range options
const TIME_RANGES = [
    { label: "1W", days: 7 },
    { label: "1M", days: 30 },
    { label: "3M", days: 90 },
    { label: "6M", days: 180 },
    { label: "1Y", days: 365 },
    { label: "5Y", days: 1825 },
    { label: "ALL", days: Infinity },
];

// Downsampling algorithm - optimized for chart performance
// Maximum points to render (prevents lag with large datasets)
const MAX_CHART_POINTS = 500;

function downsampleData(data: ChartDataPoint[], maxPoints: number): ChartDataPoint[] {
    if (data.length <= maxPoints) return data;

    const bucketSize = Math.floor(data.length / maxPoints);
    const downsampled: ChartDataPoint[] = [data[0]]; // Always keep first point

    for (let i = 1; i < maxPoints - 1; i++) {
        const bucketStart = Math.floor(i * bucketSize);
        const bucketEnd = Math.min(bucketStart + bucketSize, data.length);

        // Pick the point with highest price variation in each bucket
        let maxVariation = 0;
        let selectedPoint = data[bucketStart];

        for (let j = bucketStart; j < bucketEnd; j++) {
            const variation = Math.abs(data[j].close - (data[j].prediction || data[j].close));
            if (variation > maxVariation) {
                maxVariation = variation;
                selectedPoint = data[j];
            }
        }
        downsampled.push(selectedPoint);
    }

    downsampled.push(data[data.length - 1]); // Always keep last point
    return downsampled;
}

// Memoized Chart Component
const MemoizedChart = React.memo(({
    data,
    showPercentage,
    visibleLines,
    hasIntrinsic,
    timeRange
}: {
    data: ChartDataPoint[];
    showPercentage: boolean;
    visibleLines: { price: boolean; prediction: boolean; spy: boolean; intrinsic: boolean };
    hasIntrinsic: boolean;
    timeRange: string;
}) => {
    const yDataKey = showPercentage ? (d: ChartDataPoint) => d.closePct : (d: ChartDataPoint) => d.close;
    const predDataKey = showPercentage ? (d: ChartDataPoint) => d.predictionPct : (d: ChartDataPoint) => d.prediction;
    const spyDataKey = showPercentage ? (d: ChartDataPoint) => d.spyPct : (d: ChartDataPoint) => d.spyClose;
    const intrinsicDataKey = showPercentage ? (d: ChartDataPoint) => d.intrinsicPct : (d: ChartDataPoint) => d.intrinsic;

    if (!data || data.length === 0) return null;

    return (
        <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis
                    dataKey="date"
                    stroke="#52525b"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    minTickGap={50}
                    tickFormatter={(value, index) => {
                        // Optimize X-Axis dates based on range
                        const rangeDef = TIME_RANGES.find(r => r.label === timeRange);
                        const isLongRange = rangeDef && rangeDef.days > 365;
                        const isMediumRange = rangeDef && rangeDef.days > 90;

                        if (isLongRange) {
                            // Show Full Year for long ranges
                            const d = data[index];
                            if (d && index % 12 === 0) return d.fullDate.getFullYear().toString();
                            return "";
                        }
                        if (isMediumRange) {
                            // Show Month/Year
                            const d = data[index];
                            if (d && index % 5 === 0) return `${d.fullDate.getMonth() + 1}/${d.fullDate.getFullYear().toString().slice(2)}`;
                            return "";
                        }
                        return value;
                    }}
                />
                <YAxis
                    stroke="#52525b"
                    fontSize={10}
                    domain={['auto', 'auto']}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(val) => showPercentage ? `${val.toFixed(0)}%` : `$${val.toFixed(0)}`}
                    width={50}
                />
                <Tooltip content={<CustomTooltip showPercentage={showPercentage} />} />

                {showPercentage && <ReferenceLine y={0} stroke="#52525b" strokeDasharray="3 3" />}

                {/* S&P 500 comparison (show first so it's behind) */}
                {visibleLines.spy && (
                    <Line
                        type="monotone"
                        dataKey={spyDataKey}
                        stroke="#3b82f6"
                        strokeWidth={1.5}
                        dot={false}
                        strokeDasharray="4 4"
                        name="S&P 500"
                    />
                )}

                {/* Intrinsic Value (only if available and enabled) */}
                {hasIntrinsic && visibleLines.intrinsic && (
                    <Line
                        type="monotone"
                        dataKey={intrinsicDataKey}
                        stroke="#a78bfa"
                        strokeWidth={2}
                        dot={false}
                        name="Intrinsic"
                    />
                )}

                {/* AI Prediction */}
                {visibleLines.prediction && (
                    <Line
                        type="monotone"
                        dataKey={predDataKey}
                        stroke="#f472b6"
                        strokeWidth={2}
                        strokeDasharray="6 3"
                        dot={false}
                        name="AI Prediction"
                    />
                )}

                {/* Actual Price (on top) */}
                {visibleLines.price && (
                    <Area
                        type="monotone"
                        dataKey={yDataKey}
                        stroke="#10b981"
                        strokeWidth={2}
                        fill="url(#priceGradient)"
                        name="Price"
                    />
                )}

                <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                </defs>

                <Brush
                    dataKey="date"
                    height={30}
                    stroke="#52525b"
                    fill="#18181b"
                    tickFormatter={() => ""}
                />
            </ComposedChart>
        </ResponsiveContainer>
    );
});
MemoizedChart.displayName = "MemoizedChart";

// Custom Tooltip
const CustomTooltip = ({ active, payload, showPercentage }: {
    active?: boolean;
    payload?: Array<{ payload: ChartDataPoint }>;
    showPercentage?: boolean;
}) => {
    if (active && payload && payload.length) {
        const d = payload[0].payload;
        const diff = d.prediction !== null ? d.prediction - d.close : 0;
        const diffPct = d.prediction !== null ? (diff / d.close) * 100 : 0;

        return (
            <div className="bg-zinc-900/95 border border-zinc-700 p-3 rounded-lg shadow-xl text-xs backdrop-blur min-w-[200px]">
                <div className="text-zinc-400 mb-2 font-bold border-b border-zinc-700 pb-1 flex justify-between">
                    <span>{d.date}</span>
                    <span className="text-zinc-500 font-normal">{d.fullDate.getFullYear()}</span>
                </div>
                <div className="space-y-1">
                    <div className="flex justify-between gap-4">
                        <span className="text-emerald-400">Price:</span>
                        <span className="font-mono text-white">
                            {showPercentage ? `${d.closePct.toFixed(2)}%` : `$${d.close.toFixed(2)}`}
                        </span>
                    </div>
                    {d.prediction !== null && (
                        <>
                            <div className="flex justify-between gap-4">
                                <span className="text-pink-400">AI Pred:</span>
                                <span className="font-mono text-white">
                                    {showPercentage ? `${d.predictionPct?.toFixed(2)}%` : `$${d.prediction.toFixed(2)}`}
                                </span>
                            </div>
                            <div className="flex justify-between gap-4 pt-1 border-t border-zinc-700/50 mt-1">
                                <span className="text-zinc-400">Diff:</span>
                                <span className={`font-mono ${diff >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {diff >= 0 ? '+' : ''}{diff.toFixed(2)} ({diffPct.toFixed(2)}%)
                                </span>
                            </div>
                            <div className="text-[10px] text-zinc-500 mt-1 italic">
                                {diff > 0 ? "Prediction was higher than actual." : "Prediction was lower than actual."}
                            </div>
                        </>
                    )}
                    {d.spyClose !== null && (
                        <div className="flex justify-between gap-4 pt-1 border-t border-zinc-700/50 mt-1">
                            <span className="text-blue-400">S&P 500:</span>
                            <span className="font-mono text-white">
                                {showPercentage ? `${d.spyPct?.toFixed(2)}%` : `$${d.spyClose.toFixed(2)}`}
                            </span>
                        </div>
                    )}
                    {d.intrinsic !== null && d.intrinsic > 0 && (
                        <div className="flex justify-between gap-4">
                            <span className="text-purple-400">Intrinsic:</span>
                            <span className="font-mono text-white">
                                {showPercentage ? `${d.intrinsicPct?.toFixed(2)}%` : `$${d.intrinsic.toFixed(2)}`}
                            </span>
                        </div>
                    )}
                </div>
            </div>
        );
    }
    return null;
};

// Forecast Card Component
const ForecastCard = ({
    label,
    price,
    changePct,
    infoText,
    isResult,
    wasCorrect,
    showPercentage,
    referencePrice
}: {
    label: string;
    price: number | null;
    changePct: number | null;
    infoText: string;
    isResult?: boolean;
    wasCorrect?: boolean;
    showPercentage?: boolean;
    referencePrice?: number;
}) => {
    const isPositive = (changePct ?? 0) >= 0;

    // Calculate display value based on mode
    let displayValue = "N/A";
    if (price !== null) {
        if (showPercentage && referencePrice) {
            const relPct = ((price - referencePrice) / referencePrice) * 100;
            displayValue = `${relPct > 0 ? '+' : ''}${relPct.toFixed(2)}%`;
        } else {
            displayValue = `$${price.toFixed(2)}`;
        }
    }

    return (
        <div className={`flex-shrink-0 w-28 bg-zinc-800/50 rounded-lg p-3 border ${isResult
            ? (wasCorrect ? 'border-emerald-500/50' : 'border-rose-500/50')
            : 'border-zinc-700/50'
            }`}>
            <div className="flex items-center gap-1 mb-1">
                <span className="text-[10px] text-zinc-500 uppercase font-medium">{label}</span>
                <div className="relative inline-block group">
                    <Info size={10} className="text-zinc-600 cursor-help" />
                    <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-48 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[500] pointer-events-none">
                        <span className="text-[10px] text-zinc-300">{infoText}</span>
                    </div>
                </div>
            </div>
            {price !== null ? (
                <>
                    <div className="text-sm font-bold text-white">{displayValue}</div>
                    <div className={`text-xs font-medium flex items-center gap-0.5 ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                        {isPositive ? '+' : ''}{changePct?.toFixed(2)}%
                    </div>
                    {isResult && (
                        <div className={`mt-1 text-[10px] font-bold ${wasCorrect ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {wasCorrect ? '✓ CORRECT' : '✗ WRONG'}
                        </div>
                    )}
                </>
            ) : (
                <div className="text-sm text-zinc-500">N/A</div>
            )}
        </div>
    );
};

// Compact AI Model Info Component (small tooltip-style)
const AIModelInfo = ({ confidence, accuracy }: { confidence: number; accuracy: number | null }) => (
    <div className="relative inline-block group ml-2">
        <div className="flex items-center gap-1 px-2 py-1 bg-pink-900/30 border border-pink-700/50 rounded text-[10px] text-pink-400 cursor-help">
            <Info size={10} />
            <span>AI Model</span>
        </div>
        <div className="absolute left-0 top-full mt-1 w-64 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[200] pointer-events-none">
            <div className="text-[10px] text-zinc-300 space-y-1">
                <div><strong>Model:</strong> LSTM + MC Dropout (50 samples)</div>
                <div><strong>Input:</strong> 60 days × 37 features</div>
                <div><strong>Horizons:</strong> 1d, 1w, 1m, 6m</div>
                <div><strong>Confidence:</strong> {(confidence * 100).toFixed(0)}% - {getConfidenceExplanation(confidence)}</div>
                {accuracy !== null && <div><strong>Accuracy:</strong> {accuracy}% directional</div>}
            </div>
        </div>
    </div>
);

// Compact Intrinsic Value Info Component
const IntrinsicInfo = ({ breakdown }: { breakdown: IntrinsicBreakdown | null }) => {
    if (!breakdown) return null;

    return (
        <div className="relative inline-block group ml-2">
            <div className="flex items-center gap-1 px-2 py-1 bg-purple-900/30 border border-purple-700/50 rounded text-[10px] text-purple-400 cursor-help">
                <Info size={10} />
                <span>Intrinsic Value</span>
            </div>
            <div className="absolute left-0 top-full mt-1 w-56 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[200] pointer-events-none">
                <div className="text-[10px] text-zinc-300 space-y-1">
                    <div className="font-medium text-purple-400 mb-1">Graham Formula</div>
                    <div className="grid grid-cols-2 gap-1">
                        <span>EPS:</span><span className="font-mono">${breakdown.eps?.toFixed(2) || 'N/A'}</span>
                        <span>Growth:</span><span className="font-mono">{((breakdown.growth_rate || 0) * 100).toFixed(1)}%</span>
                        <span>Bond Yield:</span><span className="font-mono">{breakdown.bond_yield?.toFixed(2) || 'N/A'}%</span>
                        <span>Value:</span><span className="font-mono text-purple-400">${breakdown.intrinsic_value?.toFixed(2) || 'N/A'}</span>
                    </div>
                    {breakdown.is_estimated && (
                        <div className="text-amber-400 mt-1 flex items-center gap-1">
                            <AlertTriangle size={8} />
                            Estimated values
                        </div>
                    )}
                    <div className="text-zinc-500 mt-1 border-t border-zinc-700 pt-1">
                        V = EPS × (8.5 + 2g) × (4.4/Y)
                    </div>
                </div>
            </div>
        </div>
    );
};

export default function DetailDrawer({ asset, onClose }: DetailDrawerProps) {
    const [rawData, setRawData] = useState<ChartDataPoint[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [forecasts, setForecasts] = useState<ForecastsData | null>(null);
    const [accuracy, setAccuracy] = useState<number | null>(null);
    const [selectedPrediction, setSelectedPrediction] = useState<ChartDataPoint | null>(null);
    const [hasIntrinsic, setHasIntrinsic] = useState(false);
    const [modelError, setModelError] = useState<string | null>(null);
    const [intrinsicBreakdown, setIntrinsicBreakdown] = useState<IntrinsicBreakdown | null>(null);

    // View controls
    const [viewMode, setViewMode] = useState<"chart" | "table">("chart");
    const [showPercentage, setShowPercentage] = useState(false);
    const [timeRange, setTimeRange] = useState<string>("ALL");
    const [visibleLines, setVisibleLines] = useState({ price: true, prediction: true, spy: true, intrinsic: true });

    // Fetch data when asset changes
    useEffect(() => {
        if (!asset) return;

        const fetchData = async () => {
            setIsLoading(true);
            try {
                // Fetch history and forecasts in parallel
                const [historyRes, forecastRes] = await Promise.all([
                    fetch(`${API_BASE}/dashboard/${asset.ticker}`),
                    fetch(`${API_BASE}/forecasts/${asset.ticker}`).catch(() => null)
                ]);

                if (!historyRes.ok) throw new Error("Failed to fetch history");

                const historyJson = await historyRes.json();
                const history: HistoryItem[] = historyJson.history || [];

                // Get intrinsic breakdown
                if (historyJson.intrinsic_breakdown) {
                    setIntrinsicBreakdown(historyJson.intrinsic_breakdown);
                }

                // Check if intrinsic values are available
                const hasIntrinsicValue = history.some(h => h.intrinsic_value !== null && h.intrinsic_value > 0);
                setHasIntrinsic(hasIntrinsicValue);

                // Get first values for percentage calculation
                const firstClose = history.length > 0 ? history[0].close : 1;
                const firstSpy = history.find(h => h.spy_close !== null)?.spy_close || 1;
                const firstIntrinsic = history.find(h => h.intrinsic_value !== null)?.intrinsic_value || firstClose;

                // Transform to chart data
                const chartData: ChartDataPoint[] = history.map((item, index, arr) => {
                    let isCorrect = false;
                    if (index > 0 && item.predicted_close !== null) {
                        const prevClose = arr[index - 1].close;
                        const predictedDirection = item.predicted_close > prevClose;
                        const actualDirection = item.close > prevClose;
                        isCorrect = predictedDirection === actualDirection;
                    }

                    const errorPct = item.predicted_close !== null
                        ? Math.abs((item.predicted_close - item.close) / item.close) * 100
                        : 0;

                    return {
                        date: new Date(item.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
                        fullDate: new Date(item.date),
                        close: item.close,
                        prediction: item.predicted_close,
                        intrinsic: item.intrinsic_value,
                        spyClose: item.spy_close,
                        isCorrect,
                        errorPct,
                        closePct: 0, // Calculated dynamically in displayData
                        predictionPct: 0,
                        spyPct: 0,
                        intrinsicPct: 0,
                    };
                });

                setRawData(chartData);

                // Calculate accuracy based on visible data or all data?
                // User wants "Directional model accuracy" for the stock.
                // Let's calculate it based on ALL loaded history for now, or maybe visible range?
                // The previous code used all loaded history.
                const withPredictions = chartData.filter((d, i) => i > 0 && d.prediction !== null);
                if (withPredictions.length > 0) {
                    const correctCount = withPredictions.filter(d => d.isCorrect).length;
                    setAccuracy(Math.round((correctCount / withPredictions.length) * 100));
                }

                // Fetch forecasts
                if (forecastRes) {
                    if (forecastRes.status === 503) {
                        setModelError("AI Model not loaded. Check Settings for model status.");
                        setForecasts(null);
                    } else if (forecastRes.ok) {
                        const forecastData = await forecastRes.json();
                        setForecasts(forecastData);
                        setModelError(null);
                    }
                }

            } catch (err) {
                console.error("Failed to fetch data:", err);
                setRawData([]);
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [asset]);

    // Filter data by time range and apply downsampling for performance
    const displayData = useMemo(() => {
        const rangeDef = TIME_RANGES.find(r => r.label === timeRange);
        let filtered = rawData;

        if (rangeDef && rangeDef.days !== Infinity) {
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - rangeDef.days);
            filtered = rawData.filter(d => d.fullDate >= cutoffDate);
        }

        // Downsample for performance
        const downsampled = downsampleData(filtered, MAX_CHART_POINTS);

        // Recalculate Accuracy based on VISIBLE data
        const visiblePredictions = downsampled.filter((d, i) => i > 0 && d.prediction !== null);
        if (visiblePredictions.length > 0) {
            const correctCount = visiblePredictions.filter(d => d.isCorrect).length;
            setAccuracy(Math.round((correctCount / visiblePredictions.length) * 100));
        } else {
            setAccuracy(0);
        }

        // Calculate percentages relative to the first visible point
        if (downsampled.length > 0) {
            const first = downsampled[0];
            const firstClose = first.close;
            const firstSpy = first.spyClose || firstClose; // Fallback
            const firstIntrinsic = first.intrinsic || firstClose; // Fallback

            return downsampled.map(d => ({
                ...d,
                closePct: ((d.close - firstClose) / firstClose) * 100,
                predictionPct: d.prediction ? ((d.prediction - firstClose) / firstClose) * 100 : null,
                spyPct: d.spyClose ? ((d.spyClose - firstSpy) / firstSpy) * 100 : null,
                intrinsicPct: d.intrinsic ? ((d.intrinsic - firstIntrinsic) / firstIntrinsic) * 100 : null,
            }));
        }

        return downsampled;
    }, [rawData, timeRange]);

    // Helper to get reference price for % calculation
    const referencePrice = useMemo(() => {
        if (displayData.length > 0) return displayData[0].close;
        return 0;
    }, [displayData]);

    // Yesterday's result
    const yesterdayResult = useMemo(() => {
        const withPreds = rawData.filter((d, i) => i > 0 && rawData[i - 1].prediction !== null);
        if (withPreds.length === 0) return null;
        const latest = withPreds[withPreds.length - 1];
        const latestIdx = rawData.indexOf(latest);
        const prevDay = rawData[latestIdx - 1];
        if (!prevDay?.prediction) return null;

        const predMove = ((prevDay.prediction - prevDay.close) / prevDay.close) * 100;
        const actMove = ((latest.close - prevDay.close) / prevDay.close) * 100;

        return {
            price: latest.close,
            predicted: prevDay.prediction,
            changePct: actMove,
            wasCorrect: latest.isCorrect,
            predDir: predMove >= 0 ? 'UP' : 'DOWN',
            actDir: actMove >= 0 ? 'UP' : 'DOWN',
            predMove,
            actMove,
            date: latest.date
        };
    }, [rawData]);

    const toggleLine = (line: keyof typeof visibleLines) => {
        setVisibleLines(prev => ({ ...prev, [line]: !prev[line] }));
    };

    if (!asset) return null;

    return (
        <div className="fixed inset-y-0 right-0 w-[900px] bg-zinc-950 border-l border-zinc-800 shadow-2xl z-50 flex flex-col">
            {/* Header */}
            <div className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-900/50 backdrop-blur">
                <div>
                    <h2 className="text-xl font-bold text-white flex items-center gap-3">
                        {asset.ticker}
                        <span className="text-xs font-normal text-zinc-400 px-2 py-0.5 rounded border border-zinc-700 bg-zinc-800">
                            {displayData.length}{rawData.length > MAX_CHART_POINTS ? ` / ${rawData.length}` : ""} POINTS
                        </span>
                    </h2>
                    <div className="flex gap-4 mt-1 text-sm items-center">
                        <span className="text-zinc-400">
                            Current: <span className="text-white font-mono">${asset.price.toFixed(2)}</span>
                        </span>
                        {forecasts && (
                            <span className="text-zinc-400">
                                Confidence: <span className={`font-mono ${forecasts.confidence > 0.6 ? 'text-emerald-400' : 'text-yellow-400'}`}>
                                    {(forecasts.confidence * 100).toFixed(0)}%
                                </span>
                            </span>
                        )}
                        {forecasts && (
                            <AIModelInfo confidence={forecasts.confidence} accuracy={accuracy} />
                        )}
                        {intrinsicBreakdown && asset.isFavorite && (
                            <IntrinsicInfo breakdown={intrinsicBreakdown} />
                        )}
                        {['BTC', 'ETH', 'LTC', 'DOGE'].some(c => asset.ticker.includes(c)) && (
                            <div className="relative inline-block group ml-2">
                                <div className="flex items-center gap-1 px-2 py-1 bg-amber-900/30 border border-amber-700/50 rounded text-[10px] text-amber-400 cursor-help">
                                    <AlertTriangle size={10} />
                                    <span>Crypto Warning</span>
                                </div>
                                <div className="absolute left-0 top-full mt-1 w-64 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[200] pointer-events-none">
                                    <div className="text-[10px] text-zinc-300">
                                        <strong>Experimental:</strong> This model was primarily trained on stock market data. Predictions for crypto assets may be less accurate due to different market dynamics.
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-white transition-colors"
                >
                    <X size={20} />
                </button>
            </div>

            {/* Main Content */}
            <div className="flex-1 p-4 overflow-hidden flex flex-col">
                {isLoading ? (
                    <div className="flex-1 flex items-center justify-center">
                        <Loader2 className="animate-spin text-zinc-500" size={32} />
                    </div>
                ) : displayData.length === 0 ? (
                    <div className="flex-1 flex items-center justify-center text-zinc-500">
                        <p>No historical data available. Click "Sync Data" to fetch.</p>
                    </div>
                ) : (
                    <>
                        {/* Model Error State */}
                        {modelError && (
                            <div className="bg-amber-900/20 border border-amber-700/50 rounded-lg p-3 mb-4 text-amber-400 text-sm flex items-center gap-2">
                                <AlertTriangle size={16} />
                                {modelError}
                            </div>
                        )}

                        {/* Accuracy Box */}
                        <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-3 mb-4 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-zinc-800 rounded-lg">
                                    <BarChart3 size={20} className="text-zinc-400" />
                                </div>
                                <div>
                                    <div className="text-sm font-medium text-white">Model Accuracy</div>
                                    <div className="text-xs text-zinc-500">Based on last {displayData.length} days</div>
                                </div>
                            </div>
                            <div className="flex gap-6 text-right">
                                <div>
                                    <div className="text-xs text-zinc-500 uppercase font-medium">Directional</div>
                                    <div className={`text-lg font-bold font-mono ${(accuracy || 0) > 55 ? 'text-emerald-400' : 'text-zinc-300'
                                        }`}>
                                        {accuracy}%
                                    </div>
                                </div>
                                <div>
                                    <div className="text-xs text-zinc-500 uppercase font-medium">Avg Error</div>
                                    <div className="text-lg font-bold font-mono text-zinc-300">
                                        {displayData.length > 0
                                            ? (displayData.reduce((acc, d) => acc + d.errorPct, 0) / displayData.length).toFixed(2)
                                            : "0.00"}%
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Forecast Cards */}
                        <div className="flex gap-2 mb-4 pb-2 flex-wrap">
                            {yesterdayResult && (
                                <ForecastCard
                                    label="Yesterday"
                                    price={yesterdayResult.price}
                                    changePct={yesterdayResult.changePct}
                                    infoText={`AI predicted ${yesterdayResult.predDir} (${yesterdayResult.predMove > 0 ? '+' : ''}${yesterdayResult.predMove.toFixed(2)}%), actual was ${yesterdayResult.actDir} (${yesterdayResult.actMove > 0 ? '+' : ''}${yesterdayResult.actMove.toFixed(2)}%).`}
                                    isResult
                                    wasCorrect={yesterdayResult.wasCorrect}
                                    showPercentage={showPercentage}
                                    referencePrice={referencePrice}
                                />
                            )}
                            {forecasts && (
                                <>
                                    <ForecastCard
                                        label="Tomorrow"
                                        price={forecasts.forecasts["1d"].price}
                                        changePct={forecasts.forecasts["1d"].change_pct}
                                        infoText="AI prediction for tomorrow's close based on current market data and LSTM model analysis."
                                        showPercentage={showPercentage}
                                        referencePrice={referencePrice}
                                    />
                                    <ForecastCard
                                        label="1 Week"
                                        price={forecasts.forecasts["1w"].price}
                                        changePct={forecasts.forecasts["1w"].change_pct}
                                        infoText="AI prediction for next week's close. Longer horizons have higher uncertainty."
                                        showPercentage={showPercentage}
                                        referencePrice={referencePrice}
                                    />
                                    <ForecastCard
                                        label="1 Month"
                                        price={forecasts.forecasts["1m"].price}
                                        changePct={forecasts.forecasts["1m"].change_pct}
                                        infoText="AI prediction for next month. Note: Accuracy decreases with longer time horizons."
                                        showPercentage={showPercentage}
                                        referencePrice={referencePrice}
                                    />
                                    <ForecastCard
                                        label="6 Months"
                                        price={forecasts.forecasts["6m"].price}
                                        changePct={forecasts.forecasts["6m"].change_pct}
                                        infoText="Long-term AI forecast. Use with caution - long-term predictions are highly speculative."
                                        showPercentage={showPercentage}
                                        referencePrice={referencePrice}
                                    />
                                </>
                            )}
                        </div>

                        {/* Controls */}
                        <div className="flex justify-between items-center mb-3 flex-wrap gap-2">
                            {/* View Toggle */}
                            <div className="flex gap-1 bg-zinc-800/50 rounded-lg p-1">
                                <button
                                    onClick={() => setViewMode("chart")}
                                    className={`px-3 py-1 text-xs rounded flex items-center gap-1 ${viewMode === "chart" ? "bg-zinc-700 text-white" : "text-zinc-400"}`}
                                >
                                    <BarChart3 size={14} /> Chart
                                </button>
                                <button
                                    onClick={() => setViewMode("table")}
                                    className={`px-3 py-1 text-xs rounded flex items-center gap-1 ${viewMode === "table" ? "bg-zinc-700 text-white" : "text-zinc-400"}`}
                                >
                                    <TableIcon size={14} /> Table
                                </button>
                            </div>

                            {/* Time Range */}
                            <div className="flex gap-1 bg-zinc-800/50 rounded-lg p-1">
                                {TIME_RANGES.map(r => (
                                    <button
                                        key={r.label}
                                        onClick={() => setTimeRange(r.label)}
                                        className={`px-2 py-1 text-xs rounded ${timeRange === r.label ? "bg-zinc-700 text-white" : "text-zinc-400"}`}
                                    >
                                        {r.label}
                                    </button>
                                ))}
                            </div>

                            {/* Percentage Toggle */}
                            <button
                                onClick={() => setShowPercentage(!showPercentage)}
                                className={`px-3 py-1 text-xs rounded border ${showPercentage ? "bg-blue-500/20 text-blue-400 border-blue-500/50" : "bg-zinc-800/50 text-zinc-400 border-zinc-700"}`}
                            >
                                {showPercentage ? "% Mode" : "$ Mode"}
                            </button>
                        </div>

                        {/* Line Toggles */}
                        <div className="flex gap-2 mb-3 flex-wrap">
                            {[
                                { key: 'price', label: 'Price', color: 'emerald' },
                                { key: 'prediction', label: 'AI Prediction', color: 'pink' },
                                { key: 'spy', label: 'S&P 500', color: 'blue' },
                                ...(hasIntrinsic ? [{ key: 'intrinsic', label: 'Intrinsic', color: 'purple' }] : [])
                            ].map(({ key, label, color }) => (
                                <button
                                    key={key}
                                    onClick={() => toggleLine(key as keyof typeof visibleLines)}
                                    className={`px-2 py-1 text-xs rounded flex items-center gap-1 transition-all ${visibleLines[key as keyof typeof visibleLines]
                                        ? `bg-${color}-500/20 text-${color}-400 border border-${color}-500/50`
                                        : 'bg-zinc-800/50 text-zinc-500 border border-zinc-700'
                                        }`}
                                    style={{
                                        backgroundColor: visibleLines[key as keyof typeof visibleLines]
                                            ? color === 'emerald' ? 'rgba(16, 185, 129, 0.2)'
                                                : color === 'pink' ? 'rgba(244, 114, 182, 0.2)'
                                                    : color === 'blue' ? 'rgba(59, 130, 246, 0.2)'
                                                        : 'rgba(167, 139, 250, 0.2)'
                                            : undefined,
                                        color: visibleLines[key as keyof typeof visibleLines]
                                            ? color === 'emerald' ? '#10b981'
                                                : color === 'pink' ? '#f472b6'
                                                    : color === 'blue' ? '#3b82f6'
                                                        : '#a78bfa'
                                            : undefined,
                                    }}
                                >
                                    {visibleLines[key as keyof typeof visibleLines] ? <Eye size={12} /> : <EyeOff size={12} />}
                                    {label}
                                </button>
                            ))}
                        </div>

                        {/* Chart or Table */}
                        {viewMode === "chart" ? (
                            <div className="flex-1 bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 min-h-0">
                                <MemoizedChart
                                    data={displayData}
                                    showPercentage={showPercentage}
                                    visibleLines={visibleLines}
                                    hasIntrinsic={hasIntrinsic}
                                    timeRange={timeRange}
                                />
                            </div>
                        ) : (
                            <div className="flex-1 overflow-auto bg-zinc-900/30 rounded-xl border border-zinc-800">
                                <table className="w-full text-xs">
                                    <thead className="sticky top-0 bg-zinc-900">
                                        <tr className="text-zinc-500 uppercase">
                                            <th className="p-2 text-left">Date</th>
                                            <th className="p-2 text-right">Price</th>
                                            <th className="p-2 text-right">AI Pred</th>
                                            <th className="p-2 text-right">S&P 500</th>
                                            <th className="p-2 text-right">Error</th>
                                            <th className="p-2 text-center">Dir</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-zinc-800">
                                        {displayData.filter(d => d.prediction !== null).reverse().slice(0, 100).map((d, i, arr) => {
                                            // Calculate previous close for direction explanation
                                            const prevIdx = arr.length - 1 - i - 1;
                                            const prevClose = prevIdx >= 0 && arr[prevIdx] ? arr[prevIdx].close : d.close;
                                            const predictedDir = d.prediction! > prevClose ? 'UP' : 'DOWN';
                                            const actualDir = d.close > prevClose ? 'UP' : 'DOWN';
                                            const whyText = d.isCorrect
                                                ? `Predicted ${predictedDir}, Actual ${actualDir}. Direction matched.`
                                                : `Predicted ${predictedDir}, Actual ${actualDir}. Direction mismatch.`;

                                            return (
                                                <tr
                                                    key={i}
                                                    className="hover:bg-zinc-800/50 cursor-pointer"
                                                    onClick={() => setSelectedPrediction(d)}
                                                >
                                                    <td className="p-2 text-zinc-400">{d.date}</td>
                                                    <td className="p-2 text-right font-mono text-emerald-400">
                                                        {showPercentage ? `${d.closePct?.toFixed(2)}%` : `$${d.close.toFixed(2)}`}
                                                    </td>
                                                    <td className="p-2 text-right font-mono text-pink-400">
                                                        {showPercentage ? `${d.predictionPct?.toFixed(2)}%` : `$${d.prediction?.toFixed(2)}`}
                                                    </td>
                                                    <td className="p-2 text-right font-mono text-blue-400">
                                                        {d.spyClose ? (showPercentage ? `${d.spyPct?.toFixed(2)}%` : `$${d.spyClose.toFixed(2)}`) : '-'}
                                                    </td>
                                                    <td className="p-2 text-right font-mono text-zinc-400">{d.errorPct.toFixed(2)}%</td>
                                                    <td className="p-2 text-center relative group">
                                                        {d.isCorrect ? (
                                                            <CheckCircle className="inline text-emerald-500" size={14} />
                                                        ) : (
                                                            <XCircle className="inline text-rose-500" size={14} />
                                                        )}
                                                        {/* Why Tooltip */}
                                                        <div className="absolute bottom-full right-0 mb-1 w-48 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[100] pointer-events-none text-xs text-left">
                                                            <div className={`font-bold ${d.isCorrect ? 'text-emerald-400' : 'text-rose-400'}`}>
                                                                {d.isCorrect ? '✓ CORRECT' : '✗ WRONG'}
                                                            </div>
                                                            <div className="text-zinc-300 mt-1">{whyText}</div>
                                                        </div>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        )}

                        {/* Accuracy Stream */}
                        <div className="mt-3 h-20 bg-zinc-900/30 rounded-xl border border-zinc-800 p-2">
                            <div className="flex justify-between items-center mb-1">
                                <span className="text-[10px] font-medium text-zinc-500 uppercase">
                                    Accuracy: {accuracy}% (Last {displayData.length} days) - Hover for details
                                </span>
                            </div>
                            <div className="flex gap-px h-6 w-full overflow-visible rounded relative">
                                {displayData.map((d, i) => (
                                    <div key={i} className="flex-1 relative group/bar">
                                        <div
                                            className={`w-full h-full cursor-pointer transition-all ${d.isCorrect
                                                ? "bg-emerald-500/60 group-hover/bar:bg-emerald-500"
                                                : "bg-rose-500/60 group-hover/bar:bg-rose-500"
                                                }`}
                                        />
                                        {/* Hover tooltip */}
                                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-36 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover/bar:opacity-100 group-hover/bar:visible transition-all z-[100] pointer-events-none text-xs">
                                            <div className="font-bold text-white mb-1">{d.date}</div>
                                            <div className="text-zinc-400">Price: <span className="text-emerald-400">
                                                {showPercentage ? `${d.closePct?.toFixed(2)}%` : `$${d.close.toFixed(2)}`}
                                            </span></div>
                                            {d.prediction !== null && (
                                                <div className="text-zinc-400">Pred: <span className="text-pink-400">
                                                    {showPercentage ? `${d.predictionPct?.toFixed(2)}%` : `$${d.prediction.toFixed(2)}`}
                                                </span></div>
                                            )}
                                            <div className={`mt-1 font-medium ${d.isCorrect ? "text-emerald-400" : "text-rose-400"}`}>
                                                {d.isCorrect ? "✓ Correct" : "✗ Wrong"}
                                            </div>
                                            <div className="text-[9px] text-zinc-500 mt-1 italic">
                                                {d.isCorrect ? "Direction matched prediction." : "Direction mismatch."}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </>
                )}
            </div>

            {/* Detail Modal */}
            {selectedPrediction && (
                <div
                    className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 backdrop-blur-sm"
                    onClick={() => setSelectedPrediction(null)}
                >
                    <div
                        className="bg-zinc-900 border border-zinc-700 rounded-xl p-6 max-w-md w-full shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex justify-between items-start mb-4">
                            <h3 className="text-lg font-bold text-white">Prediction Detail</h3>
                            <button
                                onClick={() => setSelectedPrediction(null)}
                                className="text-zinc-500 hover:text-white transition-colors"
                            >
                                <X size={20} />
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div className="flex justify-between items-center pb-3 border-b border-zinc-800">
                                <span className="text-zinc-400">Date</span>
                                <span className="font-mono text-white">{selectedPrediction.date}</span>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-zinc-800/50 p-3 rounded-lg">
                                    <div className="text-xs text-zinc-500 mb-1">Predicted</div>
                                    <div className="text-lg font-mono text-pink-400">
                                        ${selectedPrediction.prediction?.toFixed(2) ?? "N/A"}
                                    </div>
                                </div>
                                <div className="bg-zinc-800/50 p-3 rounded-lg">
                                    <div className="text-xs text-zinc-500 mb-1">Actual</div>
                                    <div className="text-lg font-mono text-emerald-400">
                                        ${selectedPrediction.close.toFixed(2)}
                                    </div>
                                </div>
                            </div>

                            {selectedPrediction.spyClose && (
                                <div className="flex justify-between items-center">
                                    <span className="text-zinc-400">S&P 500</span>
                                    <span className="font-mono text-blue-400">${selectedPrediction.spyClose.toFixed(2)}</span>
                                </div>
                            )}

                            <div className="flex justify-between items-center">
                                <span className="text-zinc-400">Error Margin</span>
                                <span className="font-mono text-zinc-300">{selectedPrediction.errorPct.toFixed(2)}%</span>
                            </div>

                            <div className={`p-3 rounded-lg flex items-center justify-center gap-2 font-bold ${selectedPrediction.isCorrect
                                ? "bg-emerald-500/20 text-emerald-400"
                                : "bg-rose-500/20 text-rose-400"
                                }`}>
                                {selectedPrediction.isCorrect ? <CheckCircle size={20} /> : <XCircle size={20} />}
                                {selectedPrediction.isCorrect ? "CORRECT DIRECTION" : "WRONG DIRECTION"}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

