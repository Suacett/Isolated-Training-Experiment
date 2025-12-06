"use client";

import { useState, useEffect } from "react";
import { Brain, TrendingUp, TrendingDown, AlertTriangle, RefreshCw, X } from "lucide-react";

const API_BASE = "http://localhost:8000";

interface PredictionDebug {
    ticker: string;
    current_price: number;
    confidence: number;
    forecasts: {
        "1d": { label: string; price: number; change_pct: number; log_return: number };
        "1w": { label: string; price: number; change_pct: number; log_return: number };
        "1m": { label: string; price: number; change_pct: number; log_return: number };
        "6m": { label: string; price: number; change_pct: number; log_return: number };
    };
}

interface BackfillSummary {
    ticker: string;
    total_predictions: number;
    correct_predictions: number;
    accuracy: number;
    recent_predictions: Array<{
        date: string;
        actual_price: number;
        predicted_price: number;
        correct_direction: boolean;
        error_percentage: number;
    }>;
}

interface AIPredictionPanelProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function AIPredictionPanel({ isOpen, onClose }: AIPredictionPanelProps) {
    const [tickers, setTickers] = useState<string[]>([]);
    const [selectedTicker, setSelectedTicker] = useState<string>("");
    const [prediction, setPrediction] = useState<PredictionDebug | null>(null);
    const [backfillSummary, setBackfillSummary] = useState<BackfillSummary | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [modelVersion, setModelVersion] = useState<string>("Unknown");

    // Fetch available tickers and model info
    useEffect(() => {
        if (isOpen) {
            // Fetch tickers
            fetch(`${API_BASE}/dashboard`)
                .then((res) => res.json())
                .then((data) => {
                    const tickerList = data.map((item: { ticker: string }) => item.ticker);
                    setTickers(tickerList);
                    if (tickerList.length > 0 && !selectedTicker) {
                        setSelectedTicker(tickerList[0]);
                    }
                })
                .catch(console.error);

            // Fetch model status
            fetch(`${API_BASE}/status/ai`)
                .then((res) => res.json())
                .then((data) => {
                    if (data.model_version) {
                        setModelVersion(data.model_version);
                    }
                })
                .catch(console.error);
        }
    }, [isOpen, selectedTicker]);

    // Fetch prediction data when ticker changes
    useEffect(() => {
        if (!selectedTicker) return;

        const fetchPrediction = async () => {
            setIsLoading(true);
            setError(null);

            try {
                // Fetch forecasts
                const forecastRes = await fetch(`${API_BASE}/forecasts/${selectedTicker}`);
                if (forecastRes.ok) {
                    const forecastData = await forecastRes.json();
                    setPrediction(forecastData);
                } else {
                    setPrediction(null);
                }

                // Fetch recent backfill for accuracy
                const backfillRes = await fetch(`${API_BASE}/predictions/${selectedTicker}/backfill/full`, {
                    method: "POST"
                });
                if (backfillRes.ok) {
                    const backfillData = await backfillRes.json();
                    const recent = backfillData.slice(-20);
                    const correctCount = backfillData.filter((p: { correct_direction: boolean }) => p.correct_direction).length;
                    setBackfillSummary({
                        ticker: selectedTicker,
                        total_predictions: backfillData.length,
                        correct_predictions: correctCount,
                        accuracy: backfillData.length > 0 ? (correctCount / backfillData.length) * 100 : 0,
                        recent_predictions: recent.reverse()
                    });
                }
            } catch (err) {
                setError("Failed to fetch prediction data");
                console.error(err);
            } finally {
                setIsLoading(false);
            }
        };

        fetchPrediction();
    }, [selectedTicker]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-zinc-900 border border-zinc-700 rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
                    <div className="flex items-center gap-3">
                        <Brain className="text-purple-400" size={24} />
                        <div>
                            <h2 className="text-xl font-bold text-white">AI Model Inspector</h2>
                            <p className="text-xs text-zinc-400">Active Model: <span className="text-emerald-400 font-mono">{modelVersion}</span></p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-zinc-800 rounded-full transition-colors"
                    >
                        <X className="text-zinc-400" size={20} />
                    </button>
                </div>

                {/* Controls */}
                <div className="px-6 py-3 border-b border-zinc-800 bg-zinc-900/50">
                    <div className="flex items-center gap-4">
                        <label className="text-sm text-zinc-400">Ticker:</label>
                        <select
                            value={selectedTicker}
                            onChange={(e) => setSelectedTicker(e.target.value)}
                            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500"
                        >
                            {tickers.map((t) => (
                                <option key={t} value={t}>{t}</option>
                            ))}
                        </select>
                        {isLoading && <RefreshCw className="animate-spin text-zinc-500" size={16} />}
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {error ? (
                        <div className="text-rose-400 text-center py-8">{error}</div>
                    ) : (
                        <>
                            {/* Model Predictions */}
                            {prediction && (
                                <div>
                                    <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                                        <TrendingUp className="text-emerald-400" size={18} />
                                        Current Predictions for {selectedTicker}
                                    </h3>
                                    <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-4">
                                        <div className="grid grid-cols-2 gap-4 mb-4">
                                            <div>
                                                <span className="text-zinc-500 text-sm">Current Price</span>
                                                <p className="text-2xl font-bold text-white">${prediction.current_price.toFixed(2)}</p>
                                            </div>
                                            <div>
                                                <span className="text-zinc-500 text-sm">Model Confidence</span>
                                                <p className="text-2xl font-bold text-purple-400">{(prediction.confidence * 100).toFixed(1)}%</p>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-4 gap-3">
                                            {(["1d", "1w", "1m", "6m"] as const).map((horizon) => {
                                                const forecast = prediction.forecasts[horizon];
                                                const isUp = forecast.change_pct > 0;
                                                return (
                                                    <div key={horizon} className="bg-zinc-900 rounded-lg p-3 border border-zinc-700">
                                                        <div className="text-xs text-zinc-500 uppercase mb-1">{forecast.label}</div>
                                                        <div className="text-lg font-bold text-white">${forecast.price.toFixed(2)}</div>
                                                        <div className={`text-sm flex items-center gap-1 ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
                                                            {isUp ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                                            {isUp ? "+" : ""}{forecast.change_pct.toFixed(2)}%
                                                        </div>
                                                        <div className="text-xs text-zinc-600 mt-1">
                                                            log_return: {forecast.log_return.toFixed(6)}
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Accuracy Summary */}
                            {backfillSummary && (
                                <div>
                                    <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                                        <AlertTriangle className={`${backfillSummary.accuracy >= 50 ? "text-emerald-400" : "text-amber-400"}`} size={18} />
                                        Historical Accuracy Analysis
                                    </h3>
                                    <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-4">
                                        <div className="grid grid-cols-3 gap-4 mb-4">
                                            <div>
                                                <span className="text-zinc-500 text-sm">Total Predictions</span>
                                                <p className="text-2xl font-bold text-white">{backfillSummary.total_predictions.toLocaleString()}</p>
                                            </div>
                                            <div>
                                                <span className="text-zinc-500 text-sm">Correct Direction</span>
                                                <p className="text-2xl font-bold text-emerald-400">{backfillSummary.correct_predictions.toLocaleString()}</p>
                                            </div>
                                            <div>
                                                <span className="text-zinc-500 text-sm">Accuracy Rate</span>
                                                <p className={`text-2xl font-bold ${backfillSummary.accuracy >= 50 ? "text-emerald-400" : "text-rose-400"}`}>
                                                    {backfillSummary.accuracy.toFixed(1)}%
                                                </p>
                                            </div>
                                        </div>

                                        {/* Model Bias Warning */}
                                        {backfillSummary.accuracy < 50 && (
                                            <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 mb-4">
                                                <div className="flex items-start gap-2">
                                                    <AlertTriangle className="text-amber-400 mt-0.5" size={16} />
                                                    <div className="text-sm">
                                                        <p className="text-amber-400 font-medium">Model Bias Detected</p>
                                                        <p className="text-amber-300/70 mt-1">
                                                            This model shows a <strong>bullish bias</strong> - it consistently predicts upward price movements.
                                                            This is likely because the training data was primarily from an upward-trending market period (SPY 2012-2024).
                                                            The model may need retraining with more balanced data.
                                                        </p>
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        {/* Recent Predictions Table */}
                                        <div>
                                            <h4 className="text-sm font-medium text-zinc-400 mb-2">Recent Predictions (Last 20)</h4>
                                            <div className="overflow-x-auto">
                                                <table className="w-full text-xs">
                                                    <thead>
                                                        <tr className="text-zinc-500 border-b border-zinc-700">
                                                            <th className="text-left py-2 px-2">Date</th>
                                                            <th className="text-right py-2 px-2">Predicted</th>
                                                            <th className="text-right py-2 px-2">Actual</th>
                                                            <th className="text-right py-2 px-2">Error</th>
                                                            <th className="text-center py-2 px-2">Direction</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {backfillSummary.recent_predictions.map((p, i) => (
                                                            <tr key={i} className="border-b border-zinc-800 hover:bg-zinc-800/50">
                                                                <td className="py-2 px-2 text-zinc-400">
                                                                    {new Date(p.date).toLocaleDateString()}
                                                                </td>
                                                                <td className="py-2 px-2 text-right font-mono text-pink-400">
                                                                    ${p.predicted_price.toFixed(2)}
                                                                </td>
                                                                <td className="py-2 px-2 text-right font-mono text-white">
                                                                    ${p.actual_price.toFixed(2)}
                                                                </td>
                                                                <td className="py-2 px-2 text-right font-mono text-zinc-400">
                                                                    {p.error_percentage.toFixed(1)}%
                                                                </td>
                                                                <td className="py-2 px-2 text-center">
                                                                    {p.correct_direction ? (
                                                                        <span className="text-emerald-400">✓</span>
                                                                    ) : (
                                                                        <span className="text-rose-400">✗</span>
                                                                    )}
                                                                </td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Model Info */}
                            <div className="bg-zinc-800/30 border border-zinc-700/50 rounded-xl p-4">
                                <h4 className="text-sm font-medium text-zinc-400 mb-2">Model Information</h4>
                                <div className="text-xs text-zinc-500 space-y-1">
                                    <p>• <strong>Architecture:</strong> LSTM (Long Short-Term Memory) Neural Network</p>
                                    <p>• <strong>Input:</strong> 60-day sliding window of 37 technical features</p>
                                    <p>• <strong>Output:</strong> Log returns (converted to price predictions)</p>
                                    <p>• <strong>Training Data:</strong> Historical SPY data (2012-2024)</p>
                                    <p className="text-amber-400/70 mt-2">
                                        ⚠️ Direction accuracy for stocks other than SPY may be lower due to model being primarily trained on index data.
                                    </p>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
