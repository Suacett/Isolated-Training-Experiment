"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
    Brain,
    RefreshCw,
    Cpu,
    Info,
    TrendingUp,
    TrendingDown,
    AlertTriangle,
    ChevronDown,
    ChevronUp,
    Zap,
    HardDrive,
} from "lucide-react";

const API_BASE = "http://localhost:8000";

interface ModelInfo {
    version: string;
    display_name: string;
    description: string;
    reasoning: string;
    training_notes: string;
    is_available: boolean;
}

interface Prediction {
    price: number;
    change_pct: number;
    log_return: number;
    direction: "bullish" | "bearish";
}

interface ModelPrediction {
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

interface TickerComparison {
    ticker: string;
    current_price: number;
    timestamp: string;
    models: Record<string, ModelPrediction>;
}

interface PlaygroundPageProps {
    onBack: () => void;
}

export default function ModelPlayground({ onBack }: PlaygroundPageProps) {
    const [isEnabled, setIsEnabled] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [isToggling, setIsToggling] = useState(false);
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [comparisons, setComparisons] = useState<Record<string, TickerComparison>>({});
    const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
    const [expandedModel, setExpandedModel] = useState<string | null>(null);
    const [loaderStatus, setLoaderStatus] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const [accuracyDays, setAccuracyDays] = useState(10);

    // Fetch initial status
    useEffect(() => {
        fetchStatus();
        fetchModels();
    }, []);

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_BASE}/settings/playground`);
            const data = await res.json();
            setIsEnabled(data.enabled);
            setLoaderStatus(data.loader_status);

            if (data.enabled) {
                // Fetch comparisons if enabled
                fetchComparisons();
            }
        } catch (e) {
            setError("Failed to fetch playground status");
        } finally {
            setIsLoading(false);
        }
    };

    const fetchModels = async () => {
        try {
            const res = await fetch(`${API_BASE}/models`);
            const data = await res.json();
            setModels(data.models || []);
        } catch (e) {
            console.error("Failed to fetch models:", e);
        }
    };

    const fetchComparisons = useCallback(async () => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE}/playground/compare_all?accuracy_days=${accuracyDays}`);
            if (res.ok) {
                const data = await res.json();
                setComparisons(data.tickers || {});

                // Auto-select first ticker
                const tickers = Object.keys(data.tickers || {});
                if (tickers.length > 0 && !selectedTicker) {
                    setSelectedTicker(tickers[0]);
                }
            }
        } catch (e) {
            setError("Failed to fetch predictions");
        } finally {
            setIsLoading(false);
        }
    }, [accuracyDays, selectedTicker]); // Added selectedTicker to dependencies for auto-selection logic

    const togglePlayground = async () => {
        setIsToggling(true);
        try {
            const res = await fetch(`${API_BASE}/settings/playground?enabled=${!isEnabled}`, {
                method: "POST",
            });
            const data = await res.json();
            setIsEnabled(data.enabled);
            setLoaderStatus(data.loader_status);

            if (data.enabled) {
                // Wait a bit then fetch comparisons
                setTimeout(fetchComparisons, 500);
            } else {
                setComparisons({});
            }
        } catch (e) {
            setError("Failed to toggle playground");
        } finally {
            setIsToggling(false);
        }
    };

    // Reload when accuracy days changes (only if enabled)
    useEffect(() => {
        if (isEnabled) {
            fetchComparisons();
        }
    }, [accuracyDays, isEnabled, fetchComparisons]); // Added fetchComparisons to dependencies

    const horizons = ["1d", "1w", "1m", "6m"] as const;
    const horizonLabels = {
        "1d": "Tomorrow",
        "1w": "1 Week",
        "1m": "1 Month",
        "6m": "6 Months",
    };

    return (
        <div className="min-h-screen bg-[#0A0A0F] text-white">
            {/* Header */}
            <header className="bg-[#12121A]/80 border-b border-white/[0.08] sticky top-0 z-40 backdrop-blur-md">
                <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={onBack}
                            className="text-zinc-400 hover:text-white transition-colors"
                        >
                            ← Back
                        </button>
                        <div className="flex items-center gap-2">
                            <Brain className="text-amber-400" size={24} />
                            <h1 className="text-xl font-bold">Model Playground</h1>
                        </div>
                    </div>

                    <div className="flex items-center gap-4">
                        {/* Accuracy Window Selector */}
                        {isEnabled && (
                            <div className="flex items-center gap-2 bg-zinc-800 rounded-lg p-1 border border-zinc-700">
                                <span className="text-xs text-zinc-400 pl-2">Backtest:</span>
                                <select
                                    value={accuracyDays}
                                    onChange={(e) => setAccuracyDays(Number(e.target.value))}
                                    className="bg-transparent text-sm text-white font-medium focus:outline-none cursor-pointer hover:text-amber-400"
                                >
                                    <option value={10}>10 Days</option>
                                    <option value={30}>30 Days</option>
                                    <option value={60}>60 Days</option>
                                    <option value={90}>90 Days</option>
                                </select>
                            </div>
                        )}

                        {/* Refresh Button */}
                        {isEnabled && (
                            <button
                                onClick={fetchComparisons}
                                disabled={isLoading}
                                className="p-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 transition-colors text-zinc-400 hover:text-white"
                            >
                                <RefreshCw size={18} className={isLoading ? "animate-spin" : ""} />
                            </button>
                        )}

                        {/* Enable/Disable Toggle */}
                        <button
                            onClick={togglePlayground}
                            disabled={isToggling}
                            className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${isEnabled
                                ? "bg-amber-500 hover:bg-amber-400 text-[#0A0A0F] hover:shadow-[0_0_20px_rgba(245,158,11,0.4)]"
                                : "bg-zinc-800 hover:bg-zinc-700 text-zinc-300"
                                }`}
                        >
                            {isToggling ? (
                                <RefreshCw size={16} className="animate-spin" />
                            ) : (
                                <Zap size={16} />
                            )}
                            {isEnabled ? "Enabled" : "Disabled"}
                        </button>
                    </div>
                </div>
            </header>

            {/* Warning Banner */}
            {!isEnabled && (
                <div className="bg-amber-500/10 border-b border-amber-500/20 px-6 py-3">
                    <div className="max-w-7xl mx-auto flex items-center gap-3 text-amber-400">
                        <AlertTriangle size={20} />
                        <span className="text-sm">
                            <strong>Warning:</strong> Enabling Model Playground will load multiple AI models into memory.
                            This may require 4-8GB of VRAM/RAM. Models will automatically fall back to system RAM if GPU memory is insufficient.
                        </span>
                    </div>
                </div>
            )}

            {/* Loader Status Bar */}
            {isEnabled && loaderStatus && (
                <div className="bg-zinc-900/50 border-b border-zinc-800 px-6 py-2">
                    <div className="max-w-7xl mx-auto flex items-center gap-6 text-xs">
                        <span className="text-zinc-500">
                            <strong className="text-zinc-400">{loaderStatus.loaded_count}</strong> Models Loaded
                        </span>
                        {loaderStatus.gpu_available && (
                            <span className="text-zinc-500 flex items-center gap-1">
                                <Cpu size={12} className="text-blue-400" />
                                GPU Free: <strong className="text-blue-400">{loaderStatus.gpu_memory_free_gb}GB</strong>
                            </span>
                        )}
                        <div className="flex items-center gap-3 ml-auto">
                            {Object.entries(loaderStatus.models || {}).map(([version, info]: [string, any]) => (
                                <span
                                    key={version}
                                    className={`px-2 py-0.5 rounded text-[10px] font-medium ${info.device === "cuda"
                                        ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                                        : "bg-zinc-700 text-zinc-400"
                                        }`}
                                >
                                    {version.toUpperCase()} • {info.device === "cuda" ? "GPU" : "CPU"}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            <main className="max-w-7xl mx-auto px-6 py-6">
                {!isEnabled ? (
                    /* Disabled State - Show Model Info */
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {models.map((model) => (
                            <div
                                key={model.version}
                                className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4 hover:border-zinc-700 transition-colors"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <div>
                                        <h3 className="font-bold text-white">{model.display_name}</h3>
                                        <p className="text-xs text-zinc-500">{model.version}</p>
                                    </div>
                                    {model.is_available ? (
                                        <span className="text-[10px] bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30">
                                            AVAILABLE
                                        </span>
                                    ) : (
                                        <span className="text-[10px] bg-zinc-700 text-zinc-500 px-2 py-0.5 rounded">
                                            NOT FOUND
                                        </span>
                                    )}
                                </div>

                                <p className="text-sm text-zinc-400 mb-3">{model.description}</p>

                                <button
                                    onClick={() => setExpandedModel(expandedModel === model.version ? null : model.version)}
                                    className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1"
                                >
                                    {expandedModel === model.version ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                    Why this model?
                                </button>

                                {expandedModel === model.version && (
                                    <div className="mt-3 p-3 bg-zinc-950 rounded-lg border border-zinc-800 text-xs text-zinc-400 whitespace-pre-wrap">
                                        {model.reasoning}
                                        <div className="mt-2 pt-2 border-t border-zinc-800 text-zinc-500">
                                            <strong>Notes:</strong> {model.training_notes}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                ) : isLoading ? (
                    /* Loading State */
                    <div className="flex flex-col items-center justify-center py-20">
                        <RefreshCw className="animate-spin text-purple-400 mb-4" size={48} />
                        <p className="text-zinc-400">Loading predictions from all models...</p>
                    </div>
                ) : (
                    /* Comparison View */
                    <div className="space-y-6">
                        {/* Ticker Selector */}
                        <div className="flex gap-2 flex-wrap">
                            {Object.keys(comparisons).map((ticker) => (
                                <button
                                    key={ticker}
                                    onClick={() => setSelectedTicker(ticker)}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${selectedTicker === ticker
                                        ? "bg-amber-500 text-[#0A0A0F]"
                                        : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                                        }`}
                                >
                                    {ticker}
                                </button>
                            ))}
                        </div>

                        {/* Comparison Table */}
                        {selectedTicker && comparisons[selectedTicker] && (
                            <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden">
                                <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
                                    <div>
                                        <h2 className="text-xl font-bold">{selectedTicker}</h2>
                                        <p className="text-sm text-zinc-500">
                                            Current Price: <span className="text-emerald-400 font-mono">${comparisons[selectedTicker].current_price?.toFixed(2)}</span>
                                        </p>
                                    </div>
                                </div>

                                <div className="overflow-x-auto">
                                    <table className="w-full text-sm">
                                        <thead className="bg-zinc-900">
                                            <tr className="text-zinc-500 uppercase text-xs">
                                                <th className="p-3 text-left">Model</th>
                                                <th className="p-3 text-center">Device</th>
                                                <th className="p-3 text-center bg-zinc-800/30 text-zinc-400">Accuracy ({accuracyDays}d)</th>
                                                <th className="p-3 text-center bg-zinc-800/30 text-zinc-400">Signal (1d)</th>
                                                <th className="p-3 text-center bg-zinc-800/30 text-zinc-400">Yesterday (Pred)</th>
                                                {horizons.map((h) => (
                                                    <th key={h} className="p-3 text-center">{horizonLabels[h]}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-zinc-800">
                                            {Object.entries(comparisons[selectedTicker].models || {}).map(([version, model]) => (
                                                <tr key={version} className="hover:bg-zinc-800/50">
                                                    <td className="p-3">
                                                        <div className="font-medium text-white">{model.display_name}</div>
                                                        <div className="text-xs text-zinc-500">{version}</div>
                                                    </td>
                                                    <td className="p-3 text-center">
                                                        <span className={`px-2 py-0.5 rounded text-[10px] ${model.device === "cuda"
                                                            ? "bg-blue-500/20 text-blue-400"
                                                            : "bg-zinc-700 text-zinc-400"
                                                            }`}>
                                                            {model.device === "cuda" ? "GPU" : "CPU"}
                                                        </span>
                                                    </td>

                                                    {/* Accuracy Column */}
                                                    <td className="p-3 text-center bg-zinc-800/10 border-r border-zinc-800/50">
                                                        {model.accuracy !== null && model.accuracy !== undefined ? (
                                                            <div className={`font-mono font-bold ${model.accuracy >= 50 ? "text-emerald-400" : "text-rose-400"}`}>
                                                                {model.accuracy.toFixed(0)}%
                                                            </div>
                                                        ) : (
                                                            <span className="text-zinc-600">-</span>
                                                        )}
                                                    </td>

                                                    {/* Signal Column */}
                                                    <td className="p-3 text-center bg-zinc-800/10 border-r border-zinc-800/50">
                                                        {model.predictions?.["1d"] ? (
                                                            <span className={`px-2 py-1 rounded text-[10px] font-bold border ${model.predictions["1d"].direction === "bullish"
                                                                ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                                                                : "bg-rose-500/20 text-rose-400 border-rose-500/30"
                                                                }`}>
                                                                {model.predictions["1d"].direction.toUpperCase()}
                                                            </span>
                                                        ) : (
                                                            <span className="text-zinc-600">-</span>
                                                        )}
                                                    </td>

                                                    {/* Yesterday's Prediction Column */}
                                                    <td className="p-3 text-center bg-zinc-800/10 border-r border-zinc-800/50">
                                                        {model.prediction_today?.["1d"] ? (
                                                            <>
                                                                <div className={`font-mono font-bold ${model.prediction_today["1d"].direction === "bullish" ? "text-emerald-400" : "text-rose-400"}`}>
                                                                    ${model.prediction_today["1d"].price.toFixed(2)}
                                                                </div>
                                                                <div className="text-[10px] text-zinc-500">
                                                                    for Today
                                                                </div>
                                                            </>
                                                        ) : (
                                                            <span className="text-zinc-600">-</span>
                                                        )}
                                                    </td>

                                                    {horizons.map((h) => {
                                                        const pred = model.predictions?.[h];
                                                        if (model.error) {
                                                            return (
                                                                <td key={h} className="p-3 text-center text-rose-400 text-xs">
                                                                    Error
                                                                </td>
                                                            );
                                                        }
                                                        if (!pred) {
                                                            return (
                                                                <td key={h} className="p-3 text-center text-zinc-600">
                                                                    -
                                                                </td>
                                                            );
                                                        }
                                                        const isBullish = pred.direction === "bullish";
                                                        return (
                                                            <td key={h} className="p-3 text-center">
                                                                <div className={`font-mono font-bold ${isBullish ? "text-emerald-400" : "text-rose-400"}`}>
                                                                    ${pred.price.toFixed(2)}
                                                                </div>
                                                                <div className={`text-xs flex items-center justify-center gap-1 ${isBullish ? "text-emerald-500" : "text-rose-500"}`}>
                                                                    {isBullish ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                                                                    {isBullish ? "+" : ""}{pred.change_pct.toFixed(2)}%
                                                                </div>
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}

                        {/* Model Info Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {models.filter(m => m.is_available).map((model) => (
                                <div
                                    key={model.version}
                                    className="bg-zinc-900/30 border border-zinc-800 rounded-xl p-4"
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <h4 className="font-bold text-white">{model.display_name}</h4>
                                        <Info
                                            size={16}
                                            className="text-zinc-500 cursor-pointer hover:text-purple-400"
                                            onClick={() => setExpandedModel(expandedModel === model.version ? null : model.version)}
                                        />
                                    </div>
                                    <p className="text-xs text-zinc-500 mb-2">{model.description}</p>
                                    <p className="text-[10px] text-zinc-600">{model.training_notes}</p>

                                    {expandedModel === model.version && (
                                        <div className="mt-3 p-3 bg-zinc-950 rounded-lg border border-zinc-700 text-xs text-zinc-400 whitespace-pre-wrap">
                                            {model.reasoning}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}
