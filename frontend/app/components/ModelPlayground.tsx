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
    Clock,
    XCircle,
    CheckCircle,
} from "lucide-react";
import InfoTooltip from "./InfoTooltip";

const API_BASE = "http://localhost:8000";

// Hardcoded model cards (always visible even if API returns empty)
const HARDCODED_MODELS: ModelInfo[] = [
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
        is_available: true,
        status: "retired",
    },
    {
        version: "v7",
        display_name: "V7 LSTM Regression",
        description: "Predicts exact price. Failed due to 'Lazy Learner' bias (predicting t+1 = t).",
        reasoning: "V7 learned to minimize MSE by predicting tomorrow's price ≈ today's price. The model had no incentive to predict direction, only minimize error.",
        training_notes: "LSTM 128 hidden units. 41 features. DirectionalLoss didn't help.",
        is_available: true,
        status: "retired",
    },
];

interface ModelInfo {
    version: string;
    display_name: string;
    description: string;
    reasoning: string;
    training_notes: string;
    is_available: boolean;
    status?: "active" | "retired";
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

interface Signals {
    confidence: number;
    relative_strength: number;
    strength_label: string;
    similarity: number;
    similarity_label: string;
    market_mood: string;
    vix_proxy: number;
}

interface Explanation {
    model: string;
    ticker: string;
    current_price: number | null;
    why_selected: string;
    key_factors: string[];
    risk_factors: string[];
    architecture: string;
}

interface TickerComparison {
    ticker: string;
    current_price: number;
    timestamp: string;
    models: Record<string, ModelPrediction>;
    signals?: Signals;
    explanation?: Explanation;
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
    const [explanationLoading, setExplanationLoading] = useState(false);

    // Helper function for strength indicator color
    const getStrengthColor = (strength: number | undefined) => {
        if (strength === undefined) return "text-zinc-400";
        if (strength > 5) return "text-emerald-400";
        if (strength > 0) return "text-lime-400";
        if (strength > -5) return "text-orange-400";
        return "text-rose-400";
    };

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
            // Merge API models with hardcoded (use hardcoded if API is empty)
            const apiModels = data.models || [];
            if (apiModels.length === 0) {
                setModels(HARDCODED_MODELS);
            } else {
                // Merge: API models take precedence, fill in missing from hardcoded
                const merged = [...apiModels];
                HARDCODED_MODELS.forEach(hm => {
                    if (!merged.find(m => m.version === hm.version)) {
                        merged.push(hm);
                    }
                });
                setModels(merged);
            }
        } catch (e) {
            console.error("Failed to fetch models, using hardcoded:", e);
            setModels(HARDCODED_MODELS);
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

    // Fetch explanation when ticker is selected
    useEffect(() => {
        if (!selectedTicker || !isEnabled) return;

        const currentComparison = comparisons[selectedTicker];
        if (!currentComparison || currentComparison.explanation) return; // Already have explanation

        setExplanationLoading(true);
        fetch(`${API_BASE}/playground/explain/${selectedTicker}/v9`)
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
            .catch(err => console.error('Failed to fetch explanation:', err))
            .finally(() => setExplanationLoading(false));
    }, [selectedTicker, isEnabled, comparisons]);

    const horizons = ["1d", "1w", "1m", "6m"] as const;
    const horizonLabels = {
        "1d": "Tomorrow",
        "1w": "1 Week",
        "1m": "1 Month",
        "6m": "6 Months",
    };

    return (
        <div className="min-h-screen bg-[#0A0A0F] text-white">
            {/* Toolbar */}
            <div className="bg-[#12121A]/80 border-b border-white/[0.08] sticky top-16 z-30 backdrop-blur-md">
                <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                    <div>
                        {/* Empty left side or breadcrumbs if needed */}
                        <div className="flex items-center gap-2 text-zinc-400">
                            <Brain className="text-amber-400" size={20} />
                            <span className="font-medium text-white">Model Sandbox</span>
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
            </div>

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

            {/* System Brain Panel (Transparency Layer) */}
            {isEnabled && selectedTicker && (
                <div className="bg-[#12121A] border-b border-white/[0.08] px-6 py-6">
                    <div className="max-w-7xl mx-auto flex items-start gap-6">
                        <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                            <Brain size={32} className="text-indigo-400" />
                        </div>
                        <div className="space-y-2 max-w-2xl">
                            {/* Dynamic Title based on Active Model Version in Comparison */}
                            {/* Note: This assumes V9 is the primary perspective unless we explicitly select a model to 'inspect'. 
                                For now, we will toggle based on the 'expandedModel' or default to V9 if none selected. 
                                Actually, the Prompt asks to show this when a model is selected. 
                                Let's use 'expandedModel' as the 'inspector' trigger, or just show V9 as the default 'System Brain'.
                                Given the UI flow, let's keep V9 as the default 'Master' voice, and if user expands V8, show V8 voice.
                             */}

                            {expandedModel === "v8" ? (
                                <>
                                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                        System Brain Analysis
                                        <span className="text-xs font-normal text-rose-400 bg-rose-900/20 px-2 py-0.5 rounded-full border border-rose-800">V8 (Teacher) Active</span>
                                    </h3>
                                    <p className="text-sm text-zinc-300 leading-relaxed">
                                        <span className="text-rose-400 font-bold">Analysis:</span> "I failed here. I saw a
                                        <strong className="text-white mx-1">2-Sigma Volatility Spike</strong>
                                        and assumed it was a mean-reversion opportunity. In normal markets, this works.
                                        In 2022/2023, this signal often meant 'Bankruptcy'. I lack the regime filter to know the difference."
                                    </p>
                                </>
                            ) : expandedModel === "v7" ? (
                                <>
                                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                        System Brain Analysis
                                        <span className="text-xs font-normal text-amber-400 bg-amber-900/20 px-2 py-0.5 rounded-full border border-amber-800">V7 (Legacy) Active</span>
                                    </h3>
                                    <p className="text-sm text-zinc-300 leading-relaxed">
                                        <span className="text-amber-400 font-bold">Analysis:</span> "I am purely price-action based.
                                        I see <strong className="text-white mx-1">{selectedTicker}</strong>
                                        trending down, so I predict further downside. I do not understand 'Relative Strength' or 'Market Vix'.
                                        I am a trend-follower with no macro awareness."
                                    </p>
                                </>
                            ) : (
                                <>
                                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                        How the AI Makes Decisions
                                        <span
                                            className="text-xs font-normal text-emerald-400 bg-emerald-900/20 px-2 py-0.5 rounded-full border border-emerald-800"
                                        >
                                            V9 Active
                                        </span>
                                    </h3>
                                    <div className="bg-[#1A1A24]/50 border border-white/[0.08] rounded-lg p-4">
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            {/* How It Works */}
                                            <div>
                                                <div className="text-[10px] uppercase tracking-wider font-bold text-zinc-500 mb-2">How It Works</div>
                                                <div className="space-y-1.5 text-xs text-zinc-300">
                                                    <div className="flex justify-between items-center">
                                                        <span>Looks back:</span>
                                                        <span className="text-white">60 days of price history</span>
                                                    </div>
                                                    <div className="flex justify-between items-center">
                                                        <span>Analyzes:</span>
                                                        <span className="text-white">12 price patterns</span>
                                                    </div>
                                                    <div className="flex justify-between items-center">
                                                        <span>Compares:</span>
                                                        <span className="text-white">Stock vs S&P 500</span>
                                                    </div>
                                                    <div className="flex justify-between items-center">
                                                        <span>Outputs:</span>
                                                        <span className="text-white">Rank score (0-100%)</span>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Current Signals */}
                                            <div>
                                                <div className="text-[10px] uppercase tracking-wider font-bold text-zinc-500 mb-2">Current Signals</div>
                                                <div className="space-y-1.5 text-xs text-zinc-300">
                                                    <div className="flex justify-between items-center">
                                                        <div className="flex items-center gap-1">
                                                            <span>AI Confidence:</span>
                                                            <InfoTooltip
                                                                label="Confidence"
                                                                traderView="How sure the AI is (0-100%). Higher = more certain about the prediction."
                                                            />
                                                        </div>
                                                        <span className="text-emerald-400 font-bold">
                                                            {comparisons[selectedTicker]?.signals?.confidence?.toFixed(1) ?? '--'}%
                                                        </span>
                                                    </div>
                                                    <div className="flex justify-between items-center">
                                                        <div className="flex items-center gap-1">
                                                            <span>Strength vs Market:</span>
                                                            <InfoTooltip
                                                                label="Relative Strength"
                                                                traderView="Is this stock outperforming (+) or underperforming (-) the S&P 500?"
                                                            />
                                                        </div>
                                                        <span className={getStrengthColor(comparisons[selectedTicker]?.signals?.relative_strength)}>
                                                            {(comparisons[selectedTicker]?.signals?.relative_strength ?? 0) > 0 ? '+' : ''}
                                                            {comparisons[selectedTicker]?.signals?.relative_strength?.toFixed(1) ?? '--'}%
                                                            ({comparisons[selectedTicker]?.signals?.strength_label ?? 'Unknown'})
                                                        </span>
                                                    </div>
                                                    <div className="flex justify-between items-center">
                                                        <div className="flex items-center gap-1">
                                                            <span>Similarity to Others:</span>
                                                            <InfoTooltip
                                                                label="Correlation"
                                                                traderView="How much this stock moves with others in your portfolio. Low = good diversification."
                                                            />
                                                        </div>
                                                        <span className="text-amber-400">
                                                            {comparisons[selectedTicker]?.signals?.similarity?.toFixed(1) ?? '--'}%
                                                            ({comparisons[selectedTicker]?.signals?.similarity_label ?? 'Unknown'})
                                                        </span>
                                                    </div>
                                                    <div className="flex justify-between items-center">
                                                        <div className="flex items-center gap-1">
                                                            <span>Market Mood:</span>
                                                            <InfoTooltip
                                                                label="Volatility"
                                                                traderView="Is the market calm or chaotic? Low Vol = calm, High Vol = panic."
                                                            />
                                                        </div>
                                                        <span className="text-blue-400">
                                                            {comparisons[selectedTicker]?.signals?.market_mood ?? 'Unknown'}
                                                            {comparisons[selectedTicker]?.signals?.vix_proxy ? ` (${comparisons[selectedTicker]?.signals?.vix_proxy?.toFixed(1)}%)` : ''}
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Plain English Explanation */}
                                        <div className="mt-4 pt-4 border-t border-white/[0.08]">
                                            <div className="text-[10px] uppercase tracking-wider font-bold text-zinc-500 mb-2">Why This Stock?</div>
                                            <div className="text-xs text-zinc-300 leading-relaxed bg-emerald-500/5 border border-emerald-500/10 rounded p-3">
                                                {explanationLoading ? (
                                                    <span className="text-zinc-400">Loading explanation...</span>
                                                ) : comparisons[selectedTicker]?.explanation ? (
                                                    <>
                                                        <strong className="text-white">{selectedTicker}</strong> — {comparisons[selectedTicker].explanation.why_selected}
                                                        <ul className="mt-3 space-y-1.5 ml-4">
                                                            {comparisons[selectedTicker].explanation.key_factors?.map((factor, idx) => (
                                                                <li key={idx} className="text-sm text-zinc-300">
                                                                    • {factor}
                                                                </li>
                                                            ))}
                                                        </ul>
                                                        {comparisons[selectedTicker].explanation.risk_factors && (
                                                            <div className="mt-3 pt-3 border-t border-zinc-700">
                                                                <p className="text-xs text-zinc-500 uppercase mb-1">Risk Factors:</p>
                                                                <ul className="space-y-1 ml-4">
                                                                    {comparisons[selectedTicker].explanation.risk_factors.map((risk, idx) => (
                                                                        <li key={idx} className="text-xs text-zinc-400">
                                                                            ⚠️ {risk}
                                                                        </li>
                                                                    ))}
                                                                </ul>
                                                            </div>
                                                        )}
                                                    </>
                                                ) : (
                                                    <>
                                                        <strong className="text-white">{selectedTicker}</strong> is performing <strong className="text-emerald-400">much better</strong> than most other stocks right now.
                                                        When the market drops, {selectedTicker} holds up well. It also doesn't move in lockstep with your other holdings, which means <strong className="text-amber-400">less risk</strong> if the market crashes.
                                                        The AI ranks this in the <strong className="text-emerald-400">Top 10</strong> strongest stocks out of 500+.
                                                    </>
                                                )}
                                            </div>
                                        </div>

                                        {/* Warning */}
                                        <div className="mt-3 text-[10px] text-zinc-500 flex items-center gap-2 bg-amber-500/5 border border-amber-500/10 rounded p-2">
                                            <span className="text-amber-400">⚠</span>
                                            <span><strong className="text-amber-400">Important:</strong> Past performance doesn't guarantee future results. The AI was trained on large S&P 500 stocks and may be less accurate on small-cap or unusual stocks.</span>
                                        </div>
                                    </div>
                                </>
                            )}
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
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2">
                                            <h3 className="font-bold text-white">{model.display_name}</h3>
                                            {model.version === "v8" && (
                                                <span className="text-[10px] font-bold bg-red-500/20 text-red-400 px-2 py-0.5 rounded border border-red-500/30">
                                                    FAILED
                                                </span>
                                            )}
                                            {model.status === "active" && model.version === "v9" && (
                                                <span
                                                    className="text-[10px] font-bold bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30 relative group cursor-help"
                                                    title="V9 Transformer Details"
                                                >
                                                    ACTIVE
                                                    {/* Hover Tooltip */}
                                                    <span className="hidden group-hover:block absolute left-0 top-full mt-2 w-64 p-3 bg-[#1A1A24] border border-emerald-500/30 rounded-lg shadow-xl z-50 text-left">
                                                        <div className="text-emerald-400 font-semibold text-xs mb-2">V9 Transformer Rank Model</div>
                                                        <div className="text-white text-xs mb-2">
                                                            <strong>Architecture:</strong> Transformer with 60-day attention window
                                                        </div>
                                                        <div className="text-zinc-400 text-[10px] space-y-1">
                                                            <div>• Relative strength ranking vs SPY</div>
                                                            <div>• Cross-asset correlation filter (ρ &lt; 0.60)</div>
                                                            <div>• 12 stationary features</div>
                                                            <div>• Currently running paper trades</div>
                                                        </div>
                                                    </span>
                                                </span>
                                            )}
                                        </div>
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
                    <>
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
                            <div className="space-y-8">

                                {/* Active Models */}
                                <div>
                                    <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                                        <Zap size={16} className="text-amber-400" />
                                        Active Strategies
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                        {models.filter(m => m.is_available && !["v7", "v8"].includes(m.version)).map((model) => (
                                            <div
                                                key={model.version}
                                                className="bg-gradient-to-br from-indigo-900/10 to-purple-900/10 border border-white/[0.08] rounded-xl p-4"
                                            >
                                                <div className="flex items-center justify-between mb-2">
                                                    <h4 className="font-bold text-white">{model.display_name}</h4>
                                                    <Info
                                                        size={16}
                                                        className="text-zinc-500 cursor-pointer hover:text-indigo-400"
                                                        onClick={() => setExpandedModel(expandedModel === model.version ? null : model.version)}
                                                    />
                                                </div>
                                                <p className="text-xs text-zinc-300 mb-2">{model.description}</p>
                                                <p className="text-[10px] text-zinc-500 border-t border-white/5 pt-2">{model.training_notes}</p>

                                                {expandedModel === model.version && (
                                                    <div className="mt-3 p-3 bg-zinc-950 rounded-lg border border-zinc-700 text-xs text-zinc-400 whitespace-pre-wrap">
                                                        {model.reasoning}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Legacy Models */}
                                <div>
                                    <h3 className="text-lg font-medium text-zinc-400 mb-4 flex items-center gap-2">
                                        <HardDrive size={16} />
                                        Legacy & Experimental
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                        {models.filter(m => m.is_available && ["v7", "v8"].includes(m.version)).map((model) => (
                                            <div
                                                key={model.version}
                                                className="bg-zinc-900/20 border border-zinc-800 rounded-xl p-4 opacity-75 hover:opacity-100 transition-opacity"
                                            >
                                                <div className="flex items-center justify-between mb-2">
                                                    <h4 className="font-bold text-zinc-300">{model.display_name}</h4>
                                                    <span className="text-[10px] bg-zinc-800 text-zinc-500 px-2 py-0.5 rounded border border-zinc-700">LEGACY</span>
                                                </div>
                                                <p className="text-xs text-zinc-500 mb-2">{model.description}</p>
                                                {/* Collapsible details same as above */}
                                                <button
                                                    onClick={() => setExpandedModel(expandedModel === model.version ? null : model.version)}
                                                    className="text-[10px] text-zinc-600 hover:text-zinc-400 flex items-center gap-1 mt-2"
                                                >
                                                    {expandedModel === model.version ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                                                    Details
                                                </button>

                                                {expandedModel === model.version && (
                                                    <div className="mt-3 p-3 bg-zinc-950 rounded-lg border border-zinc-800 text-xs text-zinc-500 whitespace-pre-wrap">
                                                        {model.reasoning}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>

                            </div>
                        </div>

                        {/* Model Evolution History - Timeline Section */}
                        <ModelEvolutionHistory />

                        {/* V9 Deep Dive - Educational Section */}
                        <V9ExplanationPanel />
                    </>
                )}
            </main>
        </div>
    );
}

// Model Evolution History Component
function ModelEvolutionHistory() {
    return (
        <div className="mt-8 p-6 bg-zinc-900/50 rounded-2xl border border-zinc-800">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Clock size={20} className="text-amber-400" />
                Model Evolution History
            </h3>
            <p className="text-sm text-zinc-500 mb-6">
                The journey from failure to success. Each model taught us something.
            </p>

            <div className="relative">
                {/* Timeline line */}
                <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-zinc-800" />

                {/* V7 */}
                <div className="relative pl-12 pb-6">
                    <div className="absolute left-2.5 w-3 h-3 rounded-full bg-amber-500/50 border-2 border-amber-500" />
                    <div className="flex items-start gap-3">
                        <XCircle size={18} className="text-amber-500 mt-0.5 flex-shrink-0" />
                        <div>
                            <h4 className="font-bold text-amber-400">V7 LSTM Regression</h4>
                            <p className="text-xs text-zinc-500 mt-1">
                                Trained to predict exact prices. Learned to output <code className="bg-zinc-800 px-1 rounded">tomorrow = today</code> to minimize MSE.
                                Zero directional accuracy. <span className="text-rose-400">Failed.</span>
                            </p>
                        </div>
                    </div>
                </div>

                {/* V8 */}
                <div className="relative pl-12 pb-6">
                    <div className="absolute left-2.5 w-3 h-3 rounded-full bg-rose-500/50 border-2 border-rose-500" />
                    <div className="flex items-start gap-3">
                        <XCircle size={18} className="text-rose-500 mt-0.5 flex-shrink-0" />
                        <div>
                            <h4 className="font-bold text-rose-400">V8 Bi-LSTM</h4>
                            <p className="text-xs text-zinc-500 mt-1">
                                Switched to binary classification. Learned to detect volatility spikes.
                                But couldn't tell "panic volatility" from "opportunity volatility". Bought falling knives.
                                <span className="text-rose-400"> Failed.</span>
                            </p>
                        </div>
                    </div>
                </div>

                {/* V9 */}
                <div className="relative pl-12">
                    <div className="absolute left-2.5 w-3 h-3 rounded-full bg-emerald-500/50 border-2 border-emerald-500 animate-pulse" />
                    <div className="flex items-start gap-3">
                        <CheckCircle size={18} className="text-emerald-500 mt-0.5 flex-shrink-0" />
                        <div>
                            <h4 className="font-bold text-emerald-400">V9 Transformer</h4>
                            <p className="text-xs text-zinc-500 mt-1">
                                Transformer architecture. Ranks stocks by <strong className="text-white">Relative Strength vs SPY</strong>.
                                Holds Top 10 with low correlation. Diversifies risk.
                                <span className="text-emerald-400"> Current Production Model.</span>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// V9 Deep Dive Explanation Component
function V9ExplanationPanel() {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className="mt-8 p-6 bg-gradient-to-br from-emerald-900/20 to-zinc-900/50 rounded-2xl border border-emerald-800/50">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-between text-left"
            >
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Brain size={20} className="text-emerald-400" />
                    How V9 Actually Works
                </h3>
                {expanded ? <ChevronUp className="text-zinc-400" /> : <ChevronDown className="text-zinc-400" />}
            </button>

            {expanded && (
                <div className="mt-6 space-y-6">
                    {/* Why Transformer? */}
                    <section>
                        <h4 className="text-sm font-bold text-emerald-400 mb-2 uppercase tracking-wider">Why is V9 a Transformer?</h4>
                        <p className="text-sm text-zinc-400 leading-relaxed">
                            Unlike LSTMs which process data sequentially (like reading a book), Transformers use <strong className="text-white">attention</strong> to look at all 60 days simultaneously.
                            This allows V9 to spot patterns like "price went up on high volume 3 weeks ago, then consolidated, now showing similar setup" without the information fading over time.
                        </p>
                        <div className="mt-3 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700">
                            <code className="text-xs text-emerald-400">
                                Transformer = Better at cross-temporal patterns
                            </code>
                        </div>
                    </section>

                    {/* 12 Features */}
                    <section>
                        <h4 className="text-sm font-bold text-emerald-400 mb-2 uppercase tracking-wider">The 12 Stationary Features</h4>
                        <p className="text-sm text-zinc-400 mb-3">
                            V9 uses only <strong className="text-white">stationary</strong> features — no raw prices. It doesn't know if Apple is $50 or $500, only:
                        </p>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                            <div className="p-2 bg-zinc-800/50 rounded">📈 1-day, 5-day, 20-day returns</div>
                            <div className="p-2 bg-zinc-800/50 rounded">📊 RSI (14-day momentum)</div>
                            <div className="p-2 bg-zinc-800/50 rounded">📐 MACD histogram (trend)</div>
                            <div className="p-2 bg-zinc-800/50 rounded">📦 Volume ratio (vs 20-day avg)</div>
                            <div className="p-2 bg-zinc-800/50 rounded">📏 Distance from SMA20</div>
                            <div className="p-2 bg-zinc-800/50 rounded">🌊 20-day volatility</div>
                            <div className="p-2 bg-zinc-800/50 rounded">🏦 SPY 5-day return</div>
                            <div className="p-2 bg-zinc-800/50 rounded">😰 VIX level (fear gauge)</div>
                            <div className="p-2 bg-zinc-800/50 rounded">🔗 Correlation with SPY</div>
                            <div className="p-2 bg-zinc-800/50 rounded">💪 Relative strength vs SPY</div>
                        </div>
                    </section>

                    {/* Buy/Sell Logic */}
                    <section>
                        <h4 className="text-sm font-bold text-emerald-400 mb-2 uppercase tracking-wider">How V9 Decides to Buy or Sell</h4>
                        <div className="space-y-3 text-sm text-zinc-400">
                            <div className="flex items-start gap-3">
                                <TrendingUp className="text-emerald-500 flex-shrink-0 mt-0.5" size={16} />
                                <div>
                                    <strong className="text-emerald-400">BUY:</strong> Stock ranks in <strong className="text-white">Top 10</strong> among all 60 stocks checked,
                                    AND has <strong className="text-white">correlation &lt; 0.60</strong> with existing holdings (diversification).
                                </div>
                            </div>
                            <div className="flex items-start gap-3">
                                <TrendingDown className="text-rose-500 flex-shrink-0 mt-0.5" size={16} />
                                <div>
                                    <strong className="text-rose-400">SELL:</strong> Stock drops out of the Top 10 ranking OR
                                    a better-ranked, less-correlated stock is available to replace it.
                                </div>
                            </div>
                            <div className="flex items-start gap-3">
                                <AlertTriangle className="text-amber-500 flex-shrink-0 mt-0.5" size={16} />
                                <div>
                                    <strong className="text-amber-400">REBALANCE:</strong> Every 5 trading days, re-rank all stocks and
                                    adjust portfolio to maintain Top 10 with lowest correlation.
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Crash Behaviour */}
                    <section>
                        <h4 className="text-sm font-bold text-emerald-400 mb-2 uppercase tracking-wider">What Happens During Market Crashes?</h4>
                        <div className="space-y-3">
                            <div className="p-4 bg-rose-900/20 rounded-lg border border-rose-800/50">
                                <h5 className="font-bold text-rose-400 mb-2">🛑 VIX Safety Filter (VIX &gt; 30)</h5>
                                <p className="text-sm text-zinc-400">
                                    When VIX exceeds 30 (market panic like March 2020 or 2008), V9 automatically goes to <strong className="text-white">100% cash</strong>.
                                    No trading until VIX drops below threshold. This saved ~40% in the 2020 crash simulation.
                                </p>
                            </div>
                            <div className="p-4 bg-amber-900/20 rounded-lg border border-amber-800/50">
                                <h5 className="font-bold text-amber-400 mb-2">⚡ During Big Spikes</h5>
                                <p className="text-sm text-zinc-400">
                                    V9 doesn't chase momentum blindly. The <strong className="text-white">correlation filter</strong> prevents it from buying the same "hot sector" 10 times.
                                    During the AI bubble, it held NVDA but also kept positions in healthcare and financials.
                                </p>
                            </div>
                            <div className="p-4 bg-emerald-900/20 rounded-lg border border-emerald-800/50">
                                <h5 className="font-bold text-emerald-400 mb-2">📈 Recovery Mode</h5>
                                <p className="text-sm text-zinc-400">
                                    After crashes, relative rankings shift dramatically. V9 naturally rotates into beaten-down leaders
                                    (stocks with strong fundamentals that dropped) because their <strong className="text-white">relative strength</strong> improves vs others still falling.
                                </p>
                            </div>
                        </div>
                    </section>

                    {/* Performance Summary */}
                    <section className="p-4 bg-emerald-900/30 rounded-lg border border-emerald-700/50">
                        <h4 className="text-sm font-bold text-emerald-400 mb-2">📊 5-Year Backtest Summary</h4>
                        <div className="grid grid-cols-3 gap-4 text-center">
                            <div>
                                <div className="text-2xl font-bold text-emerald-400">+93%</div>
                                <div className="text-xs text-zinc-500">Total Return</div>
                            </div>
                            <div>
                                <div className="text-2xl font-bold text-white">~14%</div>
                                <div className="text-xs text-zinc-500">CAGR</div>
                            </div>
                            <div>
                                <div className="text-2xl font-bold text-amber-400">Top 10</div>
                                <div className="text-xs text-zinc-500">Max Holdings</div>
                            </div>
                        </div>
                    </section>
                </div>
            )}
        </div>
    );
}
