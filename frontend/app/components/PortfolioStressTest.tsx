"use client";

import React, { useState, useEffect, useMemo } from "react";
import { getApiUrl } from "@/config/api";
import {
    Brain,
    RefreshCw,
    TrendingUp,
    TrendingDown,
    AlertTriangle,
    ChevronDown,
    ChevronUp,
    Zap,
    Shield,
    Flame,
    Clock,
    Target,
    Activity,
    BarChart3,
    ArrowLeft,
    Eye,
    EyeOff,
    Layers,
    Play,
    Terminal,
    X,
    Info,
} from "lucide-react";
import SmartTooltip from "./SmartTooltip";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
    ReferenceLine,
} from "recharts";


interface PortfolioResult {
    session_id: string;
    display_name: string;
    description: string;
    parameters: {
        vix: number;
        top_k: number;
        correlation: number;
        rebalance: number;
        period?: string;
    };
    start_value: number;
    final_value: number;
    total_return: number;
    max_drawdown: number;
    trade_count: number;
    start_date: string | null;
    end_date: string | null;
}

interface HistoryPoint {
    date: string;
    value: number;
    equity: number;
    cash: number;
}

interface PortfolioComparisonProps {
    onBack: () => void;
}

// Color palette for each portfolio
const PORTFOLIO_COLORS: Record<string, string> = {
    "v9_golden_2020": "#10B981",
    "v9_golden_2025": "#10B981",
    "v9_aggressive": "#F97316",
    "v9_conservative": "#3B82F6",
    "v9_no_vix": "#EF4444",
    "v9_daily": "#FBBF24",
    "v9_covid_2020": "#06B6D4",
    "v9_crisis_2008": "#EC4899",
    "v9_bear_2022": "#8B5CF6",
    "v9_ultimate_20yr": "#14B8A6",
};

// Parameter explanation tooltips
const PARAM_TOOLTIPS = {
    vix: {
        title: "VIX Threshold",
        simple: "When market fear (VIX) exceeds this level, V9 sells everything and goes to 100% cash.",
        technical: "VIX measures implied volatility of S&P 500 options. Normal = 12-20. Elevated = 20-30. Panic = 30+."
    },
    top_k: {
        title: "Holdings Count",
        simple: "How many stocks V9 holds at once. More = safer but lower returns. Less = riskier but higher potential.",
        technical: "Only top K ranked stocks are held. Lower K = concentrated bets. Higher K = diversified."
    },
    correlation: {
        title: "Correlation Filter",
        simple: "Prevents V9 from buying stocks that move together. Keeps portfolio diversified.",
        technical: "New stocks must have correlation < threshold with existing holdings to be bought."
    },
    rebalance: {
        title: "Rebalance Frequency",
        simple: "How often V9 reviews and adjusts positions. Daily = reactive. Weekly = stable.",
        technical: "Every N days, V9 re-ranks stocks and adjusts. Lower = more trades. Higher = fewer trades."
    },
    drawdown: {
        title: "Maximum Drawdown",
        simple: "The biggest drop from a peak. If you had $15,000 and it fell to $10,000, that's a 33% drawdown.",
        technical: "Max Drawdown = (Peak Value - Trough Value) / Peak Value × 100%. Measures worst-case loss if you bought at the top."
    }
};

function InfoBubble({ param }: { param: keyof typeof PARAM_TOOLTIPS }) {
    const info = PARAM_TOOLTIPS[param];
    return (
        <SmartTooltip
            term={info.title}
            traderExplanation={info.simple}
            academicExplanation={info.technical}
        >
            <Info size={14} className="text-zinc-500 hover:text-amber-400 transition-colors ml-1" />
        </SmartTooltip>
    );
}

export default function PortfolioComparison({ onBack }: PortfolioComparisonProps) {
    const [portfolios, setPortfolios] = useState<PortfolioResult[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedPortfolio, setSelectedPortfolio] = useState<string | null>(null);
    const [availableConfigs, setAvailableConfigs] = useState<string[]>([]);
    const [historyData, setHistoryData] = useState<Record<string, HistoryPoint[]>>({});
    const [loadingHistory, setLoadingHistory] = useState<string | null>(null);
    const [compareMode, setCompareMode] = useState(false);
    const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
    const [showRerunModal, setShowRerunModal] = useState(false);
    const [rerunning, setRerunning] = useState(false);
    const [rerunLogs, setRerunLogs] = useState<string[]>([]);
    const [crashSectionOpen, setCrashSectionOpen] = useState(false);

    useEffect(() => {
        fetchPortfolios();
    }, []);

    const fetchPortfolios = async () => {
        setLoading(true);
        try {
            const res = await fetch(getApiUrl("/paper/portfolios/compare"));
            if (res.ok) {
                const data = await res.json();
                setPortfolios(data.portfolios || []);
                setAvailableConfigs(data.available_configs || []);
            }
        } catch (e) {
            console.error("Failed to fetch portfolios:", e);
        } finally {
            setLoading(false);
        }
    };

    const fetchHistory = async (sessionId: string) => {
        if (historyData[sessionId]) return;

        setLoadingHistory(sessionId);
        try {
            const res = await fetch(getApiUrl(`/paper/portfolios/${sessionId}/history`));
            if (res.ok) {
                const data = await res.json();
                setHistoryData(prev => ({ ...prev, [sessionId]: data }));
            }
        } catch (e) {
            console.error(`Failed to fetch history for ${sessionId}:`, e);
        } finally {
            setLoadingHistory(null);
        }
    };

    const handlePortfolioClick = async (sessionId: string) => {
        if (compareMode) {
            setSelectedForCompare(prev =>
                prev.includes(sessionId)
                    ? prev.filter(id => id !== sessionId)
                    : [...prev, sessionId]
            );
            if (!historyData[sessionId]) {
                await fetchHistory(sessionId);
            }
        } else {
            if (selectedPortfolio === sessionId) {
                setSelectedPortfolio(null);
            } else {
                setSelectedPortfolio(sessionId);
                await fetchHistory(sessionId);
            }
        }
    };

    const handleRerunSimulations = async () => {
        setRerunning(true);
        setRerunLogs(["🚀 Starting simulations...", ""]);
        setRerunLogs(prev => [...prev, "Run this command in your terminal:"]);
        setRerunLogs(prev => [...prev, ""]);
        setRerunLogs(prev => [...prev, "docker exec proxmox_stock_backend python -m scripts.seed_multi_portfolio --all"]);
        setRerunLogs(prev => [...prev, ""]);
        setRerunLogs(prev => [...prev, "⏱️ This will take approximately 10-15 minutes."]);
        setRerunLogs(prev => [...prev, "📊 Each portfolio simulation runs ~1250 trading days."]);
        setRerunLogs(prev => [...prev, ""]);
        setRerunLogs(prev => [...prev, "When complete, click 'Refresh' to see results."]);
        setRerunning(false);
    };

    // Comparison chart data
    const comparisonChartData = useMemo(() => {
        if (selectedForCompare.length === 0) return [];

        const allDates = new Set<string>();
        selectedForCompare.forEach(id => {
            (historyData[id] || []).forEach(h => allDates.add(h.date));
        });

        const sortedDates = Array.from(allDates).sort();

        return sortedDates.map(date => {
            const point: Record<string, number | string> = { date };
            selectedForCompare.forEach(id => {
                const history = historyData[id] || [];
                const match = history.find(h => h.date === date);
                if (match) {
                    const startValue = history[0]?.value || 10000;
                    point[id] = ((match.value / startValue) - 1) * 100;
                }
            });
            return point;
        });
    }, [selectedForCompare, historyData]);

    const getReturnColor = (ret: number) => {
        if (ret > 50) return "text-emerald-400";
        if (ret > 20) return "text-green-400";
        if (ret > 0) return "text-lime-400";
        if (ret > -10) return "text-yellow-400";
        if (ret > -30) return "text-orange-400";
        return "text-red-400";
    };

    const getDrawdownColor = (dd: number) => {
        if (dd < 10) return "text-emerald-400";
        if (dd < 20) return "text-green-400";
        if (dd < 30) return "text-yellow-400";
        if (dd < 40) return "text-orange-400";
        return "text-red-400";
    };

    const getConfigIcon = (sessionId: string) => {
        if (sessionId.includes("ultimate")) return <BarChart3 className="text-teal-400" size={18} />;
        if (sessionId.includes("aggressive")) return <Flame className="text-orange-400" size={18} />;
        if (sessionId.includes("conservative")) return <Shield className="text-blue-400" size={18} />;
        if (sessionId.includes("no_vix")) return <AlertTriangle className="text-red-400" size={18} />;
        if (sessionId.includes("daily")) return <Zap className="text-yellow-400" size={18} />;
        if (sessionId.includes("covid")) return <Activity className="text-cyan-400" size={18} />;
        if (sessionId.includes("crisis")) return <Activity className="text-rose-400" size={18} />;
        if (sessionId.includes("bear")) return <TrendingDown className="text-amber-400" size={18} />;
        return <Target className="text-emerald-400" size={18} />;
    };

    const formatDate = (dateStr: string) => {
        const d = new Date(dateStr);
        return `${d.getMonth() + 1}/${d.getFullYear().toString().slice(-2)}`;
    };

    return (
        <div className="min-h-screen bg-[#0A0A0F] text-white">
            {/* Rerun Modal */}
            {showRerunModal && (
                <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
                    <div className="bg-zinc-900 border border-zinc-700 rounded-xl p-6 max-w-2xl w-full mx-4">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                <Terminal className="text-amber-400" size={20} />
                                Run Portfolio Simulations
                            </h3>
                            <button
                                onClick={() => setShowRerunModal(false)}
                                className="p-1 hover:bg-zinc-700 rounded"
                            >
                                <X size={20} className="text-zinc-400" />
                            </button>
                        </div>

                        {rerunLogs.length === 0 ? (
                            <div className="space-y-4">
                                <p className="text-zinc-400 text-sm">
                                    This will run 9 different portfolio simulations with varying parameters and time periods.
                                    Each simulation processes ~1250 trading days.
                                </p>
                                <div className="bg-amber-900/20 border border-amber-700/50 rounded-lg p-4">
                                    <p className="text-amber-400 text-sm font-medium mb-2">⚠️ Note:</p>
                                    <ul className="text-sm text-zinc-400 space-y-1">
                                        <li>• Takes approximately 10-15 minutes</li>
                                        <li>• Existing data will be replaced</li>
                                        <li>• Requires GPU for best performance</li>
                                    </ul>
                                </div>
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => setShowRerunModal(false)}
                                        className="flex-1 py-2 px-4 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-300"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={handleRerunSimulations}
                                        disabled={rerunning}
                                        className="flex-1 py-2 px-4 bg-amber-600 hover:bg-amber-500 rounded-lg text-white font-semibold flex items-center justify-center gap-2"
                                    >
                                        <Play size={16} />
                                        Show Command
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="bg-zinc-950 rounded-lg p-4 font-mono text-sm max-h-80 overflow-auto">
                                    {rerunLogs.map((log, i) => (
                                        <div key={i} className={log.startsWith("docker") ? "text-amber-400 bg-zinc-800 p-2 rounded my-2 select-all" : "text-zinc-300"}>
                                            {log || "\u00A0"}
                                        </div>
                                    ))}
                                </div>
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => {
                                            setShowRerunModal(false);
                                            setRerunLogs([]);
                                        }}
                                        className="flex-1 py-2 px-4 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-300"
                                    >
                                        Close
                                    </button>
                                    <button
                                        onClick={fetchPortfolios}
                                        className="flex-1 py-2 px-4 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white font-semibold flex items-center justify-center gap-2"
                                    >
                                        <RefreshCw size={16} />
                                        Refresh Data
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Header */}
            <header className="sticky top-0 z-10 bg-[#12121A]/95 backdrop-blur border-b border-white/[0.06]">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <button
                                onClick={onBack}
                                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                            >
                                <ArrowLeft size={20} className="text-zinc-400" />
                            </button>
                            <div>
                                <h1 className="text-xl font-bold text-white flex items-center gap-2">
                                    <BarChart3 className="text-amber-400" size={24} />
                                    V9 Portfolio Laboratory
                                </h1>
                                <p className="text-xs text-zinc-500">Compare how V9 behaves with different settings and during market crises</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-3">
                            <button
                                onClick={() => setShowRerunModal(true)}
                                className="flex items-center gap-2 px-4 py-2 bg-rose-600/20 hover:bg-rose-600/30 border border-rose-500/50 text-rose-400 rounded-lg transition-colors"
                            >
                                <Play size={16} />
                                Rerun All
                            </button>
                            <button
                                onClick={() => {
                                    setCompareMode(!compareMode);
                                    if (!compareMode) setSelectedForCompare([]);
                                }}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${compareMode
                                    ? "bg-amber-500/20 text-amber-400 border border-amber-500/50"
                                    : "bg-zinc-800 hover:bg-zinc-700 text-zinc-300"
                                    }`}
                            >
                                <Layers size={16} />
                                {compareMode ? "Exit Compare" : "Compare"}
                            </button>
                            <button
                                onClick={fetchPortfolios}
                                disabled={loading}
                                className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors"
                            >
                                <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
                                Refresh
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8">
                {/* Compare Mode Chart */}
                {compareMode && selectedForCompare.length > 0 && (
                    <div className="bg-zinc-900/50 rounded-2xl border border-amber-500/30 p-6 mb-8">
                        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                            <Layers className="text-amber-400" size={20} />
                            Portfolio Comparison (% Return Over Time)
                        </h2>
                        <div className="h-[400px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={comparisonChartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis
                                        dataKey="date"
                                        tickFormatter={formatDate}
                                        stroke="#71717a"
                                        tick={{ fontSize: 10 }}
                                    />
                                    <YAxis
                                        stroke="#71717a"
                                        tick={{ fontSize: 10 }}
                                        tickFormatter={(v) => `${v > 0 ? '+' : ''}${v.toFixed(0)}%`}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "#1A1A24",
                                            border: "1px solid #3f3f46",
                                            borderRadius: "8px",
                                        }}
                                        labelStyle={{ color: "#fff" }}
                                        formatter={(value: number, name: string) => [
                                            `${value > 0 ? '+' : ''}${value.toFixed(2)}%`,
                                            portfolios.find(p => p.session_id === name)?.display_name || name
                                        ]}
                                    />
                                    <Legend />
                                    <ReferenceLine y={0} stroke="#3f3f46" strokeDasharray="3 3" />
                                    {selectedForCompare.map(id => (
                                        <Line
                                            key={id}
                                            type="monotone"
                                            dataKey={id}
                                            stroke={PORTFOLIO_COLORS[id] || "#888"}
                                            strokeWidth={2}
                                            dot={false}
                                            name={portfolios.find(p => p.session_id === id)?.display_name || id}
                                        />
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}

                {/* Parameter Legend with Drawdown Explanation */}
                <div className="bg-zinc-900/50 rounded-2xl border border-zinc-800 p-6 mb-8">
                    <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                        <Brain size={20} className="text-amber-400" />
                        Understanding the Numbers
                    </h2>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div className="p-4 bg-zinc-800/50 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="font-semibold text-amber-400">VIX</span>
                                <InfoBubble param="vix" />
                            </div>
                            <p className="text-xs text-zinc-400">Fear exit threshold</p>
                        </div>
                        <div className="p-4 bg-zinc-800/50 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="font-semibold text-amber-400">Top-K</span>
                                <InfoBubble param="top_k" />
                            </div>
                            <p className="text-xs text-zinc-400">Holdings count</p>
                        </div>
                        <div className="p-4 bg-zinc-800/50 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="font-semibold text-amber-400">Correlation</span>
                                <InfoBubble param="correlation" />
                            </div>
                            <p className="text-xs text-zinc-400">Diversification</p>
                        </div>
                        <div className="p-4 bg-zinc-800/50 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="font-semibold text-amber-400">Rebalance</span>
                                <InfoBubble param="rebalance" />
                            </div>
                            <p className="text-xs text-zinc-400">Trading frequency</p>
                        </div>
                        <div className="p-4 bg-rose-900/30 rounded-lg border border-rose-800/50">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="font-semibold text-rose-400">Drawdown</span>
                                <InfoBubble param="drawdown" />
                            </div>
                            <p className="text-xs text-zinc-400">Worst peak-to-bottom drop</p>
                        </div>
                    </div>
                </div>

                {/* Results */}
                {loading ? (
                    <div className="flex items-center justify-center py-20">
                        <RefreshCw className="animate-spin text-amber-400" size={32} />
                    </div>
                ) : portfolios.length === 0 ? (
                    <div className="bg-zinc-900/50 rounded-2xl border border-zinc-800 p-12 text-center">
                        <AlertTriangle className="mx-auto text-amber-400 mb-4" size={48} />
                        <h3 className="text-xl font-bold text-white mb-2">No Portfolio Data Yet</h3>
                        <p className="text-zinc-400 mb-6">
                            Click "Rerun All" above to generate simulation data.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {/* Summary Row */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                            <div className="bg-emerald-900/20 border border-emerald-700/50 rounded-xl p-4">
                                <div className="text-sm text-emerald-400 mb-1 flex items-center gap-2">
                                    <TrendingUp size={14} />
                                    Best Total Return
                                </div>
                                <div className="text-2xl font-bold text-white">
                                    {portfolios[0]?.total_return > 0 ? "+" : ""}
                                    {portfolios[0]?.total_return?.toFixed(1)}%
                                </div>
                                <div className="text-xs text-zinc-400">{portfolios[0]?.display_name}</div>
                            </div>
                            <div className="bg-rose-900/20 border border-rose-700/50 rounded-xl p-4">
                                <div className="text-sm text-rose-400 mb-1 flex items-center gap-2">
                                    <TrendingDown size={14} />
                                    Worst Drawdown
                                </div>
                                <div className="text-2xl font-bold text-white">
                                    -{Math.max(...portfolios.map(p => p.max_drawdown)).toFixed(1)}%
                                </div>
                                <div className="text-xs text-zinc-400">Biggest peak-to-bottom drop</div>
                            </div>
                            <div className="bg-amber-900/20 border border-amber-700/50 rounded-xl p-4">
                                <div className="text-sm text-amber-400 mb-1 flex items-center gap-2">
                                    <BarChart3 size={14} />
                                    Portfolios Tested
                                </div>
                                <div className="text-2xl font-bold text-white">{portfolios.length}</div>
                                <div className="text-xs text-zinc-400">Including 20-Year Ultimate</div>
                            </div>
                        </div>

                        {compareMode && (
                            <div className="bg-amber-900/20 border border-amber-500/30 rounded-lg p-3 mb-4 text-sm text-amber-400 flex items-center gap-2">
                                <Layers size={16} />
                                Click portfolios to add/remove from the comparison chart above
                            </div>
                        )}

                        {/* Portfolio Cards */}
                        {portfolios.map((portfolio, idx) => (
                            <div
                                key={portfolio.session_id}
                                className={`bg-zinc-900/50 rounded-xl border transition-all ${compareMode && selectedForCompare.includes(portfolio.session_id)
                                    ? "border-amber-500"
                                    : portfolio.session_id.includes("ultimate")
                                        ? "border-teal-500/50"
                                        : idx === 0
                                            ? "border-emerald-500/50"
                                            : "border-zinc-800 hover:border-amber-500/50"
                                    }`}
                            >
                                <div
                                    className="p-6 cursor-pointer"
                                    onClick={() => handlePortfolioClick(portfolio.session_id)}
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            {compareMode ? (
                                                selectedForCompare.includes(portfolio.session_id) ? (
                                                    <Eye className="text-amber-400" size={18} />
                                                ) : (
                                                    <EyeOff className="text-zinc-600" size={18} />
                                                )
                                            ) : (
                                                getConfigIcon(portfolio.session_id)
                                            )}
                                            <div>
                                                <h3 className="font-bold text-white flex items-center gap-2">
                                                    {portfolio.display_name}
                                                    {portfolio.session_id.includes("ultimate") && (
                                                        <span className="text-xs bg-teal-500/20 text-teal-400 px-2 py-0.5 rounded-full">
                                                            20 YEARS
                                                        </span>
                                                    )}
                                                    {idx === 0 && !compareMode && !portfolio.session_id.includes("ultimate") && (
                                                        <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
                                                            BEST
                                                        </span>
                                                    )}
                                                </h3>
                                                <p className="text-xs text-zinc-500">{portfolio.description}</p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-6 text-right">
                                            <div>
                                                <div className={`text-xl font-bold ${getReturnColor(portfolio.total_return)}`}>
                                                    {portfolio.total_return > 0 ? "+" : ""}{portfolio.total_return.toFixed(1)}%
                                                </div>
                                                <div className="text-xs text-zinc-500">Return</div>
                                            </div>
                                            <div>
                                                <div className={`text-xl font-bold ${getDrawdownColor(portfolio.max_drawdown)}`}>
                                                    -{portfolio.max_drawdown.toFixed(1)}%
                                                </div>
                                                <div className="text-xs text-zinc-500">Drawdown</div>
                                            </div>
                                            <div>
                                                <div className="text-xl font-bold text-white">
                                                    {portfolio.trade_count}
                                                </div>
                                                <div className="text-xs text-zinc-500">Trades</div>
                                            </div>
                                            {!compareMode && (
                                                selectedPortfolio === portfolio.session_id ? (
                                                    <ChevronUp size={20} className="text-zinc-400" />
                                                ) : (
                                                    <ChevronDown size={20} className="text-zinc-400" />
                                                )
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Expanded Chart Section */}
                                {selectedPortfolio === portfolio.session_id && !compareMode && (
                                    <div className="px-6 pb-6 border-t border-zinc-800">
                                        <div className="grid grid-cols-4 gap-4 py-4">
                                            <div className="text-center">
                                                <div className="text-xs text-zinc-500">VIX Threshold</div>
                                                <div className="font-bold text-white">
                                                    {portfolio.parameters.vix === 999 ? "OFF" : portfolio.parameters.vix}
                                                </div>
                                            </div>
                                            <div className="text-center">
                                                <div className="text-xs text-zinc-500">Max Holdings</div>
                                                <div className="font-bold text-white">{portfolio.parameters.top_k}</div>
                                            </div>
                                            <div className="text-center">
                                                <div className="text-xs text-zinc-500">Correlation</div>
                                                <div className="font-bold text-white">{portfolio.parameters.correlation}</div>
                                            </div>
                                            <div className="text-center">
                                                <div className="text-xs text-zinc-500">Rebalance</div>
                                                <div className="font-bold text-white">
                                                    {portfolio.parameters.rebalance === 1 ? "Daily" : `${portfolio.parameters.rebalance}d`}
                                                </div>
                                            </div>
                                        </div>

                                        <div className="h-[300px] mt-4">
                                            {loadingHistory === portfolio.session_id ? (
                                                <div className="flex items-center justify-center h-full">
                                                    <RefreshCw className="animate-spin text-amber-400" size={24} />
                                                </div>
                                            ) : historyData[portfolio.session_id]?.length > 0 ? (
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart data={historyData[portfolio.session_id]}>
                                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                                        <XAxis
                                                            dataKey="date"
                                                            tickFormatter={formatDate}
                                                            stroke="#71717a"
                                                            tick={{ fontSize: 10 }}
                                                        />
                                                        <YAxis
                                                            stroke="#71717a"
                                                            tick={{ fontSize: 10 }}
                                                            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                                                        />
                                                        <Tooltip
                                                            contentStyle={{
                                                                backgroundColor: "#1A1A24",
                                                                border: "1px solid #3f3f46",
                                                                borderRadius: "8px",
                                                            }}
                                                            labelStyle={{ color: "#fff" }}
                                                            formatter={(value: number) => [`$${value.toLocaleString()}`, "Value"]}
                                                        />
                                                        <ReferenceLine y={10000} stroke="#3f3f46" strokeDasharray="3 3" label="Start" />
                                                        <Line
                                                            type="monotone"
                                                            dataKey="value"
                                                            stroke={PORTFOLIO_COLORS[portfolio.session_id] || "#F59E0B"}
                                                            strokeWidth={2}
                                                            dot={false}
                                                            name="Portfolio Value"
                                                        />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            ) : (
                                                <div className="flex items-center justify-center h-full text-zinc-500">
                                                    No history data available
                                                </div>
                                            )}
                                        </div>

                                        <div className="mt-4 text-xs text-zinc-400 flex justify-between">
                                            <span>
                                                Period: {portfolio.start_date || "N/A"} → {portfolio.end_date || "N/A"}
                                            </span>
                                            <span>
                                                ${portfolio.start_value?.toLocaleString()} → ${portfolio.final_value?.toLocaleString()}
                                            </span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}

                {/* Collapsible Crash Behavior Explanation */}
                <div className="mt-12">
                    <button
                        onClick={() => setCrashSectionOpen(!crashSectionOpen)}
                        className="w-full bg-gradient-to-br from-amber-900/20 to-zinc-900/50 rounded-2xl border border-amber-800/50 p-6 text-left hover:border-amber-600/50 transition-colors"
                    >
                        <div className="flex items-center justify-between">
                            <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                <Brain className="text-amber-400" size={20} />
                                How V9 Handles a Market Crash (Step by Step)
                            </h3>
                            {crashSectionOpen ? (
                                <ChevronUp size={24} className="text-amber-400" />
                            ) : (
                                <ChevronDown size={24} className="text-amber-400" />
                            )}
                        </div>
                        {!crashSectionOpen && (
                            <p className="text-sm text-zinc-500 mt-2">Click to learn about V9's 3-phase crash response protocol</p>
                        )}
                    </button>

                    {crashSectionOpen && (
                        <div className="bg-gradient-to-br from-amber-900/10 to-zinc-900/30 rounded-b-2xl border-x border-b border-amber-800/50 p-6 -mt-2 space-y-6">
                            {/* Phase 1 */}
                            <div className="bg-rose-900/30 rounded-lg p-5 border border-rose-800/50">
                                <div className="flex items-start gap-4">
                                    <div className="w-12 h-12 rounded-full bg-rose-500/20 flex items-center justify-center flex-shrink-0">
                                        <AlertTriangle className="text-rose-400" size={24} />
                                    </div>
                                    <div>
                                        <h4 className="font-bold text-rose-400 text-lg mb-2">Phase 1: Panic Detection (Day 1-2)</h4>
                                        <p className="text-zinc-300 mb-3">
                                            When bad news hits (COVID, Lehman Brothers, etc.), market fear spikes instantly.
                                            The VIX index jumps from normal levels (15-20) to panic levels (30+).
                                        </p>
                                        <div className="bg-rose-950/50 rounded p-3 text-sm">
                                            <strong className="text-rose-400">What V9 Does:</strong>
                                            <ul className="mt-2 space-y-1 text-zinc-400">
                                                <li>• Detects VIX &gt; 30 threshold</li>
                                                <li>• Immediately sells <strong>ALL</strong> holdings at market price</li>
                                                <li>• Moves to 100% cash</li>
                                                <li>• Stops making any new trades</li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Phase 2 */}
                            <div className="bg-amber-900/30 rounded-lg p-5 border border-amber-800/50">
                                <div className="flex items-start gap-4">
                                    <div className="w-12 h-12 rounded-full bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                                        <Clock className="text-amber-400" size={24} />
                                    </div>
                                    <div>
                                        <h4 className="font-bold text-amber-400 text-lg mb-2">Phase 2: The Waiting Period (Weeks 2-6)</h4>
                                        <p className="text-zinc-300 mb-3">
                                            While the market is crashing 20-40%, V9 sits in cash doing nothing.
                                            This is intentional - most traders panic sell at the bottom or try to "catch the falling knife."
                                        </p>
                                        <div className="bg-amber-950/50 rounded p-3 text-sm">
                                            <strong className="text-amber-400">What V9 Does:</strong>
                                            <ul className="mt-2 space-y-1 text-zinc-400">
                                                <li>• Checks VIX every rebalance day</li>
                                                <li>• If VIX still &gt; 30 → stays in cash</li>
                                                <li>• Makes ZERO trades (no panic, no FOMO)</li>
                                                <li>• Protects capital while others lose 30-50%</li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Phase 3 */}
                            <div className="bg-emerald-900/30 rounded-lg p-5 border border-emerald-800/50">
                                <div className="flex items-start gap-4">
                                    <div className="w-12 h-12 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
                                        <TrendingUp className="text-emerald-400" size={24} />
                                    </div>
                                    <div>
                                        <h4 className="font-bold text-emerald-400 text-lg mb-2">Phase 3: Smart Re-Entry (Week 7+)</h4>
                                        <p className="text-zinc-300 mb-3">
                                            Once the VIX drops below 30, it signals the panic is subsiding.
                                            V9 re-enters the market, but only buys the stocks that have stabilized first (natural leaders).
                                        </p>
                                        <div className="bg-emerald-950/50 rounded p-3 text-sm">
                                            <strong className="text-emerald-400">What V9 Does:</strong>
                                            <ul className="mt-2 space-y-1 text-zinc-400">
                                                <li>• Resumes normal ranking</li>
                                                <li>• Buys Top 10 stocks by V9 score</li>
                                                <li>• These are often the strongest companies (AAPL, MSFT, etc.)</li>
                                                <li>• Catches the recovery while weak stocks continue falling</li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="p-4 bg-zinc-800/50 rounded-lg">
                                <p className="text-sm text-zinc-400">
                                    <strong className="text-amber-400">💡 Key Insight:</strong> Compare the "No VIX Filter" portfolio to "Golden Config"
                                    during March 2020. Without the VIX filter, the portfolio experiences the full 34% crash.
                                    With the filter, it exits early and re-enters after the worst is over.
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
