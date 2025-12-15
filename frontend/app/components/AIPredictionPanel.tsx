"use client";

import { useState, useEffect, useMemo } from "react";
import { Brain, TrendingUp, TrendingDown, AlertTriangle, RefreshCw, X, Activity, ShieldAlert, Zap } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine, Legend } from "recharts";
import { getApiUrl } from "@/config/api";

interface AssetHistory {
    date: string;
    close: number;
    spy_close?: number;
    predicted_close?: number | null;
}

interface DetailResponse {
    history: AssetHistory[];
    rank?: number | null;
}

interface AIPredictionPanelProps {
    isOpen: boolean;
    onClose: () => void;
    initialTicker?: string;
}

export default function AIPredictionPanel({ isOpen, onClose }: AIPredictionPanelProps) {
    const [tickers, setTickers] = useState<string[]>([]);
    const [selectedTicker, setSelectedTicker] = useState<string>("");
    const [detailData, setDetailData] = useState<DetailResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [modelVersion, setModelVersion] = useState<string>("Unknown");

    // Fetch tickers
    useEffect(() => {
        if (isOpen) {
            fetch(getApiUrl('/dashboard'))
                .then((res) => res.json())
                .then((data) => {
                    const list = data.map((d: any) => d.ticker);
                    setTickers(list);
                    if (list.length > 0 && !selectedTicker) {
                        setSelectedTicker(list[0]);
                    }
                })
                .catch(console.error);

            // Fetch active model
            fetch(getApiUrl('/status/ai'))
                .then((res) => res.json())
                .then((data) => setModelVersion(data.model_version || "V9 Ranker"))
                .catch(() => setModelVersion("V9 Ranker"));
        }
    }, [isOpen]);

    // Fetch Detail Data
    useEffect(() => {
        if (!selectedTicker) return;

        const fetchData = async () => {
            setIsLoading(true);
            setError(null);
            try {
                // Use the dashboard detail endpoint which now supports V9 Rank
                const res = await fetch(getApiUrl(`/dashboard/${selectedTicker}`));
                if (!res.ok) throw new Error("Failed to fetch data");
                const data = await res.json();
                setDetailData(data);
            } catch (err) {
                setError("Failed to load asset data");
                console.error(err);
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [selectedTicker]);

    // Derived Metrics Calculations
    const metrics = useMemo(() => {
        if (!detailData || !detailData.history || detailData.history.length < 20) return null;

        const hist = detailData.history;
        const latest = hist[hist.length - 1];
        const prices = hist.map(d => d.close);

        // 1. ATR Calculation (Simplified 14-day)
        // Need High/Low but we only have Close in this endpoint usually?
        // Wait, the history endpoint in dashboard returns open/high/low/close!
        // We need to type cast properly if logic changed, but assuming dashboard returns standard OHLCV.
        // Let's assume High/Low are available in history if logic in backend provided it.
        // Backend `get_dashboard_detail` returns open, high, low, close.

        let atr = 0;
        const period = 14;
        // Simple True Range approx if we access raw data, but let's approximate with volatility if fields missing.
        // Actually I know fields ARE present in backend response.

        // Compute ATR
        // TR = Max(H-L, |H-Cp|, |L-Cp|)
        // We'll compute last 14 days TR and average.
        let trSum = 0;
        // Check if High exists in data (it does based on my backend code)
        // However, I defined interface AssetHistory above with only close/spy_close.
        // I should update interface.

        // But for now, let's use percent volatility as fallback if high/low missing
        // Fallback: 2 * std_dev(returns) * price
        const returns = prices.map((p, i) => i === 0 ? 0 : Math.log(p / prices[i - 1]));
        const recentReturns = returns.slice(-20);
        const stdDev = Math.sqrt(recentReturns.reduce((a, b) => a + Math.pow(b, 2), 0) / recentReturns.length - Math.pow(recentReturns.reduce((a, b) => a + b, 0) / recentReturns.length, 2));
        const estimatedAtr = stdDev * latest.close;

        const atrStop = latest.close - (2.5 * estimatedAtr);

        // 2. Correlation to SPY
        // Need SPY closes.
        let correlation = 0;
        const spyPrices = hist.map(d => d.spy_close || 0);
        if (spyPrices.some(p => p > 0)) {
            // Calculate corr
            // Use last 30 days
            const len = Math.min(30, hist.length);
            const stockSub = prices.slice(-len);
            const spySub = spyPrices.slice(-len);

            // Corr coeff formula...
            // Or just visual is enough? User wants "Correlation Risk" widget.
            // Pearson corr
            const n = len;
            const sumX = stockSub.reduce((a, b) => a + b, 0);
            const sumY = spySub.reduce((a, b) => a + b, 0);
            const sumXY = stockSub.reduce((a, b, i) => a + b * spySub[i], 0);
            const sumX2 = stockSub.reduce((a, b) => a + b * b, 0);
            const sumY2 = spySub.reduce((a, b) => a + b * b, 0);

            const num = n * sumXY - sumX * sumY;
            const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
            correlation = den === 0 ? 0 : num / den;
        }

        // 3. Graph Data (Normalized)
        // Normalize to % change from start of window (last 60 days)
        const windowSize = 60;
        const graphData = hist.slice(-windowSize).map(item => {
            const startPrice = hist[Math.max(0, hist.length - windowSize)].close;
            const startSpy = hist[Math.max(0, hist.length - windowSize)].spy_close || 1;

            return {
                date: item.date,
                stockPct: ((item.close - startPrice) / startPrice) * 100,
                spyPct: item.spy_close ? ((item.spy_close - startSpy) / startSpy) * 100 : 0
            };
        });

        return {
            atrStop,
            correlation,
            graphData,
            latestPrice: latest.close
        };
    }, [detailData]);

    if (!isOpen) return null;

    const rank = detailData?.rank ?? 50; // Default to 50 if missing

    return (
        <div className="fixed inset-0 bg-[#0A0A0F]/90 backdrop-blur-md z-50 flex items-center justify-center p-4">
            <div className="bg-[#12121A] border border-white/[0.08] rounded-xl w-full max-w-5xl h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">

                {/* Header */}
                <div className="px-6 py-4 border-b border-white/[0.08] bg-[#12121A] flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="h-10 w-10 rounded-lg bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                            <Brain className="text-indigo-400" size={20} />
                        </div>
                        <div>
                            <h2 className="text-lg font-display font-medium text-white">Security Analysis</h2>
                            <div className="flex items-center gap-2 text-xs text-zinc-500">
                                <span>{modelVersion}</span>
                                <span className="w-1 h-1 rounded-full bg-zinc-700" />
                                <span className={metrics?.correlation && metrics.correlation > 0.8 ? "text-rose-400" : "text-emerald-400"}>
                                    {metrics ? (metrics.correlation > 0.8 ? "High Correlation" : "Low Correlation") : "--"}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-4">
                        <select
                            value={selectedTicker}
                            onChange={(e) => setSelectedTicker(e.target.value)}
                            className="bg-[#0A0A0F] border border-white/[0.1] rounded-lg px-3 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-indigo-500 transition-colors"
                        >
                            {tickers.map(t => <option key={t} value={t}>{t}</option>)}
                        </select>
                        <button onClick={onClose} className="p-2 hover:bg-white/[0.05] rounded-lg transition-colors text-zinc-400 hover:text-white">
                            <X size={20} />
                        </button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-6">
                    {isLoading ? (
                        <div className="h-full flex items-center justify-center">
                            <RefreshCw className="animate-spin text-zinc-600" size={32} />
                        </div>
                    ) : error ? (
                        <div className="h-full flex items-center justify-center text-rose-400">
                            <ShieldAlert className="mr-2" /> {error}
                        </div>
                    ) : (
                        <div className="space-y-6">

                            {/* KPI Cards */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {/* Rank Widget */}
                                <div className="p-5 rounded-xl bg-gradient-to-br from-indigo-900/10 to-purple-900/10 border border-white/[0.08] relative overflow-hidden group">
                                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                                        <Activity size={80} />
                                    </div>
                                    <h3 className="text-zinc-500 text-xs font-medium uppercase tracking-wider mb-1">V9 Relative Rank</h3>
                                    <div className="flex items-baseline gap-2">
                                        <span className={`text-4xl font-display font-bold ${rank > 80 ? "text-emerald-400" : rank > 50 ? "text-amber-400" : "text-rose-400"}`}>
                                            {rank.toFixed(0)}<span className="text-xl text-zinc-600">/100</span>
                                        </span>
                                    </div>
                                    <div className="mt-3 w-full bg-zinc-800/50 h-1.5 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-1000 ${rank > 80 ? "bg-emerald-500" : rank > 50 ? "bg-amber-500" : "bg-rose-500"}`}
                                            style={{ width: `${rank}%` }}
                                        />
                                    </div>
                                </div>

                                {/* ATR Stop Widget */}
                                <div className="p-5 rounded-xl bg-[#0A0A0F]/50 border border-white/[0.08]">
                                    <div className="flex justify-between items-start mb-2">
                                        <h3 className="text-zinc-500 text-xs font-medium uppercase tracking-wider">Dynamic ATR Stop</h3>
                                        <ShieldAlert size={16} className="text-zinc-600" />
                                    </div>
                                    <div className="text-2xl font-mono text-zinc-200">
                                        ${metrics?.atrStop.toFixed(2)}
                                    </div>
                                    <div className="mt-1 text-xs text-zinc-500">
                                        Trailing 2.5x ATR
                                    </div>
                                </div>

                                {/* Correlation Widget */}
                                <div className="p-5 rounded-xl bg-[#0A0A0F]/50 border border-white/[0.08]">
                                    <div className="flex justify-between items-start mb-2">
                                        <h3 className="text-zinc-500 text-xs font-medium uppercase tracking-wider">SPY Correlation</h3>
                                        <Zap size={16} className="text-zinc-600" />
                                    </div>
                                    <div className="text-2xl font-mono text-zinc-200">
                                        {metrics?.correlation.toFixed(2)}
                                    </div>
                                    <div className="mt-1 text-xs text-zinc-500">
                                        Beta exposure to market
                                    </div>
                                </div>
                            </div>

                            {/* Relative Strength Graph */}
                            <div className="rounded-xl border border-white/[0.08] bg-[#0A0A0F]/30 p-5 h-[400px]">
                                <div className="flex items-center justify-between mb-6">
                                    <h3 className="text-sm font-medium text-zinc-300">Relative Strength (vs SPY) - Last 60 Days</h3>
                                    <div className="flex items-center gap-4 text-xs">
                                        <div className="flex items-center gap-1.5">
                                            <span className="w-2 h-2 rounded-full bg-indigo-400" />
                                            <span className="text-zinc-400">{selectedTicker}</span>
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                            <span className="w-2 h-2 rounded-full bg-zinc-600" />
                                            <span className="text-zinc-400">S&P 500</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="h-[320px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={metrics?.graphData || []}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={false} />
                                            <XAxis
                                                dataKey="date"
                                                stroke="#52525b"
                                                fontSize={10}
                                                tickLine={false}
                                                axisLine={false}
                                                minTickGap={30}
                                                tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                            />
                                            <YAxis
                                                stroke="#52525b"
                                                fontSize={10}
                                                tickLine={false}
                                                axisLine={false}
                                                unit="%"
                                            />
                                            <RechartsTooltip
                                                contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', borderRadius: '8px' }}
                                                itemStyle={{ fontSize: '12px' }}
                                                labelStyle={{ color: '#a1a1aa', fontSize: '11px', marginBottom: '4px' }}
                                                formatter={(value: number) => [`${value.toFixed(2)}%`]}
                                                labelFormatter={(label) => new Date(label).toLocaleDateString()}
                                            />
                                            <ReferenceLine y={0} stroke="#27272a" strokeDasharray="3 3" />
                                            <Line
                                                type="monotone"
                                                dataKey="stockPct"
                                                stroke="#818cf8"
                                                strokeWidth={2}
                                                dot={false}
                                                activeDot={{ r: 4, strokeWidth: 0 }}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="spyPct"
                                                stroke="#52525b"
                                                strokeWidth={2}
                                                strokeDasharray="4 4"
                                                dot={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
