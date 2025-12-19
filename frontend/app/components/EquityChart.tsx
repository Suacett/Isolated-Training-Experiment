"use client";

import React from "react";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip } from "recharts";
import { TrendingUp, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { cn } from "@/lib/utils";

// Types
export interface EquityPoint {
    date: string;
    value: number;
    cash: number;
    equity: number;
}

export interface PortfolioSummary {
    total_value: number;
    daily_pnl?: number;
    daily_pnl_pct?: number;
}

export type TimeRangeKey = "1M" | "3M" | "6M" | "1Y" | "ALL";

interface TimeRange {
    key: TimeRangeKey;
    label: string;
    days: number;
}

export const TIME_RANGES: TimeRange[] = [
    { key: "1M", label: "1M", days: 30 },
    { key: "3M", label: "3M", days: 90 },
    { key: "6M", label: "6M", days: 180 },
    { key: "1Y", label: "1Y", days: 365 },
    { key: "ALL", label: "All", days: 9999 },
];

interface EquityChartProps {
    equityHistory: EquityPoint[];
    summary: PortfolioSummary | null;
    selectedRange: TimeRangeKey;
    onRangeChange: (range: TimeRangeKey) => void;
}

export function EquityChart({ equityHistory, summary, selectedRange, onRangeChange }: EquityChartProps) {
    // Filter equity history based on selected time range
    const filteredEquityHistory = React.useMemo(() => {
        if (!equityHistory.length) return [];

        const range = TIME_RANGES.find((r) => r.key === selectedRange);
        if (!range) return equityHistory;

        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - range.days);

        return equityHistory.filter((point) => new Date(point.date) >= cutoffDate);
    }, [equityHistory, selectedRange]);

    return (
        <div className="lg:col-span-2 p-6 rounded-2xl bg-[#12121A] border border-white/[0.08] relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-6 opacity-5 group-hover:opacity-10 transition-opacity">
                <TrendingUp size={100} />
            </div>

            <div className="relative z-10 flex flex-col h-[380px]">
                <div className="flex justify-between items-start mb-6">
                    <div>
                        <h2 className="text-zinc-400 text-sm font-medium uppercase tracking-wider mb-1">Portfolio Equity</h2>
                        <div className="flex items-baseline gap-4">
                            <span className="text-4xl font-display font-bold text-white">
                                {summary ? `$${summary.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "$-.--"}
                            </span>
                            {summary && (
                                <div
                                    className={cn(
                                        "flex items-center text-sm font-medium px-2 py-1 rounded-full bg-white/5",
                                        (summary.daily_pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                                    )}
                                >
                                    {(summary.daily_pnl || 0) >= 0 ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
                                    {Math.abs(summary.daily_pnl_pct || 0).toFixed(2)}%
                                </div>
                            )}
                        </div>
                        <p className="text-[10px] text-zinc-600 mt-2 font-mono">
                            *Historical performance is simulated (Backtest). Future results may vary.
                        </p>
                    </div>

                    {/* Time Range Selector */}
                    <div className="flex items-center gap-1">
                        {TIME_RANGES.map((range) => (
                            <button
                                key={range.key}
                                onClick={() => onRangeChange(range.key)}
                                className={cn(
                                    "px-3 py-1.5 text-xs font-medium rounded-md transition-all",
                                    selectedRange === range.key
                                        ? "bg-amber-500 text-[#0A0A0F] shadow-[0_0_10px_rgba(245,158,11,0.3)]"
                                        : "bg-zinc-800/50 text-zinc-400 hover:bg-zinc-700 hover:text-white border border-zinc-700/50"
                                )}
                            >
                                {range.label}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="flex-1 w-full min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={filteredEquityHistory}>
                            <defs>
                                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={false} />
                            <XAxis
                                dataKey="date"
                                stroke="#52525b"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                minTickGap={40}
                                tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                            />
                            <YAxis
                                stroke="#52525b"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                domain={["auto", "auto"]}
                                tickFormatter={(val) => `$${val.toLocaleString()}`}
                                width={60}
                            />
                            <RechartsTooltip
                                contentStyle={{ backgroundColor: "#09090b", borderColor: "#27272a", borderRadius: "8px" }}
                                itemStyle={{ fontSize: "12px", color: "#818cf8" }}
                                labelStyle={{ color: "#a1a1aa", fontSize: "11px", marginBottom: "4px" }}
                                formatter={(value: number) => [`$${value.toLocaleString()}`, "Equity"]}
                                labelFormatter={(label) => new Date(label).toLocaleDateString()}
                            />
                            <Area type="monotone" dataKey="value" stroke="#818cf8" strokeWidth={2} fillOpacity={1} fill="url(#colorEquity)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
