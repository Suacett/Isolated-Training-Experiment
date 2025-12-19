"use client";

import React from "react";
import { DollarSign, Unlock, Percent, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import SmartTooltip from "./SmartTooltip";

export interface MetricsData {
    daily_pnl?: number;
    sharpe_ratio?: number;
    alpha?: number;
}

interface MetricsCardsProps {
    summary: MetricsData | null;
}

export function MetricsCards({ summary }: MetricsCardsProps) {
    return (
        <div className="space-y-4">
            {/* Daily PnL */}
            <div className="p-5 rounded-xl bg-[#12121A] border border-white/[0.08] flex flex-col justify-center h-[calc(33%-11px)]">
                <div className="flex items-center gap-2 mb-2">
                    <DollarSign size={16} className="text-emerald-500" />
                    <span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Daily PnL</span>
                    <SmartTooltip
                        term="Daily Profit & Loss"
                        traderExplanation="Your profit or loss for the current trading session."
                        academicExplanation="Calculated as (Current Value - Yesterday Close). Updates in real-time."
                    >
                        <Info size={14} className="text-zinc-500 hover:text-amber-400 transition-colors ml-1 cursor-help" />
                    </SmartTooltip>
                </div>
                <div className={cn("text-2xl font-mono font-bold", (summary?.daily_pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400")}>
                    {summary ? `${(summary.daily_pnl || 0) >= 0 ? "+" : ""}$${Math.abs(summary.daily_pnl || 0).toLocaleString()}` : "$-.--"}
                </div>
            </div>

            {/* Sharpe Ratio */}
            <div className="p-5 rounded-xl bg-[#12121A] border border-white/[0.08] flex flex-col justify-center h-[calc(33%-11px)]">
                <div className="flex items-center gap-2 mb-2">
                    <Unlock size={16} className="text-amber-500" />
                    <span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Sharpe Ratio (1y)</span>
                    <SmartTooltip
                        term="Sharpe Ratio"
                        traderExplanation="Risk-adjusted return metric. Higher is better. Values > 1.0 indicate good performance."
                        academicExplanation="Sharpe = (Rp - Rf) / σp. Measures excess return per unit of volatility."
                    >
                        <Info size={14} className="text-zinc-500 hover:text-amber-400 transition-colors ml-1 cursor-help" />
                    </SmartTooltip>
                </div>
                <div className="text-2xl font-mono font-bold text-zinc-200">
                    {summary?.sharpe_ratio !== undefined ? summary.sharpe_ratio.toFixed(2) : "--"}
                </div>
            </div>

            {/* Alpha */}
            <div className="p-5 rounded-xl bg-[#12121A] border border-white/[0.08] flex flex-col justify-center h-[calc(33%-11px)]">
                <div className="flex items-center gap-2 mb-2">
                    <Percent size={16} className="text-indigo-500" />
                    <span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Alpha</span>
                    <SmartTooltip
                        term="Alpha (α)"
                        traderExplanation="Excess return vs S&P 500 benchmark. Positive alpha means beating the market."
                        academicExplanation="Alpha (α) = Rp - [Rf + β(Rm - Rf)]. Active return on investment adjusted for market risk."
                    >
                        <Info size={14} className="text-zinc-500 hover:text-amber-400 transition-colors ml-1 cursor-help" />
                    </SmartTooltip>
                </div>
                <div className={cn("text-2xl font-mono font-bold", (summary?.alpha || 0) >= 0 ? "text-indigo-400" : "text-rose-400")}>
                    {summary?.alpha !== undefined ? `${summary.alpha >= 0 ? "+" : ""}${summary.alpha.toFixed(2)}%` : "--"}
                </div>
            </div>
        </div>
    );
}
