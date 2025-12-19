"use client";

import React from "react";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";
import SmartTooltip from "./SmartTooltip";

export interface Holding {
    ticker: string;
    quantity: number;
    entry_price: number;
    current_price: number;
    stop_loss_level: number;
    profit_pct: number;
    value: number;
}

export interface Trade {
    ticker: string;
    action: string;
    reason: string;
}

interface HoldingsTableProps {
    holdings: Holding[];
    trades: Trade[];
}

export function HoldingsTable({ holdings, trades }: HoldingsTableProps) {
    return (
        <table className="w-full text-left text-sm text-zinc-400">
            <thead className="bg-[#0A0A0F] text-xs uppercase text-zinc-500 font-medium">
                <tr>
                    <th className="px-6 py-4">Ticker</th>
                    <th className="px-6 py-4 text-right">Qty</th>
                    <th className="px-6 py-4 text-right">Entry</th>
                    <th className="px-6 py-4 text-right">Current</th>
                    <th className="px-6 py-4 text-right">Value</th>
                    <th className="px-6 py-4 text-right">PnL</th>
                    <th className="px-6 py-4 text-left pl-8 w-1/3 flex items-center gap-1">
                        Why?
                        <SmartTooltip
                            term="Trade Reasoning"
                            traderExplanation="Shows V9's AI ranking and correlation data that triggered this buy/sell decision."
                            academicExplanation="Rank: V9 Transformer relative strength score (0-1). Corr: Average correlation with other holdings. Lower corr = better diversification."
                        >
                            <Info size={14} className="text-zinc-500 hover:text-amber-400 transition-colors ml-1 cursor-help" />
                        </SmartTooltip>
                    </th>
                </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
                {holdings.length === 0 ? (
                    <tr>
                        <td colSpan={7} className="px-6 py-12 text-center text-zinc-600 italic">
                            No active holdings. Portfolio is 100% Cash.
                        </td>
                    </tr>
                ) : (
                    holdings.map((h) => {
                        // Find last buy trade reasoning if available
                        const lastBuy = [...trades].reverse().find((t) => t.ticker === h.ticker && t.action === "BUY");
                        const reason = lastBuy ? lastBuy.reason : "Rebalance";

                        return (
                            <tr key={h.ticker} className="hover:bg-white/[0.02] transition-colors group">
                                <td className="px-6 py-4 font-bold text-white font-display flex items-center gap-2">
                                    <div className="w-8 h-8 rounded bg-zinc-800 flex items-center justify-center text-xs font-mono">
                                        {h.ticker[0]}
                                    </div>
                                    {h.ticker}
                                </td>
                                <td className="px-6 py-4 text-right font-mono">{h.quantity}</td>
                                <td className="px-6 py-4 text-right font-mono text-zinc-500">${h.entry_price.toFixed(2)}</td>
                                <td className="px-6 py-4 text-right font-mono text-white">${h.current_price.toFixed(2)}</td>
                                <td className="px-6 py-4 text-right font-mono text-zinc-300 font-bold">${h.value.toFixed(2)}</td>
                                <td className="px-6 py-4 text-right font-mono">
                                    <div
                                        className={cn(
                                            "inline-flex items-center px-1.5 py-0.5 rounded",
                                            h.profit_pct >= 0 ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
                                        )}
                                    >
                                        {h.profit_pct >= 0 ? "+" : ""}
                                        {h.profit_pct.toFixed(2)}%
                                    </div>
                                </td>
                                <td className="px-6 py-4 text-left pl-8 text-xs text-amber-500/80 font-mono">{reason}</td>
                            </tr>
                        );
                    })
                )}
            </tbody>
        </table>
    );
}
