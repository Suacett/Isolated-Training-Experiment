"use client";

import React from "react";
import { cn } from "@/lib/utils";

export interface Trade {
    id: number;
    date: string;
    action: string;
    ticker: string;
    price: number;
    quantity: number;
    reason: string;
    profit_loss?: number;
}

interface TradeLogsProps {
    trades: Trade[];
}

export function TradeLogs({ trades }: TradeLogsProps) {
    return (
        <table className="w-full text-left text-sm text-zinc-400">
            <thead className="bg-[#0A0A0F] text-xs uppercase text-zinc-500 font-medium">
                <tr>
                    <th className="px-6 py-4">Date</th>
                    <th className="px-6 py-4">Ticker</th>
                    <th className="px-6 py-4">Action</th>
                    <th className="px-6 py-4 text-right">Price</th>
                    <th className="px-6 py-4 text-right">Qty</th>
                    <th className="px-6 py-4">Reason</th>
                    <th className="px-6 py-4 text-right">Result</th>
                </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
                {trades.length === 0 ? (
                    <tr>
                        <td colSpan={7} className="px-6 py-12 text-center text-zinc-600 italic">
                            No trade history available yet.
                        </td>
                    </tr>
                ) : (
                    trades.map((t) => (
                        <tr key={t.id} className="hover:bg-white/[0.02] transition-colors">
                            <td className="px-6 py-4 font-mono text-zinc-500 text-xs text-nowrap">
                                {new Date(t.date).toLocaleDateString()}
                            </td>
                            <td className="px-6 py-4 font-bold text-white font-display">{t.ticker}</td>
                            <td className="px-6 py-4">
                                <span
                                    className={cn(
                                        "px-2 py-0.5 rounded text-[10px] font-bold border",
                                        t.action === "BUY"
                                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                            : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                                    )}
                                >
                                    {t.action}
                                </span>
                            </td>
                            <td className="px-6 py-4 text-right font-mono">${t.price.toFixed(2)}</td>
                            <td className="px-6 py-4 text-right font-mono">{t.quantity}</td>
                            <td className="px-6 py-4 text-xs text-amber-500/80 font-mono max-w-xs truncate" title={t.reason}>
                                {t.reason}
                            </td>
                            <td className="px-6 py-4 text-right font-mono">
                                {t.profit_loss !== null && t.profit_loss !== undefined ? (
                                    <span className={t.profit_loss >= 0 ? "text-emerald-400" : "text-rose-400"}>
                                        {t.profit_loss >= 0 ? "+" : ""}
                                        {t.profit_loss.toFixed(2)}
                                    </span>
                                ) : (
                                    "-"
                                )}
                            </td>
                        </tr>
                    ))
                )}
            </tbody>
        </table>
    );
}
