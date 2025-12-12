import React from "react";

interface Holding {
    ticker: string;
    quantity: number;
    entry_price: number;
    current_price: number;
    stop_loss_level: number;
    highest_price: number;
    profit_pct: number;
    value: number;
}

interface PaperHoldingsProps {
    holdings: Holding[];
}

export default function PaperHoldings({ holdings }: PaperHoldingsProps) {
    if (!holdings || holdings.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-10 text-zinc-500 border border-white/[0.05] rounded-xl bg-[#12121A]/50">
                <p>No active positions</p>
            </div>
        );
    }

    return (
        <div className="bg-[#12121A]/50 border border-white/[0.08] rounded-xl overflow-hidden backdrop-blur-sm">
            <div className="py-3 px-4 border-b border-white/[0.08] flex items-center justify-between bg-white/[0.02]">
                <h3 className="text-white font-medium text-sm flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></span>
                    Active Holdings
                </h3>
                <span className="text-xs text-zinc-500 font-mono">{holdings.length} Positions</span>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="text-xs text-zinc-500 uppercase font-mono border-b border-white/[0.05]">
                            <th className="py-3 px-4 font-normal">Ticker</th>
                            <th className="py-3 px-4 font-normal text-right">Shares</th>
                            <th className="py-3 px-4 font-normal text-right">Avg Cost</th>
                            <th className="py-3 px-4 font-normal text-right">Price</th>
                            <th className="py-3 px-4 font-normal text-right">Value</th>
                            <th className="py-3 px-4 font-normal text-right">P/L %</th>
                            <th className="py-3 px-4 font-normal text-right">Stop Loss</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.05]">
                        {holdings.map((h) => (
                            <tr key={h.ticker} className="group hover:bg-white/[0.02] transition-colors">
                                <td className="py-3 px-4">
                                    <span className="font-bold text-white text-sm tracking-wide">{h.ticker}</span>
                                </td>
                                <td className="py-3 px-4 text-right text-zinc-300 font-mono text-sm">
                                    {h.quantity.toFixed(2)}
                                </td>
                                <td className="py-3 px-4 text-right text-zinc-400 font-mono text-sm">
                                    ${h.entry_price.toFixed(2)}
                                </td>
                                <td className="py-3 px-4 text-right text-white font-mono text-sm text-shadow-sm">
                                    ${h.current_price.toFixed(2)}
                                </td>
                                <td className="py-3 px-4 text-right font-mono text-sm text-white">
                                    ${h.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </td>
                                <td className="py-3 px-4 text-right text-sm">
                                    <span
                                        className={`inline-block px-1.5 py-0.5 rounded text-xs font-bold ${h.profit_pct >= 0
                                                ? "text-emerald-400 bg-emerald-500/10 border border-emerald-500/20"
                                                : "text-red-400 bg-red-500/10 border border-red-500/20"
                                            }`}
                                    >
                                        {h.profit_pct > 0 ? "+" : ""}{h.profit_pct.toFixed(2)}%
                                    </span>
                                </td>
                                <td className="py-3 px-4 text-right text-amber-500/80 font-mono text-sm">
                                    ${h.stop_loss_level.toFixed(2)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
