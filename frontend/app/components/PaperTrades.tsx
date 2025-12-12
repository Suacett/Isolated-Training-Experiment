import React from "react";
import { ArrowDownLeft, ArrowUpRight, Ban, RefreshCcw } from "lucide-react";

interface Trade {
    date: string;
    action: string;
    ticker: string;
    price: number;
    quantity: number;
    reason: string;
    profit_loss: number | null;
}

interface PaperTradesProps {
    trades: Trade[];
}

export default function PaperTrades({ trades }: PaperTradesProps) {
    if (!trades || trades.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-10 text-zinc-500 border border-white/[0.05] rounded-xl bg-[#12121A]/50">
                <p>No trade history</p>
            </div>
        );
    }

    const getReasonBadge = (reason: string) => {
        switch (reason) {
            case "RANK_ENTRY":
                return <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/10 text-purple-400 border border-purple-500/20 uppercase">Rank Entry</span>;
            case "ATR_EXIT":
                return <span className="px-1.5 py-0.5 rounded text-[10px] bg-red-500/10 text-red-400 border border-red-500/20 uppercase">Stop Loss</span>;
            case "REBALANCE_EXIT":
                return <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-500/10 text-blue-400 border border-blue-500/20 uppercase">Rebalance</span>;
            default:
                return <span className="text-zinc-500 text-xs">{reason}</span>;
        }
    };

    const formatTime = (dateStr: string) => {
        try {
            const d = new Date(dateStr);
            return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch {
            return dateStr;
        }
    };

    return (
        <div className="bg-[#12121A]/50 border border-white/[0.08] rounded-xl overflow-hidden backdrop-blur-sm">
            <div className="py-3 px-4 border-b border-white/[0.08] flex items-center justify-between bg-white/[0.02]">
                <h3 className="text-white font-medium text-sm flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-zinc-500"></span>
                    Recent Trades
                </h3>
                <span className="text-xs text-zinc-500 font-mono">Last {trades.length}</span>
            </div>

            <div className="overflow-x-auto overflow-y-auto max-h-[600px]">
                <table className="w-full text-left border-collapse">
                    <thead className="sticky top-0 bg-[#12121A] z-10 shadow-sm">
                        <tr className="text-xs text-zinc-500 uppercase font-mono border-b border-white/[0.05]">
                            <th className="py-3 px-4 font-normal">Date</th>
                            <th className="py-3 px-4 font-normal">Action</th>
                            <th className="py-3 px-4 font-normal">Ticker</th>
                            <th className="py-3 px-4 font-normal text-right">Qty</th>
                            <th className="py-3 px-4 font-normal text-right">Price</th>
                            <th className="py-3 px-4 font-normal text-right">Total</th>
                            <th className="py-3 px-4 font-normal">Reason</th>
                            <th className="py-3 px-4 font-normal text-right">Realized P/L</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.05]">
                        {trades.map((t, idx) => (
                            <tr key={idx} className="group hover:bg-white/[0.02] transition-colors">
                                <td className="py-2.5 px-4 text-zinc-500 text-xs font-mono">
                                    {formatTime(t.date)}
                                </td>
                                <td className="py-2.5 px-4">
                                    <span className={`flex items-center gap-1 text-xs font-bold uppercase ${t.action === "BUY" ? "text-emerald-400" : "text-red-400"
                                        }`}>
                                        {t.action === "BUY" ? <ArrowUpRight size={12} /> : <ArrowDownLeft size={12} />}
                                        {t.action}
                                    </span>
                                </td>
                                <td className="py-2.5 px-4 font-bold text-white text-sm">
                                    {t.ticker}
                                </td>
                                <td className="py-2.5 px-4 text-right text-zinc-300 font-mono text-sm">
                                    {t.quantity.toFixed(2)}
                                </td>
                                <td className="py-2.5 px-4 text-right text-zinc-400 font-mono text-sm">
                                    ${t.price.toFixed(2)}
                                </td>
                                <td className="py-2.5 px-4 text-right text-zinc-300 font-mono text-sm">
                                    ${(t.price * t.quantity).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </td>
                                <td className="py-2.5 px-4">
                                    {getReasonBadge(t.reason)}
                                </td>
                                <td className="py-2.5 px-4 text-right text-sm">
                                    {t.profit_loss !== null ? (
                                        <span className={`font-mono ${t.profit_loss > 0 ? "text-emerald-400" : t.profit_loss < 0 ? "text-red-400" : "text-zinc-400"
                                            }`}>
                                            {t.profit_loss > 0 ? "+" : ""}${t.profit_loss.toFixed(2)}
                                        </span>
                                    ) : (
                                        <span className="text-zinc-700">-</span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
