"use client";

import { useState, useMemo } from "react";
import { Bitcoin, Star, Loader2, TrendingUp, TrendingDown, Minus, Trash2 } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { getApiUrl } from "@/config/api";

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export interface Asset {
    ticker: string;
    price: number;
    // prediction: number; // Removed in V9
    rank?: number | null; // V9 Rank (0-100)
    signal?: string; // "BUY", "sell", "hold"
    intrinsic: number;
    accuracy: boolean; // true = green dot, false = red dot
    isCrypto: boolean;
    source?: string;
    isFavorite?: boolean;
}

interface AssetTableProps {
    assets: Asset[];
    onSelect: (asset: Asset) => void;
    onDelete: (ticker: string) => void;
    onRefresh: (ticker: string, mode?: "daily" | "full") => void;
    onToggleFavorite?: (ticker: string, isFavorite: boolean) => void;
    refreshingTicker?: string | null;
}

export default function AssetTable({ assets, onSelect, onDelete, onRefresh, onToggleFavorite, refreshingTicker }: AssetTableProps) {
    const [togglingFavorite, setTogglingFavorite] = useState<string | null>(null);

    // Sort assets by Rank (Descending) by default
    const sortedAssets = useMemo(() => {
        return [...assets].sort((a, b) => {
            const rankA = a.rank ?? -1;
            const rankB = b.rank ?? -1;
            return rankB - rankA;
        });
    }, [assets]);

    const handleToggleFavorite = async (e: React.MouseEvent, ticker: string, currentFavorite: boolean) => {
        e.stopPropagation();
        if (!onToggleFavorite) return;

        setTogglingFavorite(ticker);
        try {
            await onToggleFavorite(ticker, !currentFavorite);
        } finally {
            setTogglingFavorite(null);
        }
    };

    const getSignalColor = (signal?: string) => {
        if (!signal) return "text-zinc-500";
        const s = signal.toUpperCase();
        if (s.includes("BUY") || s.includes("STRONG")) return "text-emerald-400";
        if (s.includes("SELL") || s.includes("HIGH CORR")) return "text-rose-400";
        return "text-amber-400"; // HOLD / NEUTRAL
    };

    const getSignalBadge = (signal?: string) => {
        if (!signal) return null;
        const s = signal.toUpperCase();
        if (s.includes("BUY")) return <span className="flex items-center gap-1"><TrendingUp size={14} /> {s}</span>;
        if (s.includes("SELL")) return <span className="flex items-center gap-1"><TrendingDown size={14} /> {s}</span>;
        return <span className="flex items-center gap-1"><Minus size={14} /> {s}</span>;
    };

    return (
        <div className="w-full overflow-hidden rounded-xl border border-white/[0.08] bg-[#0A0A0F]/80 backdrop-blur-xl shadow-2xl">
            <table className="w-full text-left text-sm text-zinc-400 table-fixed">
                <thead className="bg-[#12121A] text-xs uppercase text-zinc-500 border-b border-white/[0.08] font-display tracking-wider">
                    <tr>
                        <th className="px-4 py-4 font-medium w-[50px]"></th>
                        <th className="px-4 py-4 font-medium w-[120px]">Ticker</th>
                        <th className="px-6 py-4 font-medium text-right">Price</th>
                        <th className="px-6 py-4 font-medium text-center">V9 Rank</th>
                        <th className="px-6 py-4 font-medium text-center">Signal Status</th>
                        <th className="px-6 py-4 font-medium text-right">Intrinsic Val</th>
                        <th className="px-6 py-4 font-medium text-center w-[60px]"></th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                    {sortedAssets.map((asset) => (
                        <tr
                            key={asset.ticker}
                            className="group hover:bg-white/[0.02] transition-all cursor-pointer"
                            onClick={() => onSelect(asset)}
                        >
                            {/* Favorite Star */}
                            <td className="px-4 py-4">
                                <button
                                    onClick={(e) => handleToggleFavorite(e, asset.ticker, asset.isFavorite ?? true)}
                                    disabled={togglingFavorite === asset.ticker}
                                    className={cn(
                                        "p-1.5 rounded-full transition-all hover:bg-white/10",
                                        asset.isFavorite !== false
                                            ? "text-amber-400"
                                            : "text-zinc-700 hover:text-zinc-400"
                                    )}
                                >
                                    {togglingFavorite === asset.ticker ? (
                                        <Loader2 size={16} className="animate-spin" />
                                    ) : (
                                        <Star size={16} fill={asset.isFavorite !== false ? "currentColor" : "none"} />
                                    )}
                                </button>
                            </td>
                            <td className="px-4 py-4 font-medium text-zinc-200">
                                <div className="flex items-center gap-2">
                                    {asset.isCrypto && <Bitcoin size={16} className="text-[#F7931A]" />}
                                    <span className="font-display tracking-tight text-base">{asset.ticker}</span>
                                </div>
                            </td>
                            <td className="px-6 py-4 text-right">
                                <div className="flex flex-col items-end">
                                    <span className="font-mono text-zinc-100">${asset.price.toFixed(2)}</span>
                                    {asset.source && (
                                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800/50 text-zinc-500 uppercase">
                                            {asset.source}
                                        </span>
                                    )}
                                </div>
                            </td>

                            {/* V9 Rank Column */}
                            <td className="px-6 py-4 text-center">
                                <div className="flex flex-col items-center gap-1">
                                    {asset.rank !== null && asset.rank !== undefined ? (
                                        <>
                                            <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                                <div
                                                    className={cn("h-full rounded-full transition-all duration-500",
                                                        asset.rank > 80 ? "bg-emerald-500" :
                                                            asset.rank > 50 ? "bg-amber-500" : "bg-rose-500"
                                                    )}
                                                    style={{ width: `${asset.rank}%` }}
                                                />
                                            </div>
                                            <span className={cn("text-xs font-mono font-bold",
                                                asset.rank > 80 ? "text-emerald-400" :
                                                    asset.rank > 50 ? "text-amber-400" : "text-rose-400"
                                            )}>
                                                {asset.rank.toFixed(0)}/100
                                            </span>
                                        </>
                                    ) : (
                                        <span className="text-zinc-700 text-xs">--</span>
                                    )}
                                </div>
                            </td>

                            {/* Signal Status Column */}
                            <td className="px-6 py-4 text-center">
                                <div className={cn(
                                    "inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium border border-white/5 bg-white/5 backdrop-blur-sm",
                                    getSignalColor(asset.signal)
                                )}>
                                    {getSignalBadge(asset.signal) || "--"}
                                </div>
                            </td>

                            <td className="px-6 py-4 text-right font-mono">
                                {asset.isFavorite !== false && asset.intrinsic > 0 ? (
                                    <span className={cn(
                                        asset.intrinsic > asset.price ? "text-emerald-400" : "text-zinc-500"
                                    )}>
                                        ${asset.intrinsic.toFixed(2)}
                                    </span>
                                ) : (
                                    <span className="text-zinc-700 text-xs">N/A</span>
                                )}
                            </td>

                            {/* Actions / Delete */}
                            <td className="px-6 py-4 text-center">
                                <div className="flex justify-center group/actions">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onDelete(asset.ticker);
                                        }}
                                        className="p-2 text-zinc-600 hover:text-rose-500 hover:bg-rose-500/10 rounded-full transition-colors opacity-0 group-hover:opacity-100"
                                        title="Remove from Watchlist"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
