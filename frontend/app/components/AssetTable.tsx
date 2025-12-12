"use client";

import { useState } from "react";
import { Bitcoin, CheckCircle2, XCircle, Trash2, Star, Loader2 } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export interface Asset {
    ticker: string;
    price: number;
    prediction: number;
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

const API_BASE = "http://localhost:8000";

export default function AssetTable({ assets, onSelect, onDelete, onRefresh, onToggleFavorite, refreshingTicker }: AssetTableProps) {
    const [togglingFavorite, setTogglingFavorite] = useState<string | null>(null);

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

    return (
        <div className="w-full overflow-hidden rounded-xl border border-white/[0.08] bg-[#1A1A24]/60 backdrop-blur-md">
            <table className="w-full text-left text-sm text-zinc-400 table-fixed">
                <thead className="bg-[#12121A] text-xs uppercase text-zinc-500 border-b border-white/[0.08]">
                    <tr>
                        <th className="px-4 py-3 font-medium w-[50px]"></th>
                        <th className="px-4 py-3 font-medium w-[120px]">Ticker</th>
                        <th className="px-6 py-3 font-medium text-right">Price</th>
                        <th className="px-6 py-3 font-medium text-right">AI Prediction</th>
                        <th className="px-6 py-3 font-medium text-right">Intrinsic Val</th>
                        <th className="px-6 py-3 font-medium text-center w-[120px]">Accuracy</th>
                        <th className="px-6 py-3 font-medium text-center w-[120px]">Actions</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800">
                    {assets.map((asset) => (
                        <tr
                            key={asset.ticker}
                            className="group hover:bg-zinc-800 transition-colors"
                        >
                            {/* Favorite Star */}
                            <td className="px-4 py-4">
                                <button
                                    onClick={(e) => handleToggleFavorite(e, asset.ticker, asset.isFavorite ?? true)}
                                    disabled={togglingFavorite === asset.ticker}
                                    className={cn(
                                        "p-1.5 rounded-full transition-all",
                                        asset.isFavorite !== false
                                            ? "text-amber-400 hover:text-amber-300"
                                            : "text-zinc-600 hover:text-zinc-400"
                                    )}
                                    title={asset.isFavorite !== false ? "Remove from Favorites (stops intrinsic value)" : "Add to Favorites (enables intrinsic value)"}
                                >
                                    {togglingFavorite === asset.ticker ? (
                                        <Loader2 size={16} className="animate-spin" />
                                    ) : (
                                        <Star size={16} fill={asset.isFavorite !== false ? "currentColor" : "none"} />
                                    )}
                                </button>
                            </td>
                            <td
                                className="px-4 py-4 font-medium text-zinc-200 flex items-center gap-2 cursor-pointer"
                                onClick={() => onSelect(asset)}
                            >
                                {asset.isCrypto && <Bitcoin size={16} className="text-orange-500" />}
                                {asset.ticker}
                            </td>
                            <td
                                className="px-6 py-4 text-right font-mono text-zinc-300 cursor-pointer"
                                onClick={() => onSelect(asset)}
                            >
                                <div className="flex flex-col items-end">
                                    <span>${asset.price.toFixed(2)}</span>
                                    {asset.source && (
                                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500">
                                            {asset.source}
                                        </span>
                                    )}
                                </div>
                            </td>
                            <td
                                className="px-6 py-4 text-right font-mono cursor-pointer"
                                onClick={() => onSelect(asset)}
                            >
                                <span
                                    className={cn(
                                        asset.prediction > asset.price
                                            ? "text-emerald-400"
                                            : "text-rose-400"
                                    )}
                                >
                                    ${asset.prediction.toFixed(2)}
                                </span>
                            </td>
                            <td
                                className="px-6 py-4 text-right font-mono cursor-pointer"
                                onClick={() => onSelect(asset)}
                            >
                                {asset.isFavorite !== false && asset.intrinsic > 0 ? (
                                    <span className="text-zinc-300">${asset.intrinsic.toFixed(2)}</span>
                                ) : (
                                    <span className="text-zinc-600 text-xs">N/A</span>
                                )}
                            </td>
                            <td className="px-6 py-4" onClick={() => onSelect(asset)}>
                                <div className="flex justify-center cursor-pointer">
                                    <div
                                        className={cn(
                                            "flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border",
                                            asset.accuracy
                                                ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
                                                : "bg-rose-500/10 text-rose-500 border-rose-500/20"
                                        )}
                                    >
                                        <div
                                            className={cn(
                                                "w-1.5 h-1.5 rounded-full",
                                                asset.accuracy ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
                                            )}
                                        />
                                        {asset.accuracy ? "Hit" : "Miss"}
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="flex flex-row gap-2 justify-center">
                                    <div className="relative group/sync">
                                        <button
                                            disabled={refreshingTicker === asset.ticker}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                            }}
                                            className={`p-2 rounded-full transition-colors ${refreshingTicker === asset.ticker
                                                ? "text-emerald-500 bg-emerald-500/10 opacity-100"
                                                : "text-zinc-500 hover:text-emerald-500 hover:bg-emerald-500/10 opacity-0 group-hover:opacity-100"
                                                }`}
                                            title="Sync Options"
                                        >
                                            <svg
                                                xmlns="http://www.w3.org/2000/svg"
                                                width="16"
                                                height="16"
                                                viewBox="0 0 24 24"
                                                fill="none"
                                                stroke="currentColor"
                                                strokeWidth="2"
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                className={refreshingTicker === asset.ticker ? "animate-spin" : ""}
                                            >
                                                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                                                <path d="M21 3v5h-5" />
                                                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                                                <path d="M3 21v-5h5" />
                                            </svg>
                                        </button>

                                        {/* Dropdown Menu */}
                                        <div className="absolute right-0 top-full pt-2 w-32 z-50 hidden group-hover/sync:block">
                                            <div className="bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl overflow-hidden">
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        onRefresh(asset.ticker, "daily");
                                                    }}
                                                    className="w-full px-4 py-2 text-left text-xs text-zinc-400 hover:bg-zinc-800 hover:text-emerald-400 transition-colors"
                                                >
                                                    Daily Sync (Fast)
                                                </button>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        onRefresh(asset.ticker, "full");
                                                    }}
                                                    className="w-full px-4 py-2 text-left text-xs text-zinc-400 hover:bg-zinc-800 hover:text-emerald-400 transition-colors border-t border-zinc-800"
                                                >
                                                    Full Sync (Slow)
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onDelete(asset.ticker);
                                        }}
                                        className="p-2 text-zinc-500 hover:text-rose-500 hover:bg-rose-500/10 rounded-full transition-colors opacity-0 group-hover:opacity-100"
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
