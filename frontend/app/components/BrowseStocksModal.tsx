"use client";

import React, { useState, useEffect } from "react";
import { X, Search, Plus, Check, Star, Loader2, ChevronDown, ChevronUp } from "lucide-react";

const API_BASE = "http://localhost:8000";

interface BrowseStock {
    ticker: string;
    in_watchlist: boolean;
}

interface BrowseStocksData {
    [category: string]: BrowseStock[];
}

interface BrowseStocksModalProps {
    isOpen: boolean;
    onClose: () => void;
    onAdd: (ticker: string, isFavorite: boolean) => void;
}

export default function BrowseStocksModal({ isOpen, onClose, onAdd }: BrowseStocksModalProps) {
    const [stocks, setStocks] = useState<BrowseStocksData>({});
    const [isLoading, setIsLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(["Popular", "ETFs"]));
    const [addingTicker, setAddingTicker] = useState<string | null>(null);
    const [customTicker, setCustomTicker] = useState("");
    const [addAsFavorite, setAddAsFavorite] = useState(false);

    useEffect(() => {
        if (isOpen) {
            fetchBrowseStocks();
        }
    }, [isOpen]);

    const fetchBrowseStocks = async () => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE}/stocks/browse`);
            if (res.ok) {
                const data = await res.json();
                setStocks(data);
            }
        } catch (e) {
            console.error("Failed to fetch browse stocks:", e);
        } finally {
            setIsLoading(false);
        }
    };

    const handleAddStock = async (ticker: string, isFavorite: boolean = false) => {
        setAddingTicker(ticker);
        try {
            await onAdd(ticker, isFavorite);
            // Update local state to show it's in watchlist
            setStocks(prev => {
                const updated = { ...prev };
                for (const category of Object.keys(updated)) {
                    updated[category] = updated[category].map(s =>
                        s.ticker === ticker ? { ...s, in_watchlist: true } : s
                    );
                }
                return updated;
            });
        } finally {
            setAddingTicker(null);
        }
    };

    const handleAddCustom = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!customTicker.trim()) return;
        await handleAddStock(customTicker.toUpperCase(), addAsFavorite);
        setCustomTicker("");
    };

    const toggleCategory = (category: string) => {
        setExpandedCategories(prev => {
            const next = new Set(prev);
            if (next.has(category)) {
                next.delete(category);
            } else {
                next.add(category);
            }
            return next;
        });
    };

    // Filter stocks based on search
    const filteredStocks = Object.entries(stocks).reduce<BrowseStocksData>((acc, [category, stockList]) => {
        if (!searchQuery) {
            acc[category] = stockList;
        } else {
            const filtered = stockList.filter(s =>
                s.ticker.toLowerCase().includes(searchQuery.toLowerCase())
            );
            if (filtered.length > 0) {
                acc[category] = filtered;
            }
        }
        return acc;
    }, {});

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 flex items-center justify-center bg-black/80 z-50 backdrop-blur-sm">
            <div className="bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden">
                {/* Header */}
                <div className="p-4 border-b border-zinc-800 flex justify-between items-center">
                    <div>
                        <h2 className="text-xl font-bold text-white">Browse Stocks</h2>
                        <p className="text-xs text-zinc-500 mt-1">Add stocks to your watchlist</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-white transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Search & Custom Add */}
                <div className="p-4 border-b border-zinc-800 space-y-3">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" size={18} />
                        <input
                            type="text"
                            placeholder="Search stocks..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-zinc-800 border border-zinc-700 rounded-lg pl-10 pr-4 py-2 text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                        />
                    </div>

                    <form onSubmit={handleAddCustom} className="flex gap-2">
                        <input
                            type="text"
                            placeholder="Enter custom ticker (e.g. NVDA)"
                            value={customTicker}
                            onChange={(e) => setCustomTicker(e.target.value.toUpperCase())}
                            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 font-mono"
                        />
                        <button
                            type="button"
                            onClick={() => setAddAsFavorite(!addAsFavorite)}
                            className={`px-3 py-2 rounded-lg border transition-colors flex items-center gap-1 ${addAsFavorite
                                ? "bg-amber-500/20 border-amber-500/50 text-amber-400"
                                : "bg-zinc-800 border-zinc-700 text-zinc-400"
                                }`}
                            title={addAsFavorite ? "Will be added as Favorite (gets intrinsic value)" : "Will be added without intrinsic value"}
                        >
                            <Star size={16} fill={addAsFavorite ? "currentColor" : "none"} />
                        </button>
                        <button
                            type="submit"
                            disabled={!customTicker.trim() || addingTicker === customTicker}
                            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                        >
                            {addingTicker === customTicker ? (
                                <Loader2 size={16} className="animate-spin" />
                            ) : (
                                <Plus size={16} />
                            )}
                            Add
                        </button>
                    </form>

                    <p className="text-xs text-zinc-500 flex items-center gap-2">
                        <Star size={12} className="text-amber-400" fill="currentColor" />
                        Favorites get Alpha Vantage intrinsic value calculations (limited API calls)
                    </p>
                </div>

                {/* Stock Categories */}
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                    {isLoading ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="animate-spin text-zinc-500" size={32} />
                        </div>
                    ) : Object.entries(filteredStocks).length === 0 ? (
                        <div className="text-center py-12 text-zinc-500">
                            No stocks found matching &quot;{searchQuery}&quot;
                        </div>
                    ) : (
                        Object.entries(filteredStocks).map(([category, stockList]) => (
                            <div
                                key={category}
                                className="bg-zinc-800/50 rounded-xl border border-zinc-700/50 overflow-hidden"
                            >
                                <button
                                    onClick={() => toggleCategory(category)}
                                    className="w-full px-4 py-3 flex justify-between items-center text-left hover:bg-zinc-700/30 transition-colors"
                                >
                                    <span className="font-medium text-white">{category}</span>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-zinc-500">{stockList.length} stocks</span>
                                        {expandedCategories.has(category) ? (
                                            <ChevronUp size={18} className="text-zinc-400" />
                                        ) : (
                                            <ChevronDown size={18} className="text-zinc-400" />
                                        )}
                                    </div>
                                </button>

                                {expandedCategories.has(category) && (
                                    <div className="px-4 pb-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
                                        {stockList.map((stock) => (
                                            <button
                                                key={stock.ticker}
                                                onClick={() => !stock.in_watchlist && handleAddStock(stock.ticker, addAsFavorite)}
                                                disabled={stock.in_watchlist || addingTicker === stock.ticker}
                                                className={`px-3 py-2 rounded-lg text-sm font-mono flex items-center justify-between gap-2 transition-colors ${stock.in_watchlist
                                                    ? "bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 cursor-default"
                                                    : addingTicker === stock.ticker
                                                        ? "bg-zinc-700 text-zinc-300"
                                                        : "bg-zinc-700/50 hover:bg-zinc-600 text-white"
                                                    }`}
                                            >
                                                <span>{stock.ticker}</span>
                                                {stock.in_watchlist ? (
                                                    <Check size={14} />
                                                ) : addingTicker === stock.ticker ? (
                                                    <Loader2 size={14} className="animate-spin" />
                                                ) : (
                                                    <Plus size={14} className="opacity-50" />
                                                )}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-zinc-800 flex justify-between items-center">
                    <span className="text-xs text-zinc-500">
                        {Object.values(stocks).flat().filter(s => s.in_watchlist).length} stocks in watchlist
                    </span>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg font-medium transition-colors"
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
}
