"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import AddAssetBar from "../components/AddAssetBar";
import AssetTable, { Asset } from "../components/AssetTable";
import DetailDrawer from "../components/DetailDrawer";
import DeleteConfirmationModal from "../components/DeleteConfirmationModal";
import BrowseStocksModal from "../components/BrowseStocksModal";
import { useToast } from "../components/Toast";
import AIPredictionPanel from "../components/AIPredictionPanel";
import { Grid, Star, ScanSearch } from "lucide-react";

// NOTE: Navbar already includes API_BASE, but we define it here for local fetches
const API_BASE = "http://localhost:8000";

// API response types
interface DashboardItem {
    ticker: string;
    current_price: number;
    prediction: number;
    signal: string;
    intrinsic_value?: number;
    accuracy?: boolean;
    source?: string;
    is_favorite?: boolean;
}

export default function ScannerPage() {
    const [isBrowseOpen, setIsBrowseOpen] = useState(false);
    const [assets, setAssets] = useState<Asset[]>([]);
    const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [favorites, setFavorites] = useState<Set<string>>(new Set());

    // Delete Modal State
    const [deleteTicker, setDeleteTicker] = useState<string | null>(null);

    // Refreshing State (for individual ticker refresh)
    const [refreshingTicker, setRefreshingTicker] = useState<string | null>(null);

    // Toast notifications
    const { showToast } = useToast();

    // AI Prediction Panel
    const [isAIPanelOpen, setIsAIPanelOpen] = useState(false);

    // Prevent duplicate initial fetch
    const hasFetchedRef = useRef(false);

    // Helper to detect crypto tickers
    const isCryptoTicker = (ticker: string): boolean => {
        return ticker.includes("/") || ["BTC", "ETH", "LTC", "SOL", "DOGE"].some(c => ticker.includes(c));
    };

    // Fetch dashboard data from API
    const fetchDashboard = useCallback(async () => {
        if (isLoading) return; // Prevent concurrent fetches

        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE}/dashboard`);
            if (!res.ok) throw new Error("Failed to fetch dashboard");
            const data: DashboardItem[] = await res.json();

            // Build favorites set from response
            const favSet = new Set(data.filter(item => item.is_favorite).map(item => item.ticker));
            setFavorites(favSet);

            const mappedAssets: Asset[] = data.map((item) => ({
                ticker: item.ticker,
                price: item.current_price || 0,
                prediction: item.prediction || 0,
                intrinsic: item.intrinsic_value || 0,
                accuracy: item.accuracy ?? (item.prediction > item.current_price),
                isCrypto: isCryptoTicker(item.ticker),
                source: item.source,
                isFavorite: item.is_favorite !== false,
            }));

            // Sort: Favorites first, then alphabetical
            mappedAssets.sort((a, b) => {
                if (a.isFavorite === b.isFavorite) {
                    return a.ticker.localeCompare(b.ticker);
                }
                return a.isFavorite ? -1 : 1;
            });

            setAssets(mappedAssets);
        } catch (err) {
            console.error("Failed to fetch dashboard:", err);
        } finally {
            setIsLoading(false);
        }
    }, []); // No dependencies

    // Initial load
    useEffect(() => {
        if (hasFetchedRef.current) return;
        hasFetchedRef.current = true;
        fetchDashboard();
    }, [fetchDashboard]);

    // Add asset to watchlist
    const handleAddAsset = async (ticker: string, isFavorite: boolean = true) => {
        showToast(`Adding ${ticker.toUpperCase()} to watchlist...`, "info");
        try {
            const watchlistRes = await fetch(`${API_BASE}/stocks/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ticker, is_favorite: isFavorite }),
            });

            if (!watchlistRes.ok) {
                const err = await watchlistRes.json();
                showToast(`Failed to add ${ticker}: ${err.detail || "Unknown error"}`, "error");
                return;
            }

            showToast(`${ticker.toUpperCase()} added! Fetching data...`, "success");
            setTimeout(() => fetchDashboard(), 500);

        } catch (err) {
            console.error("Failed to add asset:", err);
            showToast("Network error while adding asset.", "error");
        }
    };

    // Toggle favorite
    const handleToggleFavorite = async (ticker: string, isFavorite: boolean) => {
        try {
            const res = await fetch(`${API_BASE}/stocks/${ticker}/favorite`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ is_favorite: isFavorite }),
            });

            if (!res.ok) throw new Error("Failed to update favorite");

            setFavorites(prev => {
                const next = new Set(prev);
                if (isFavorite) next.add(ticker);
                else next.delete(ticker);
                return next;
            });

            setAssets(prev => prev.map(asset =>
                asset.ticker === ticker ? { ...asset, isFavorite } : asset
            ));

            showToast(isFavorite ? `${ticker} added to favorites` : `${ticker} removed from favorites`, "success");
        } catch (err) {
            console.error("Failed to toggle favorite:", err);
            showToast("Failed to update favorite status", "error");
        }
    };

    // Remove asset
    const handleConfirmDelete = async (cascade: boolean) => {
        if (!deleteTicker) return;

        try {
            const res = await fetch(`${API_BASE}/stocks/${deleteTicker}?cascade=${cascade}`, {
                method: "DELETE"
            });

            if (!res.ok) throw new Error("Failed to delete");

            showToast(`${deleteTicker} removed from watchlist`, "success");
            setDeleteTicker(null);
            await fetchDashboard();
        } catch (err) {
            console.error("Failed to remove asset:", err);
            showToast("Failed to remove asset", "error");
        }
    };

    // Refresh single asset
    const handleRefresh = async (ticker: string, mode: "daily" | "full" = "full") => {
        setRefreshingTicker(ticker);
        showToast(`Refreshing ${ticker} (${mode} sync)...`, "info");
        try {
            const res = await fetch(`${API_BASE}/ingest/${ticker}?mode=${mode}`, { method: "POST" });
            if (!res.ok) throw new Error("Refresh failed");
            showToast(`${ticker} data updated successfully`, "success");
            await fetchDashboard();
        } catch (err) {
            console.error(`Failed to refresh ${ticker}:`, err);
            showToast(`Failed to refresh ${ticker}`, "error");
        } finally {
            setRefreshingTicker(null);
        }
    };

    return (
        <div className="flex flex-col min-h-screen font-sans">
            <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">

                {/* Page Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-semibold text-white flex items-center gap-3">
                            <ScanSearch size={28} className="text-amber-500" />
                            V9 Market Scanner
                            <span className="text-xs px-2 py-1 bg-amber-500/20 text-amber-400 rounded-full border border-amber-500/30 flex items-center gap-1">
                                <Star size={12} fill="currentColor" />
                                {assets.filter(a => a.isFavorite !== false).length} Favorites
                            </span>
                        </h2>
                        <p className="text-zinc-400 text-sm mt-1">
                            Raw ranking data and deep analysis mode
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => setIsBrowseOpen(true)}
                            className="p-2 rounded-lg bg-zinc-800 border border-zinc-700 hover:bg-zinc-700 hover:border-zinc-600 transition-colors text-zinc-400 hover:text-white"
                            title="Browse Stocks"
                        >
                            <Grid size={18} />
                        </button>
                        <AddAssetBar onAdd={(ticker) => handleAddAsset(ticker, false)} />
                    </div>
                </div>

                <DeleteConfirmationModal
                    ticker={deleteTicker || ""}
                    isOpen={!!deleteTicker}
                    onClose={() => setDeleteTicker(null)}
                    onConfirm={handleConfirmDelete}
                />

                <BrowseStocksModal
                    isOpen={isBrowseOpen}
                    onClose={() => setIsBrowseOpen(false)}
                    onAdd={handleAddAsset}
                />

                <AIPredictionPanel
                    isOpen={isAIPanelOpen}
                    onClose={() => setIsAIPanelOpen(false)}
                />

                {assets.length === 0 && !isLoading ? (
                    <div className="flex flex-col items-center justify-center py-20 text-zinc-500">
                        <p className="text-lg mb-2">Watchlist Empty</p>
                        <button
                            onClick={() => setIsBrowseOpen(true)}
                            className="px-6 py-3 bg-amber-500 hover:bg-amber-400 text-[#0A0A0F] rounded-lg font-semibold transition-all hover:shadow-[0_0_20px_rgba(245,158,11,0.4)] flex items-center gap-2"
                        >
                            <Grid size={18} />
                            Browse Stocks
                        </button>
                    </div>
                ) : (
                    <AssetTable
                        assets={assets}
                        onSelect={setSelectedAsset}
                        onDelete={setDeleteTicker}
                        onRefresh={handleRefresh}
                        onToggleFavorite={handleToggleFavorite}
                        refreshingTicker={refreshingTicker}
                    />
                )}
            </main>

            <DetailDrawer
                asset={selectedAsset}
                onClose={() => setSelectedAsset(null)}
            />
        </div>
    );
}
