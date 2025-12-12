"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import SetupModal from "./components/SetupModal";
import AddAssetBar from "./components/AddAssetBar";
import AssetTable, { Asset } from "./components/AssetTable";
import DetailDrawer from "./components/DetailDrawer";
import LogViewer from "./components/LogViewer";
import DeleteConfirmationModal from "./components/DeleteConfirmationModal";
import BrowseStocksModal from "./components/BrowseStocksModal";
import { useToast } from "./components/Toast";
import AIPredictionPanel from "./components/AIPredictionPanel";
import ModelPlayground from "./components/ModelPlayground";
import { RefreshCw, Settings, Terminal, Brain, FlaskConical, Grid, Star } from "lucide-react";

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

interface WatchlistItem {
  ticker: string;
  is_favorite: boolean;
}

export default function Home() {
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [isSetupOpen, setIsSetupOpen] = useState(false);
  const [isLogsOpen, setIsLogsOpen] = useState(false);
  const [isBrowseOpen, setIsBrowseOpen] = useState(false);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  const [lastSync, setLastSync] = useState<Date | null>(null);

  // Delete Modal State
  const [deleteTicker, setDeleteTicker] = useState<string | null>(null);

  // Refreshing State (for individual ticker refresh)
  const [refreshingTicker, setRefreshingTicker] = useState<string | null>(null);

  // Toast notifications
  const { showToast } = useToast();

  // AI Prediction Panel
  const [isAIPanelOpen, setIsAIPanelOpen] = useState(false);

  // Model Playground Page
  const [isPlaygroundOpen, setIsPlaygroundOpen] = useState(false);

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
  }, []); // No dependencies - doesn't depend on changing state

  // Sync data from Alpaca (offline-first: only sync when user clicks)
  const handleSyncData = async () => {
    setIsSyncing(true);
    try {
      const res = await fetch(`${API_BASE}/ingest/all`, { method: "POST" });
      if (!res.ok) throw new Error("Sync failed");
      setLastSync(new Date());
      // Refresh dashboard after sync
      await fetchDashboard();
    } catch (err) {
      console.error("Failed to sync data:", err);
    } finally {
      setIsSyncing(false);
    }
  };

  // Initial load - only fetch once
  useEffect(() => {
    if (hasFetchedRef.current) return;

    fetch(`${API_BASE}/status`)
      .then((res) => res.json())
      .then((data) => {
        setConfigured(data.configured);
        if (data.configured && !hasFetchedRef.current) {
          hasFetchedRef.current = true;
          fetchDashboard();
        }
      })
      .catch((err) => console.error("Failed to fetch status", err));
  }, [fetchDashboard]);

  // Add asset to watchlist and ingest data
  const handleAddAsset = async (ticker: string, isFavorite: boolean = true) => {
    showToast(`Adding ${ticker.toUpperCase()} to watchlist...`, "info");
    try {
      // 1. Add to watchlist (using new stocks router)
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

      // Update favorites set
      if (isFavorite) {
        setFavorites(prev => new Set([...prev, ticker.toUpperCase()]));
      }

      // Refresh dashboard after short delay to allow background ingestion
      setTimeout(() => fetchDashboard(), 500);

    } catch (err) {
      console.error("Failed to add asset:", err);
      showToast("Network error while adding asset.", "error");
    }
  };

  // Toggle favorite status
  const handleToggleFavorite = async (ticker: string, isFavorite: boolean) => {
    try {
      const res = await fetch(`${API_BASE}/stocks/${ticker}/favorite`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_favorite: isFavorite }),
      });

      if (!res.ok) throw new Error("Failed to update favorite");

      // Update local state
      setFavorites(prev => {
        const next = new Set(prev);
        if (isFavorite) {
          next.add(ticker);
        } else {
          next.delete(ticker);
        }
        return next;
      });

      // Update assets state
      setAssets(prev => prev.map(asset =>
        asset.ticker === ticker ? { ...asset, isFavorite } : asset
      ));

      showToast(
        isFavorite
          ? `${ticker} added to favorites (intrinsic value enabled)`
          : `${ticker} removed from favorites`,
        "success"
      );
    } catch (err) {
      console.error("Failed to toggle favorite:", err);
      showToast("Failed to update favorite status", "error");
    }
  };

  // Remove asset from watchlist
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
      // Refresh dashboard to show updated data
      await fetchDashboard();
    } catch (err) {
      console.error(`Failed to refresh ${ticker}:`, err);
      showToast(`Failed to refresh ${ticker}`, "error");
    } finally {
      setRefreshingTicker(null);
    }
  };

  if (configured === null) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#0A0A0F] text-white">
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-zinc-400">Loading...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-[#0A0A0F] text-[#FAFAFA] font-sans selection:bg-amber-500/30">
      {(!configured || isSetupOpen) && (
        <SetupModal onClose={configured ? () => setIsSetupOpen(false) : undefined} />
      )}

      {isLogsOpen && (
        <LogViewer onClose={() => setIsLogsOpen(false)} />
      )}

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

      {/* Model Playground Full-Page View */}
      {isPlaygroundOpen && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-zinc-950">
          <ModelPlayground onBack={() => setIsPlaygroundOpen(false)} />
        </div>
      )}

      <header className="border-b border-white/[0.08] bg-[#12121A]/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${configured ? "bg-amber-500 animate-pulse shadow-[0_0_10px_rgba(245,158,11,0.5)]" : "bg-red-500"}`} />
            <h1 className="text-lg font-bold tracking-tight text-white font-display">
              Stock AI Dashboard
            </h1>
          </div>
          <div className="flex items-center gap-4 text-sm text-zinc-500">
            {lastSync && (
              <span className="text-xs text-zinc-600">
                Last sync: {lastSync.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={() => setIsLogsOpen(true)}
              className="p-2 rounded-lg bg-[#1A1A24]/60 border border-white/[0.08] hover:border-white/[0.15] transition-all text-zinc-400 hover:text-white"
              title="System Logs"
            >
              <Terminal size={18} />
            </button>
            <button
              onClick={() => setIsAIPanelOpen(true)}
              className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/30 hover:bg-amber-500/20 hover:border-amber-500/50 transition-all text-amber-400 hover:text-amber-300 hover:shadow-[0_0_15px_rgba(245,158,11,0.2)]"
              title="AI Model Inspector"
            >
              <Brain size={18} />
            </button>
            <button
              onClick={() => setIsPlaygroundOpen(true)}
              className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/30 hover:bg-amber-500/20 hover:border-amber-500/50 transition-all text-amber-400 hover:text-amber-300 hover:shadow-[0_0_15px_rgba(245,158,11,0.2)]"
              title="Model Playground"
            >
              <FlaskConical size={18} />
            </button>
            <button
              onClick={() => setIsSetupOpen(true)}
              className="p-2 rounded-lg bg-[#1A1A24]/60 border border-white/[0.08] hover:border-white/[0.15] transition-all text-zinc-400 hover:text-white"
              title="Settings"
            >
              <Settings size={18} />
            </button>
            <span className="flex items-center gap-1.5 ml-2">
              <span className={`w-2 h-2 rounded-full ${configured ? "bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.5)]" : "bg-red-500"}`} />
              <span className={configured ? "text-amber-400" : "text-red-500"}>
                {configured ? "Active" : "Setup Required"}
              </span>
            </span>
            <span className="px-2 py-0.5 rounded bg-[#1A1A24]/60 border border-white/[0.08] font-mono text-xs">
              v1.0.0
            </span>
          </div>
        </div>
      </header>

      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-white flex items-center gap-3">
                Market Overview
                <span className="text-xs px-2 py-1 bg-amber-500/20 text-amber-400 rounded-full border border-amber-500/30 flex items-center gap-1">
                  <Star size={12} fill="currentColor" />
                  {assets.filter(a => a.isFavorite !== false).length} Favorites
                </span>
              </h2>
              <p className="text-zinc-400 text-sm mt-1">
                {isLoading
                  ? "Loading data..."
                  : assets.length === 0
                    ? "Add assets to your watchlist to begin"
                    : `Tracking ${assets.length} asset${assets.length !== 1 ? "s" : ""}`}
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

          {assets.length === 0 && !isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-zinc-500">
              <p className="text-lg mb-2">No assets in watchlist</p>
              <p className="text-sm mb-4">Add a ticker above or browse popular stocks</p>
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
        </div>
      </main >

      <DetailDrawer
        asset={selectedAsset}
        onClose={() => setSelectedAsset(null)}
      />
    </div >
  );
}
