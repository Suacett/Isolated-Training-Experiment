"use client";

import { useState, useEffect, useCallback } from "react";
import SetupModal from "./components/SetupModal";
import AddAssetBar from "./components/AddAssetBar";
import AssetTable, { Asset } from "./components/AssetTable";
import DetailDrawer from "./components/DetailDrawer";
import { RefreshCw } from "lucide-react";

const API_BASE = "http://localhost:8000";

// API response types
interface DashboardItem {
  ticker: string;
  current_price: number;
  prediction: number;
  signal: string;
  intrinsic_value?: number;
  accuracy?: boolean;
}

export default function Home() {
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);

  // Helper to detect crypto tickers
  const isCryptoTicker = (ticker: string): boolean => {
    return ticker.includes("/") || ["BTC", "ETH", "LTC", "SOL", "DOGE"].some(c => ticker.includes(c));
  };

  // Fetch dashboard data from API
  const fetchDashboard = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/dashboard`);
      if (!res.ok) throw new Error("Failed to fetch dashboard");
      const data: DashboardItem[] = await res.json();

      const mappedAssets: Asset[] = data.map((item) => ({
        ticker: item.ticker,
        price: item.current_price || 0,
        prediction: item.prediction || 0,
        intrinsic: item.intrinsic_value || 0,
        accuracy: item.accuracy ?? (item.prediction > item.current_price),
        isCrypto: isCryptoTicker(item.ticker),
      }));

      setAssets(mappedAssets);
    } catch (err) {
      console.error("Failed to fetch dashboard:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

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

  // Initial load
  useEffect(() => {
    fetch(`${API_BASE}/status`)
      .then((res) => res.json())
      .then((data) => {
        setConfigured(data.configured);
        if (data.configured) {
          fetchDashboard();
        }
      })
      .catch((err) => console.error("Failed to fetch status", err));
  }, [fetchDashboard]);

  // Add asset to watchlist and ingest data
  const handleAddAsset = async (ticker: string) => {
    try {
      // 1. Add to watchlist
      await fetch(`${API_BASE}/watchlist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });

      // 2. Ingest data for this ticker
      await fetch(`${API_BASE}/ingest/${ticker}`, { method: "POST" });

      // 3. Refresh dashboard
      await fetchDashboard();
    } catch (err) {
      console.error("Failed to add asset:", err);
    }
  };

  // Remove asset from watchlist
  const handleRemoveAsset = async (ticker: string) => {
    try {
      await fetch(`${API_BASE}/watchlist/${ticker}`, { method: "DELETE" });
      await fetchDashboard();
    } catch (err) {
      console.error("Failed to remove asset:", err);
    }
  };

  if (configured === null) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-black text-white">
        Loading...
      </div>
    );
  }

  if (!configured) {
    return <SetupModal />;
  }

  return (
    <div className="flex min-h-screen flex-col bg-black text-zinc-100 font-sans selection:bg-emerald-500/30">
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
            <h1 className="text-lg font-bold tracking-tight text-white">
              PORTFOLIO COMMAND CENTER
            </h1>
          </div>
          <div className="flex items-center gap-4 text-sm text-zinc-500">
            {lastSync && (
              <span className="text-xs text-zinc-600">
                Last sync: {lastSync.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={handleSyncData}
              disabled={isSyncing}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 hover:bg-zinc-700 hover:border-zinc-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw size={14} className={isSyncing ? "animate-spin" : ""} />
              <span>{isSyncing ? "Syncing..." : "Sync Data"}</span>
            </button>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              System Online
            </span>
            <span className="px-2 py-0.5 rounded bg-zinc-800 border border-zinc-700 font-mono text-xs">
              v1.0.0
            </span>
          </div>
        </div>
      </header>

      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-white">Market Overview</h2>
              <p className="text-zinc-400 text-sm mt-1">
                {isLoading
                  ? "Loading data..."
                  : assets.length === 0
                  ? "Add assets to your watchlist to begin"
                  : `Tracking ${assets.length} asset${assets.length !== 1 ? "s" : ""}`}
              </p>
            </div>
            <AddAssetBar onAdd={handleAddAsset} />
          </div>

          {assets.length === 0 && !isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-zinc-500">
              <p className="text-lg mb-2">No assets in watchlist</p>
              <p className="text-sm">Add a ticker above to start tracking</p>
            </div>
          ) : (
            <AssetTable assets={assets} onSelect={setSelectedAsset} />
          )}
        </div>
      </main>

      <DetailDrawer
        asset={selectedAsset}
        onClose={() => setSelectedAsset(null)}
      />
    </div>
  );
}
