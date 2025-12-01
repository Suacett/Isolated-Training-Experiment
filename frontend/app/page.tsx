"use client";

import { useState, useEffect } from "react";
import SetupModal from "./components/SetupModal";
import AddAssetBar from "./components/AddAssetBar";
import AssetTable, { Asset } from "./components/AssetTable";
import DetailDrawer from "./components/DetailDrawer";

// Mock data generator
const generateMockAssets = (): Asset[] => [
  {
    ticker: "AAPL",
    price: 185.92,
    prediction: 188.45,
    intrinsic: 175.20,
    accuracy: true,
    isCrypto: false,
  },
  {
    ticker: "BTC/USD",
    price: 43500.00,
    prediction: 42100.00,
    intrinsic: 45000.00,
    accuracy: false,
    isCrypto: true,
  },
  {
    ticker: "NVDA",
    price: 485.00,
    prediction: 495.20,
    intrinsic: 460.00,
    accuracy: true,
    isCrypto: false,
  },
  {
    ticker: "ETH/USD",
    price: 2250.00,
    prediction: 2300.00,
    intrinsic: 2400.00,
    accuracy: true,
    isCrypto: true,
  },
  {
    ticker: "TSLA",
    price: 240.00,
    prediction: 235.00,
    intrinsic: 210.00,
    accuracy: false,
    isCrypto: false,
  },
];

export default function Home() {
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/status")
      .then((res) => res.json())
      .then((data) => setConfigured(data.configured))
      .catch((err) => console.error("Failed to fetch status", err));

    // Load mock data
    setAssets(generateMockAssets());
  }, []);

  const handleAddAsset = (ticker: string) => {
    // Mock adding asset
    const newAsset: Asset = {
      ticker,
      price: Math.random() * 1000,
      prediction: Math.random() * 1000,
      intrinsic: Math.random() * 1000,
      accuracy: Math.random() > 0.5,
      isCrypto: ticker.includes("/") || ["BTC", "ETH", "LTC"].includes(ticker),
    };
    setAssets((prev) => [newAsset, ...prev]);
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
                Real-time AI analysis and intrinsic valuation
              </p>
            </div>
            <AddAssetBar onAdd={handleAddAsset} />
          </div>

          <AssetTable assets={assets} onSelect={setSelectedAsset} />
        </div>
      </main>

      <DetailDrawer
        asset={selectedAsset}
        onClose={() => setSelectedAsset(null)}
      />
    </div>
  );
}
