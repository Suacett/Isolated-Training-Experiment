import { useState, useEffect } from "react";
import { X, CheckCircle2, XCircle, Cpu, RefreshCw, FlaskConical, AlertTriangle } from "lucide-react";
import { useToast } from "./Toast";

interface SetupModalProps {
    onClose?: () => void;
}

interface AIStatus {
    model_loaded: boolean;
    model_info: {
        type: string;
        input_features: number;
        hidden_dim: number;
        horizons: string[];
    } | null;
    scaler_loaded: boolean;
    device: string;
    ready_for_predictions: boolean;
}

// Playground Toggle Component
function PlaygroundToggle() {
    const [enabled, setEnabled] = useState(false);
    const [loading, setLoading] = useState(true);
    const [toggling, setToggling] = useState(false);
    const { showToast } = useToast();

    useEffect(() => {
        fetch("http://localhost:8000/settings/playground")
            .then(res => res.json())
            .then(data => {
                setEnabled(data.enabled);
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, []);

    const handleToggle = async () => {
        setToggling(true);
        try {
            const res = await fetch(`http://localhost:8000/settings/playground?enabled=${!enabled}`, {
                method: "POST"
            });
            const data = await res.json();
            setEnabled(data.enabled);
            showToast(data.enabled ? "Model Playground Enabled" : "Model Playground Disabled", "success");
        } catch (e) {
            console.error("Failed to toggle playground:", e);
            showToast("Failed to toggle playground", "error");
        } finally {
            setToggling(false);
        }
    };

    if (loading) return null;

    return (
        <div className="bg-amber-900/20 border border-amber-700/50 rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <FlaskConical size={20} className="text-amber-400" />
                    <span className="text-white font-medium">Model Playground</span>
                </div>
                <button
                    onClick={handleToggle}
                    disabled={toggling}
                    className={`relative w-14 h-7 rounded-full transition-colors flex-shrink-0 ${enabled ? "bg-amber-500" : "bg-zinc-700"
                        }`}
                >
                    <span
                        className={`absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full transition-all shadow-sm ${enabled ? "translate-x-7" : "translate-x-0"
                            }`}
                    />
                </button>
            </div>
            <p className="text-xs text-amber-400/70 flex items-start gap-1">
                <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
                Loads all AI models for comparison. May use 4-8GB VRAM/RAM. Setting persists across restarts.
            </p>
            {enabled && (
                <p className="text-xs text-emerald-400 mt-2">
                    ✓ Models will load when you open the Playground
                </p>
            )}
        </div>
    );
}

export default function SetupModal({ onClose }: SetupModalProps) {
    const [alphaVantageKey, setAlphaVantageKey] = useState("");
    const [loading, setLoading] = useState(false);
    const [resetting, setResetting] = useState(false);
    const [keyStatus, setKeyStatus] = useState({ alpha_vantage: false });
    const [aiStatus, setAIStatus] = useState<AIStatus | null>(null);
    const { showToast } = useToast();

    useEffect(() => {
        // Fetch system status
        fetch("http://localhost:8000/status")
            .then(res => res.json())
            .then(data => {
                setKeyStatus({
                    alpha_vantage: data.alpha_vantage_configured
                });
            })
            .catch(console.error);

        // Fetch AI status
        fetch("http://localhost:8000/status/ai")
            .then(res => res.json())
            .then(data => setAIStatus(data))
            .catch(console.error);
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!alphaVantageKey.trim()) return;

        setLoading(true);

        try {
            const res = await fetch("http://localhost:8000/settings/keys", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ALPHA_VANTAGE_KEY: alphaVantageKey }),
            });

            if (res.ok) {
                showToast("Settings saved successfully!", "success");
                setKeyStatus(prev => ({ ...prev, alpha_vantage: true }));
                setTimeout(() => onClose?.(), 1500);
            } else {
                showToast("Failed to save settings.", "error");
            }
        } catch (error) {
            console.error(error);
            showToast("Network error occurred.", "error");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 flex items-center justify-center bg-[#0A0A0F]/90 backdrop-blur-sm z-50">
            <div className="bg-[#12121A] border border-white/[0.08] p-8 rounded-xl shadow-2xl max-w-lg w-full relative animate-scale-in">
                {onClose && (
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
                    >
                        <X size={24} />
                    </button>
                )}
                <h2 className="text-2xl font-bold text-white mb-4 text-center">
                    {onClose ? "Settings" : "Welcome"}
                </h2>

                {/* AI Model Status Card */}
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 mb-6">
                    <div className="flex items-center gap-2 mb-3">
                        <Cpu size={20} className="text-blue-400" />
                        <span className="text-white font-medium">AI Model Status</span>
                    </div>

                    {aiStatus ? (
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Model</span>
                                <span className={aiStatus.model_loaded ? "text-emerald-400" : "text-red-400"}>
                                    {aiStatus.model_loaded ? `${aiStatus.model_info?.type} (${aiStatus.model_info?.input_features} features)` : "Not Loaded"}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Scaler</span>
                                <span className={aiStatus.scaler_loaded ? "text-emerald-400" : "text-red-400"}>
                                    {aiStatus.scaler_loaded ? "Loaded" : "Not Loaded"}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Device</span>
                                <span className="text-blue-400">{aiStatus.device}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Ready</span>
                                <span className={aiStatus.ready_for_predictions ? "text-emerald-400" : "text-yellow-400"}>
                                    {aiStatus.ready_for_predictions ? "✓ Ready for Predictions" : "⚠ Missing Components"}
                                </span>
                            </div>
                        </div>
                    ) : (
                        <div className="text-gray-500 text-sm">Loading...</div>
                    )}
                </div>

                {/* Data Source Status */}
                <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg px-4 py-3 mb-4 flex items-center justify-between">
                    <span className="text-blue-300">📊 Yahoo Finance (Data Source)</span>
                    <span className="text-xs text-emerald-400 flex items-center gap-1">
                        <CheckCircle2 size={12} /> Active (No API Key Needed)
                    </span>
                </div>

                {/* Data Management */}
                <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4 mb-6">
                    <div className="flex items-center gap-2 mb-3">
                        <RefreshCw size={20} className="text-purple-400" />
                        <span className="text-white font-medium">Data Management</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <button
                            onClick={async () => {
                                // Non-blocking confirmation using toast
                                showToast("Starting Daily Sync...", "info");
                                setLoading(true);
                                try {
                                    const watchlistRes = await fetch("http://localhost:8000/stocks/");
                                    const watchlist = await watchlistRes.json();

                                    let successCount = 0;
                                    let errorCount = 0;
                                    for (const ticker of watchlist) {
                                        try {
                                            const res = await fetch(`http://localhost:8000/ingest/${ticker}?mode=daily`, { method: "POST" });
                                            if (res.ok) {
                                                successCount++;
                                            } else {
                                                const errData = await res.json();
                                                console.error(`Failed to sync ${ticker}:`, errData);
                                                errorCount++;
                                            }
                                        } catch (e) {
                                            console.error(`Failed to sync ${ticker}`, e);
                                            errorCount++;
                                        }
                                    }
                                    if (successCount > 0) {
                                        showToast(`Synced ${successCount}/${watchlist.length} assets (Daily).${errorCount > 0 ? ` ${errorCount} failed.` : ""}`, errorCount === 0 ? "success" : "warning");
                                    } else {
                                        showToast(`Failed to sync all assets. Check console.`, "error");
                                    }
                                } catch (e) {
                                    showToast("Failed to sync data.", "error");
                                } finally {
                                    setLoading(false);
                                }
                            }}
                            disabled={loading}
                            className="bg-zinc-700 hover:bg-zinc-600 text-white text-sm font-medium py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-2"
                        >
                            {loading ? <RefreshCw className="animate-spin" size={14} /> : <RefreshCw size={14} />}
                            Sync Daily
                        </button>

                        <button
                            onClick={async () => {
                                showToast("Starting Full Sync (1980-Now)... This may take a moment.", "info");
                                setLoading(true);
                                try {
                                    // Use the new backend bulk endpoint
                                    const res = await fetch("http://localhost:8000/ingest/all?mode=full", { method: "POST" });

                                    if (res.ok) {
                                        const data = await res.json();
                                        const successCount = data.predictions ? data.predictions.length : 0;
                                        const backfillCount = data.backfills ? data.backfills.length : 0;
                                        showToast(`Sync Complete! Processed ${successCount} assets with ${backfillCount} historical backfills.`, "success");
                                    } else {
                                        const errData = await res.json();
                                        console.error("Bulk sync failed:", errData);
                                        showToast(`Failed to sync assets: ${errData.detail || "Unknown error"}`, "error");
                                    }
                                } catch (e) {
                                    console.error("Bulk sync network error:", e);
                                    showToast("Failed to connect to sync service.", "error");
                                } finally {
                                    setLoading(false);
                                }
                            }}
                            disabled={loading}
                            className="bg-zinc-700 hover:bg-zinc-600 text-white text-sm font-medium py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-2"
                        >
                            {loading ? <RefreshCw className="animate-spin" size={14} /> : <RefreshCw size={14} />}
                            Sync Full
                        </button>
                    </div>
                    <p className="text-xs text-zinc-500 mt-2 flex items-start gap-1">
                        <AlertTriangle size={12} className="mt-0.5 flex-shrink-0 text-amber-500" />
                        <span className="text-amber-500/80">Warning:</span> Full sync fetches deep history (1980+) and generates thousands of predictions. This process may take several minutes.
                    </p>
                </div>

                {/* Model Playground Toggle */}
                <PlaygroundToggle />

                {/* Alpha Vantage for Intrinsic Value */}
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <div className="flex items-center justify-between mb-1">
                            <label className="block text-sm font-medium text-gray-400">
                                Alpha Vantage API Key (for Intrinsic Value)
                            </label>
                            {keyStatus.alpha_vantage ? (
                                <span className="text-xs text-emerald-500 flex items-center gap-1">
                                    <CheckCircle2 size={12} /> Active
                                </span>
                            ) : (
                                <span className="text-xs text-yellow-500 flex items-center gap-1">
                                    Optional
                                </span>
                            )}
                        </div>
                        <input
                            type="text"
                            value={alphaVantageKey}
                            onChange={(e) => setAlphaVantageKey(e.target.value)}
                            placeholder="Your Alpha Vantage API key"
                            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        />
                    </div>
                    <p className="text-xs text-gray-500">
                        Get a free API key from{" "}
                        <a
                            href="https://www.alphavantage.co/support/#api-key"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-emerald-500 hover:underline"
                        >
                            Alpha Vantage
                        </a>
                        . Required for EPS data and intrinsic value calculations.
                    </p>

                    <div className="flex flex-col gap-3">
                        <button
                            type="submit"
                            disabled={loading || resetting}
                            className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {loading && <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
                            {loading ? "Saving..." : "Save Settings"}
                        </button>

                        <button
                            type="button"
                            onClick={async () => {
                                if (!confirm("Are you sure you want to reset all settings?")) return;
                                setResetting(true);

                                try {
                                    const res = await fetch("http://localhost:8000/settings/keys", {
                                        method: "DELETE",
                                    });
                                    if (res.ok) {
                                        setAlphaVantageKey("");
                                        setKeyStatus({ alpha_vantage: false });
                                        showToast("Settings Reset", "success");
                                        setTimeout(() => {
                                            if (onClose) onClose();
                                            else window.location.reload();
                                        }, 1500);
                                    } else {
                                        showToast("Failed to reset settings.", "error");
                                    }
                                } catch (error) {
                                    console.error(error);
                                    showToast("Network error occurred.", "error");
                                } finally {
                                    setResetting(false);
                                }
                            }}
                            disabled={loading || resetting}
                            className="w-full bg-red-900/30 hover:bg-red-900/50 border border-red-900/50 text-red-400 font-bold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                            {resetting && <div className="w-4 h-4 border-2 border-red-400 border-t-transparent rounded-full animate-spin" />}
                            {resetting ? "Resetting..." : "Reset Settings"}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
