import { useState, useEffect } from "react";
import { X, CheckCircle2, XCircle } from "lucide-react";

type DataSource = "alpaca" | "alpha_vantage";

interface SetupModalProps {
    onClose?: () => void;
}

export default function SetupModal({ onClose }: SetupModalProps) {
    const [dataSource, setDataSource] = useState<DataSource>("alpaca");
    const [alpacaApiKey, setAlpacaApiKey] = useState("");
    const [alpacaSecretKey, setAlpacaSecretKey] = useState("");
    const [alphaVantageKey, setAlphaVantageKey] = useState("");
    const [loading, setLoading] = useState(false);
    const [resetting, setResetting] = useState(false);
    const [status, setStatus] = useState<"idle" | "success" | "error">("idle");
    const [message, setMessage] = useState("");
    const [keyStatus, setKeyStatus] = useState({ alpaca: false, alpha_vantage: false });

    useEffect(() => {
        fetch("http://localhost:8000/status")
            .then(res => res.json())
            .then(data => {
                setKeyStatus({
                    alpaca: data.alpaca_configured,
                    alpha_vantage: data.alpha_vantage_configured
                });
            })
            .catch(console.error);
    }, []);

    const isFormValid = () => {
        if (dataSource === "alpaca") {
            return alpacaApiKey.trim() !== "" && alpacaSecretKey.trim() !== "";
        } else {
            return alphaVantageKey.trim() !== "";
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!isFormValid()) return;

        setLoading(true);
        setStatus("idle");
        setMessage("");

        try {
            const payload: Record<string, string | null> = {
                ALPACA_API_KEY: null,
                ALPACA_SECRET_KEY: null,
                ALPHA_VANTAGE_KEY: null,
            };

            if (dataSource === "alpaca") {
                payload.ALPACA_API_KEY = alpacaApiKey;
                payload.ALPACA_SECRET_KEY = alpacaSecretKey;
            } else {
                payload.ALPHA_VANTAGE_KEY = alphaVantageKey;
            }

            const res = await fetch("http://localhost:8000/settings/keys", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            if (res.ok) {
                setStatus("success");
                setMessage("Configuration saved successfully!");
                // Update status locally
                if (dataSource === "alpaca") setKeyStatus(prev => ({ ...prev, alpaca: true }));
                else setKeyStatus(prev => ({ ...prev, alpha_vantage: true }));

                setTimeout(() => {
                    if (onClose) {
                        onClose();
                    }
                }, 1500);
            } else {
                setStatus("error");
                setMessage("Failed to save keys. Please check your input.");
            }
        } catch (error) {
            console.error(error);
            setStatus("error");
            setMessage("Network error occurred.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-80 z-50">
            <div className="bg-gray-900 border border-gray-700 p-8 rounded-xl shadow-2xl max-w-md w-full relative">
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
                <p className="text-gray-400 mb-6 text-center">
                    Configure your data source to begin.
                </p>

                {/* Data Source Selector */}
                <div className="flex gap-2 mb-6">
                    <button
                        type="button"
                        onClick={() => setDataSource("alpaca")}
                        className={`flex-1 py-2 px-4 rounded-lg border transition-colors flex items-center justify-center gap-2 ${dataSource === "alpaca"
                            ? "bg-emerald-600 border-emerald-500 text-white"
                            : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600"
                            }`}
                    >
                        Alpaca Markets
                        {keyStatus.alpaca ? <CheckCircle2 size={16} className="text-white" /> : null}
                    </button>
                    <button
                        type="button"
                        onClick={() => setDataSource("alpha_vantage")}
                        className={`flex-1 py-2 px-4 rounded-lg border transition-colors flex items-center justify-center gap-2 ${dataSource === "alpha_vantage"
                            ? "bg-emerald-600 border-emerald-500 text-white"
                            : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600"
                            }`}
                    >
                        Alpha Vantage
                        {keyStatus.alpha_vantage ? <CheckCircle2 size={16} className="text-white" /> : null}
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                    {dataSource === "alpaca" ? (
                        <>
                            <div>
                                <div className="flex items-center justify-between mb-1">
                                    <label className="block text-sm font-medium text-gray-400">
                                        Alpaca Key ID
                                    </label>
                                    {keyStatus.alpaca ? (
                                        <span className="text-xs text-emerald-500 flex items-center gap-1">
                                            <CheckCircle2 size={12} /> Active
                                        </span>
                                    ) : (
                                        <span className="text-xs text-red-500 flex items-center gap-1">
                                            <XCircle size={12} /> Not Configured
                                        </span>
                                    )}
                                </div>
                                <input
                                    type="text"
                                    value={alpacaApiKey}
                                    onChange={(e) => setAlpacaApiKey(e.target.value)}
                                    placeholder="PK..."
                                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-1">
                                    Alpaca Secret Key
                                </label>
                                <input
                                    type="password"
                                    value={alpacaSecretKey}
                                    onChange={(e) => setAlpacaSecretKey(e.target.value)}
                                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                                    required
                                />
                            </div>
                            <p className="text-xs text-gray-500">
                                Get your API keys from{" "}
                                <a
                                    href="https://app.alpaca.markets/paper/dashboard/overview"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-emerald-500 hover:underline"
                                >
                                    Alpaca Dashboard
                                </a>
                                . Supports stocks and crypto.
                            </p>
                        </>
                    ) : (
                        <>
                            <div>
                                <div className="flex items-center justify-between mb-1">
                                    <label className="block text-sm font-medium text-gray-400">
                                        Alpha Vantage API Key
                                    </label>
                                    {keyStatus.alpha_vantage ? (
                                        <span className="text-xs text-emerald-500 flex items-center gap-1">
                                            <CheckCircle2 size={12} /> Active
                                        </span>
                                    ) : (
                                        <span className="text-xs text-red-500 flex items-center gap-1">
                                            <XCircle size={12} /> Not Configured
                                        </span>
                                    )}
                                </div>
                                <input
                                    type="text"
                                    value={alphaVantageKey}
                                    onChange={(e) => setAlphaVantageKey(e.target.value)}
                                    placeholder="Your API key"
                                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                                    required
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
                                . Free tier: 5 calls/min, 500 calls/day.
                            </p>
                        </>
                    )}

                    {status === "error" && (
                        <div className="text-red-500 text-sm text-center bg-red-500/10 p-2 rounded">
                            {message}
                        </div>
                    )}

                    {status === "success" && (
                        <div className="text-emerald-500 text-sm text-center bg-emerald-500/10 p-2 rounded">
                            {message}
                        </div>
                    )}

                    <div className="flex flex-col gap-3">
                        <button
                            type="submit"
                            disabled={loading || resetting || !isFormValid()}
                            className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {loading && <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
                            {loading ? "Saving..." : "Save Settings"}
                        </button>

                        <button
                            type="button"
                            onClick={async () => {
                                if (!confirm("Are you sure you want to reset all API keys? This cannot be undone.")) return;
                                setResetting(true);
                                setStatus("idle");
                                setMessage("");

                                try {
                                    const res = await fetch("http://localhost:8000/settings/keys", {
                                        method: "DELETE",
                                    });
                                    if (res.ok) {
                                        setAlpacaApiKey("");
                                        setAlpacaSecretKey("");
                                        setAlphaVantageKey("");
                                        setStatus("success");
                                        setMessage("Keys Reset");
                                        setTimeout(() => {
                                            if (onClose) onClose();
                                            else window.location.reload();
                                        }, 1500);
                                    } else {
                                        setStatus("error");
                                        setMessage("Failed to reset keys.");
                                    }
                                } catch (error) {
                                    console.error(error);
                                    setStatus("error");
                                    setMessage("Network error occurred.");
                                } finally {
                                    setResetting(false);
                                }
                            }}
                            disabled={loading || resetting}
                            className="w-full bg-red-900/30 hover:bg-red-900/50 border border-red-900/50 text-red-400 font-bold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                            {resetting && <div className="w-4 h-4 border-2 border-red-400 border-t-transparent rounded-full animate-spin" />}
                            {resetting ? "Resetting..." : "Reset Keys"}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
