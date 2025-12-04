"use client";

import { useState } from "react";

type DataSource = "alpaca" | "alpha_vantage";

export default function SetupModal() {
    const [dataSource, setDataSource] = useState<DataSource>("alpaca");
    const [alpacaApiKey, setAlpacaApiKey] = useState("");
    const [alpacaSecretKey, setAlpacaSecretKey] = useState("");
    const [alphaVantageKey, setAlphaVantageKey] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

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
                window.location.reload();
            } else {
                alert("Failed to save keys");
            }
        } catch (error) {
            console.error(error);
            alert("Error saving keys");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-80 z-50">
            <div className="bg-gray-900 border border-gray-700 p-8 rounded-xl shadow-2xl max-w-md w-full">
                <h2 className="text-2xl font-bold text-white mb-4 text-center">
                    Welcome
                </h2>
                <p className="text-gray-400 mb-6 text-center">
                    Configure your data source to begin.
                </p>

                {/* Data Source Selector */}
                <div className="flex gap-2 mb-6">
                    <button
                        type="button"
                        onClick={() => setDataSource("alpaca")}
                        className={`flex-1 py-2 px-4 rounded-lg border transition-colors ${
                            dataSource === "alpaca"
                                ? "bg-emerald-600 border-emerald-500 text-white"
                                : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600"
                        }`}
                    >
                        Alpaca Markets
                    </button>
                    <button
                        type="button"
                        onClick={() => setDataSource("alpha_vantage")}
                        className={`flex-1 py-2 px-4 rounded-lg border transition-colors ${
                            dataSource === "alpha_vantage"
                                ? "bg-emerald-600 border-emerald-500 text-white"
                                : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600"
                        }`}
                    >
                        Alpha Vantage
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                    {dataSource === "alpaca" ? (
                        <>
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-1">
                                    Alpaca API Key
                                </label>
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
                                <label className="block text-sm font-medium text-gray-400 mb-1">
                                    Alpha Vantage API Key
                                </label>
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

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-4 rounded-lg transition-colors disabled:opacity-50"
                    >
                        {loading ? "Saving..." : "Start Trading"}
                    </button>
                </form>
            </div>
        </div>
    );
}
