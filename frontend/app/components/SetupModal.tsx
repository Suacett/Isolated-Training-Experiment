"use client";

import { useState } from "react";

export default function SetupModal() {
    const [apiKey, setApiKey] = useState("");
    const [secretKey, setSecretKey] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

        try {
            const res = await fetch("http://localhost:8000/settings/keys", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    ALPACA_API_KEY: apiKey,
                    ALPACA_SECRET_KEY: secretKey,
                }),
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
                    Enter your Alpaca API Keys to begin.
                </p>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">
                            API Key
                        </label>
                        <input
                            type="text"
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">
                            Secret Key
                        </label>
                        <input
                            type="password"
                            value={secretKey}
                            onChange={(e) => setSecretKey(e.target.value)}
                            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                            required
                        />
                    </div>
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
