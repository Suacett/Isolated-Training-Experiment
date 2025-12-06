import { useState } from "react";
import { Plus } from "lucide-react";

interface AddAssetBarProps {
    onAdd: (ticker: string) => void;
}

export default function AddAssetBar({ onAdd }: AddAssetBarProps) {
    const [ticker, setTicker] = useState("");

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (ticker.trim()) {
            onAdd(ticker.trim().toUpperCase());
            setTicker("");
        }
    };

    return (
        <div className="flex gap-2 w-full max-w-md">
            <form onSubmit={handleSubmit} className="flex gap-2 flex-1">
                <div className="relative flex-1">
                    <input
                        type="text"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value)}
                        placeholder="Add Ticker (e.g. AAPL, BTC/USD)"
                        className="w-full bg-zinc-900 border border-zinc-700 text-zinc-100 px-4 py-2 rounded-lg focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 placeholder-zinc-500"
                    />
                </div>
                <button
                    type="submit"
                    disabled={!ticker.trim()}
                    className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <Plus size={18} />
                    Add
                </button>
            </form>
        </div>
    );
}
