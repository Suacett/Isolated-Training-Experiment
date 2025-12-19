import { useState } from "react";
import { Plus } from "lucide-react";
import { useMounted } from "../hooks/useMounted";

interface AddAssetBarProps {
    onAdd: (ticker: string) => void;
}

export default function AddAssetBar({ onAdd }: AddAssetBarProps) {
    const [ticker, setTicker] = useState("");
    const mounted = useMounted();

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
                        className="w-full bg-[#1A1A24]/60 backdrop-blur-md border border-white/[0.08] text-[#FAFAFA] px-4 py-2 rounded-lg focus:outline-none focus:border-amber-500/50 focus:ring-2 focus:ring-amber-500/20 focus:shadow-[0_0_20px_rgba(245,158,11,0.1)] placeholder-zinc-500 transition-all"
                    />
                </div>
                <button
                    type="submit"
                    disabled={!ticker.trim()}
                    className="bg-amber-500 hover:bg-amber-400 text-[#0A0A0F] px-4 py-2 rounded-lg flex items-center gap-2 transition-all font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[0_0_20px_rgba(245,158,11,0.4)]"
                >
                    {mounted && <Plus size={18} />}
                    Add
                </button>
            </form>
        </div>
    );
}
