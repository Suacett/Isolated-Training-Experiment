import { Info, AlertTriangle } from "lucide-react";
import { IntrinsicBreakdown, getConfidenceExplanation } from "./DetailDrawer.types";

export function AIModelInfo({ confidence, accuracy }: { confidence: number; accuracy: number | null }) {
    return (
        <div className="relative inline-block group ml-2">
            <div className="flex items-center gap-1 px-2 py-1 bg-pink-900/30 border border-pink-700/50 rounded text-[10px] text-pink-400 cursor-help">
                <Info size={10} />
                <span>AI Model</span>
            </div>
            <div className="absolute left-0 top-full mt-1 w-64 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[200] pointer-events-none">
                <div className="text-[10px] text-zinc-300 space-y-1">
                    <div><strong>Model:</strong> Adaptive AI (LSTM/Transformer)</div>
                    <div><strong>Input:</strong> Multi-factor Market Data</div>
                    <div><strong>Horizons:</strong> 1d, 1w, 1m, 6m</div>
                    <div><strong>Confidence:</strong> {(confidence * 100).toFixed(0)}% - {getConfidenceExplanation(confidence)}</div>
                    {accuracy !== null && <div><strong>Accuracy:</strong> {accuracy}% directional</div>}
                </div>
            </div>
        </div>
    );
}

export function IntrinsicInfo({ breakdown }: { breakdown: IntrinsicBreakdown | null }) {
    if (!breakdown) return null;

    return (
        <div className="relative inline-block group ml-2">
            <div className="flex items-center gap-1 px-2 py-1 bg-purple-900/30 border border-purple-700/50 rounded text-[10px] text-purple-400 cursor-help">
                <Info size={10} />
                <span>Intrinsic Value</span>
            </div>
            <div className="absolute left-0 top-full mt-1 w-56 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[200] pointer-events-none">
                <div className="text-[10px] text-zinc-300 space-y-1">
                    <div className="font-medium text-purple-400 mb-1">Graham Formula</div>
                    <div className="grid grid-cols-2 gap-1">
                        <span>EPS:</span><span className="font-mono">${breakdown.eps?.toFixed(2) || 'N/A'}</span>
                        <span>Growth:</span><span className="font-mono">{((breakdown.growth_rate || 0) * 100).toFixed(1)}%</span>
                        <span>Bond Yield:</span><span className="font-mono">{breakdown.bond_yield?.toFixed(2) || 'N/A'}%</span>
                        <span>Value:</span><span className="font-mono text-purple-400">${breakdown.intrinsic_value?.toFixed(2) || 'N/A'}</span>
                    </div>
                    {breakdown.is_estimated && (
                        <div className="text-amber-400 mt-1 flex items-center gap-1">
                            <AlertTriangle size={8} />
                            Estimated values
                        </div>
                    )}
                    <div className="text-zinc-500 mt-1 border-t border-zinc-700 pt-1">
                        V = EPS × (8.5 + 2g) × (4.4/Y)
                    </div>
                </div>
            </div>
        </div>
    );
}
