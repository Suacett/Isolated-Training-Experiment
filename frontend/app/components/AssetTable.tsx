"use client";

import { Bitcoin, CheckCircle2, XCircle } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export interface Asset {
    ticker: string;
    price: number;
    prediction: number;
    intrinsic: number;
    accuracy: boolean; // true = green dot, false = red dot
    isCrypto: boolean;
    source?: string;
}

interface AssetTableProps {
    assets: Asset[];
    onSelect: (asset: Asset) => void;
}

export default function AssetTable({ assets, onSelect }: AssetTableProps) {
    return (
        <div className="w-full overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/50">
            <table className="w-full text-left text-sm text-zinc-400">
                <thead className="bg-zinc-900 text-xs uppercase text-zinc-500 border-b border-zinc-800">
                    <tr>
                        <th className="px-6 py-3 font-medium">Ticker</th>
                        <th className="px-6 py-3 font-medium text-right">Price</th>
                        <th className="px-6 py-3 font-medium text-right">AI Prediction</th>
                        <th className="px-6 py-3 font-medium text-right">Intrinsic Val</th>
                        <th className="px-6 py-3 font-medium text-center">Accuracy</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800">
                    {assets.map((asset) => (
                        <tr
                            key={asset.ticker}
                            onClick={() => onSelect(asset)}
                            className="cursor-pointer hover:bg-zinc-800/50 transition-colors"
                        >
                            <td className="px-6 py-4 font-medium text-zinc-200 flex items-center gap-2">
                                {asset.isCrypto && <Bitcoin size={16} className="text-orange-500" />}
                                {asset.ticker}
                            </td>
                            <td className="px-6 py-4 text-right font-mono text-zinc-300">
                                <div className="flex flex-col items-end">
                                    <span>${asset.price.toFixed(2)}</span>
                                    {asset.source && (
                                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500">
                                            {asset.source}
                                        </span>
                                    )}
                                </div>
                            </td>
                            <td className="px-6 py-4 text-right font-mono">
                                <span
                                    className={cn(
                                        asset.prediction > asset.price
                                            ? "text-emerald-400"
                                            : "text-rose-400"
                                    )}
                                >
                                    ${asset.prediction.toFixed(2)}
                                </span>
                            </td>
                            <td className="px-6 py-4 text-right font-mono text-zinc-300">
                                ${asset.intrinsic.toFixed(2)}
                            </td>
                            <td className="px-6 py-4 flex justify-center">
                                <div
                                    className={cn(
                                        "flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border",
                                        asset.accuracy
                                            ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
                                            : "bg-rose-500/10 text-rose-500 border-rose-500/20"
                                    )}
                                >
                                    <div
                                        className={cn(
                                            "w-1.5 h-1.5 rounded-full",
                                            asset.accuracy ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
                                        )}
                                    />
                                    {asset.accuracy ? "Hit" : "Miss"}
                                </div>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
