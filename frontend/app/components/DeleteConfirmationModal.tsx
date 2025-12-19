"use client";

import { AlertTriangle, X } from "lucide-react";

interface DeleteConfirmationModalProps {
    ticker: string;
    isOpen: boolean;
    onClose: () => void;
    onConfirm: (cascade: boolean) => void;
}

export default function DeleteConfirmationModal({ ticker, isOpen, onClose, onConfirm }: DeleteConfirmationModalProps) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0A0A0F]/90 backdrop-blur-sm p-4">
            <div className="w-full max-w-md bg-[#12121A] border border-white/[0.08] rounded-xl shadow-2xl overflow-hidden animate-scale-in">
                <div className="p-6">
                    <div className="flex items-center gap-3 mb-4 text-rose-500">
                        <AlertTriangle size={24} />
                        <h3 className="text-xl font-bold text-white">Delete {ticker}?</h3>
                    </div>

                    <p className="text-zinc-400 mb-6">
                        You are about to remove <span className="text-white font-mono font-bold">{ticker}</span> from your watchlist.
                        Please choose how you want to proceed.
                    </p>

                    <div className="flex flex-col gap-3">
                        <button
                            onClick={() => onConfirm(false)}
                            className="w-full py-3 px-4 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-white font-medium transition-colors border border-zinc-700 flex items-center justify-center gap-2"
                        >
                            <span>Just Remove from Watchlist</span>
                            <span className="text-xs text-zinc-500 font-normal">(Keep Data)</span>
                        </button>

                        <button
                            onClick={() => onConfirm(true)}
                            className="w-full py-3 px-4 rounded-lg bg-rose-600 hover:bg-rose-700 text-white font-medium transition-colors flex items-center justify-center gap-2 shadow-lg shadow-rose-900/20"
                        >
                            <span>Delete All Data</span>
                            <span className="text-xs text-rose-200 font-normal">(Cascade)</span>
                        </button>
                    </div>
                </div>

                <div className="bg-zinc-950/50 px-6 py-4 border-t border-zinc-800 flex justify-center">
                    <button
                        onClick={onClose}
                        className="text-sm text-zinc-500 hover:text-white transition-colors"
                    >
                        Cancel
                    </button>
                </div>
            </div>
        </div>
    );
}
