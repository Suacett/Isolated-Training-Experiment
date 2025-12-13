"use client";

import React, { useState } from "react";
import { Info } from "lucide-react";

interface InfoTooltipProps {
    label: string;
    traderView: string;
    academicView?: string;
    className?: string;
}

/**
 * InfoTooltip - Lightweight (i) icon tooltip with dual Trader/Academic views
 * 
 * Usage:
 *   <InfoTooltip 
 *     label="Sharpe Ratio" 
 *     traderView="Measures risk-adjusted return. > 1.0 is good."
 *     academicView="Sharpe = (Rp - Rf) / σp"
 *   />
 */
export default function InfoTooltip({
    label,
    traderView,
    academicView,
    className = ""
}: InfoTooltipProps) {
    const [isVisible, setIsVisible] = useState(false);

    return (
        <span
            className={`relative inline-flex items-center cursor-help ${className}`}
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
        >
            <Info size={14} className="text-zinc-500 hover:text-amber-400 transition-colors" />

            {/* Popover */}
            {isVisible && (
                <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-64 bg-[#1A1A24] border border-white/10 rounded-lg shadow-2xl z-[999] overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200">

                    {/* Header */}
                    <div className="px-3 py-2 border-b border-white/5 bg-[#12121A]">
                        <h4 className="font-bold text-white text-xs">{label}</h4>
                    </div>

                    {/* Trader View */}
                    <div className="px-3 py-2">
                        <span className="text-[9px] uppercase tracking-wider text-amber-500 font-bold block mb-0.5">
                            Trader View
                        </span>
                        <p className="text-[11px] text-zinc-300 leading-relaxed">
                            {traderView}
                        </p>
                    </div>

                    {/* Academic View (Optional) */}
                    {academicView && (
                        <div className="px-3 py-2 bg-zinc-900/50 border-t border-white/5">
                            <span className="text-[9px] uppercase tracking-wider text-zinc-500 font-bold block mb-0.5">
                                Formula
                            </span>
                            <p className="text-[11px] text-zinc-500 font-mono italic">
                                {academicView}
                            </p>
                        </div>
                    )}

                    {/* Arrow */}
                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-zinc-900 border-r border-b border-white/10" />
                </div>
            )}
        </span>
    );
}
