"use client";

import React, { useState } from "react";
import { Info } from "lucide-react";

interface SmartTooltipProps {
    term: string;
    value?: string | number | null;
    traderExplanation: string;
    academicExplanation: string;
    children?: React.ReactNode;
}

export default function SmartTooltip({
    term,
    value,
    traderExplanation,
    academicExplanation,
    children
}: SmartTooltipProps) {
    const [isVisible, setIsVisible] = useState(false);

    return (
        <div
            className="relative inline-flex items-center cursor-help group outline-none focus:ring-1 focus:ring-amber-500/50 rounded"
            tabIndex={0}
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
            onFocus={() => setIsVisible(true)}
            onBlur={() => setIsVisible(false)}
            onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setIsVisible(!isVisible);
                }
            }}
        >
            {children || (
                <div className="flex items-center gap-1">
                    <span className="border-b border-dotted border-zinc-500">{term}</span>
                    <Info size={12} className="text-zinc-500" />
                </div>
            )}

            {/* Popover */}
            {isVisible && (
                <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-72 bg-[#1A1A24] border border-white/10 rounded-xl shadow-2xl z-[999] overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200">

                    {/* Header: Term & Value */}
                    <div className="px-4 py-3 border-b border-white/5 bg-[#12121A]">
                        <div className="flex justify-between items-center">
                            <h4 className="font-bold text-white text-sm">{term}</h4>
                            {value && <span className="font-mono text-emerald-400 text-xs">{value}</span>}
                        </div>
                    </div>

                    {/* Trader View (Practical) */}
                    <div className="px-4 py-3 bg-[#1A1A24]">
                        <span className="text-[10px] uppercase tracking-wider text-amber-500 font-bold mb-1 block">Trader View</span>
                        <p className="text-xs text-zinc-300 leading-relaxed">
                            {traderExplanation}
                        </p>
                    </div>

                    {/* Academic View (Theoretical) */}
                    <div className="px-4 py-3 bg-zinc-900 border-t border-white/5">
                        <span className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold mb-1 block">Academic Formula</span>
                        <p className="text-xs text-zinc-500 font-serif italic leading-relaxed">
                            {academicExplanation}
                        </p>
                    </div>

                    {/* Arrow */}
                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-zinc-900 border-r border-b border-white/10"></div>
                </div>
            )}
        </div>
    );
}
