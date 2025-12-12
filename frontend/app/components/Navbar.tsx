"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Terminal, Brain, FlaskConical, ScrollText, Settings, LayoutDashboard, ScanSearch } from "lucide-react";
import SetupModal from "./SetupModal";
import LogViewer from "./LogViewer";

const API_BASE = "http://localhost:8000";

export default function Navbar() {
    const pathname = usePathname();
    const [status, setStatus] = useState<{ configured: boolean; model_loaded: boolean } | null>(null);
    const [equity, setEquity] = useState<number | null>(null);
    const [isSetupOpen, setIsSetupOpen] = useState(false);
    const [isLogsOpen, setIsLogsOpen] = useState(false);

    useEffect(() => {
        // Fetch System Status
        fetch(`${API_BASE}/status`)
            .then(res => res.json())
            .then(data => setStatus({
                configured: data.configured,
                model_loaded: data.ai_model_loaded
            }))
            .catch(console.error);

        // Fetch Live Equity
        fetch(`${API_BASE}/paper/status`)
            .then(res => res.json())
            .then(data => {
                if (data && data.total_value) {
                    setEquity(data.total_value);
                }
            })
            .catch(console.error);
    }, []);

    const isActive = (path: string) => pathname === path;
    const navClass = (path: string) => `p-2 rounded-lg transition-all flex items-center gap-2 ${isActive(path) ? "bg-amber-500/20 text-amber-400 border border-amber-500/30" : "hover:bg-white/5 text-zinc-400 hover:text-white"}`;

    return (
        <>
        <header className="border-b border-white/[0.08] bg-[#12121A]/80 backdrop-blur-md sticky top-0 z-40 h-16">
            <div className="max-w-7xl mx-auto px-6 h-full flex items-center justify-between">

                {/* Logo & Brand */}
                <div className="flex items-center gap-8">
                    <Link href="/" className="flex items-center gap-2 group">
                        <div className={`w-3 h-3 rounded-full transition-all ${status?.configured ? "bg-amber-500 group-hover:shadow-[0_0_10px_rgba(245,158,11,0.5)]" : "bg-red-500"}`} />
                        <h1 className="text-lg font-bold tracking-tight text-white font-display">
                            V9 Terminal
                        </h1>
                    </Link>

                    {/* Main Nav */}
                    <nav className="flex items-center gap-1">
                        <Link href="/" className={navClass("/")}>
                            <LayoutDashboard size={18} />
                            <span className="text-sm font-medium">Dashboard</span>
                        </Link>
                        <Link href="/scanner" className={navClass("/scanner")}>
                            <ScanSearch size={18} />
                            <span className="text-sm font-medium">Scanner</span>
                        </Link>
                        <Link href="/playground" className={navClass("/playground")}>
                            <FlaskConical size={18} />
                            <span className="text-sm font-medium">Playground</span>
                        </Link>
                    </nav>
                </div>

                {/* Right Actions */}
                <div className="flex items-center gap-3">

                    {/* Live Equity Badge (New) */}
                    {equity !== null && (
                        <div className="hidden md:flex flex-col items-end mr-4">
                            <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-bold">Paper Equity</span>
                            <span className={`text-sm font-mono font-bold ${equity >= 10000 ? "text-emerald-400" : "text-rose-400"}`}>
                                ${equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                            </span>
                        </div>
                    )}

                    {/* Status Badge */}
                    {status && (
                        <div className="px-3 py-1 rounded-full bg-white/5 border border-white/10 flex items-center gap-2 text-xs">
                            <span className={`w-1.5 h-1.5 rounded-full ${status.model_loaded ? "bg-emerald-500" : "bg-amber-500"}`} />
                            <span className="text-zinc-400">{status.model_loaded ? "V9 Active" : "No Model"}</span>
                        </div>
                    )}

                    <div className="h-6 w-px bg-white/10 mx-2" />

                    {/* Logs Button */}
                    <button
                        className="p-2 text-zinc-400 hover:text-white transition-colors"
                        title="System Logs"
                        onClick={() => setIsLogsOpen(true)}
                    >
                        <Terminal size={18} />
                    </button>

                    {/* Settings Button */}
                    <button
                        className="p-2 text-zinc-400 hover:text-white transition-colors"
                        title="Settings"
                        onClick={() => setIsSetupOpen(true)}
                    >
                        <Settings size={18} />
                    </button>

                    <div className="ml-2 px-2 py-0.5 rounded bg-[#1A1A24] border border-white/10 font-mono text-[10px] text-zinc-500">
                        v2.0
                    </div>
                </div>
            </div>
        </header>

        {/* Modals - Outside header to avoid z-index stacking context */}
        {isSetupOpen && <SetupModal onClose={() => setIsSetupOpen(false)} />}
        {isLogsOpen && <LogViewer onClose={() => setIsLogsOpen(false)} />}
        </>
    );
}
