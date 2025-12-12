"use client";

import { useEffect, useState } from "react";
import { X, Terminal } from "lucide-react";

interface LogViewerProps {
    onClose: () => void;
}

export default function LogViewer({ onClose }: LogViewerProps) {
    const [logs, setLogs] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const res = await fetch("http://localhost:8000/logs");
                if (res.ok) {
                    const data = await res.json();
                    setLogs(data.logs || []);
                }
            } catch (error) {
                console.error("Failed to fetch logs:", error);
                setLogs(["Failed to connect to backend logs."]);
            } finally {
                setLoading(false);
            }
        };

        fetchLogs();
        const interval = setInterval(fetchLogs, 2000); // Poll every 2s

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="fixed inset-0 flex items-center justify-center bg-[#0A0A0F]/90 backdrop-blur-sm z-50">
            <div className="bg-[#12121A] border border-white/[0.08] rounded-xl shadow-2xl max-w-3xl w-full h-[600px] flex flex-col relative animate-scale-in">
                <div className="flex items-center justify-between p-4 border-b border-white/[0.08]">
                    <div className="flex items-center gap-2 text-white">
                        <Terminal size={20} className="text-amber-500" />
                        <h2 className="font-bold">System Logs</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-zinc-400 hover:text-white transition-colors"
                    >
                        <X size={24} />
                    </button>
                </div>

                <div className="flex-1 p-4 overflow-y-auto font-mono text-sm bg-black/50">
                    {loading && logs.length === 0 ? (
                        <div className="text-zinc-500 italic">Connecting to log stream...</div>
                    ) : (
                        <div className="flex flex-col gap-1">
                            {logs.map((log, i) => (
                                <div key={i} className="text-zinc-300 border-b border-zinc-800/50 pb-1 mb-1 last:border-0">
                                    <span className="text-zinc-500 mr-2">[{log.split(" - ")[0]}]</span>
                                    <span className={log.includes("ERROR") ? "text-red-400" : log.includes("WARNING") ? "text-yellow-400" : "text-emerald-400"}>
                                        {log.split(" - ").slice(1).join(" - ")}
                                    </span>
                                </div>
                            ))}
                            <div id="log-end" />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
