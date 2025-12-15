"use client";

import { useState } from "react";
import { Brain, ChevronDown, ChevronUp, TrendingUp, TrendingDown, AlertTriangle } from "lucide-react";

export function V9ExplanationPanel() {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className="mt-8 p-6 bg-gradient-to-br from-emerald-900/20 to-zinc-900/50 rounded-2xl border border-emerald-800/50">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-between text-left"
            >
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Brain size={20} className="text-emerald-400" />
                    How V9 Actually Works
                </h3>
                {expanded ? <ChevronUp className="text-zinc-400" /> : <ChevronDown className="text-zinc-400" />}
            </button>

            {expanded && (
                <div className="mt-6 space-y-6">
                    {/* Why Transformer? */}
                    <section>
                        <h4 className="text-sm font-bold text-emerald-400 mb-2 uppercase tracking-wider">Why is V9 a Transformer?</h4>
                        <p className="text-sm text-zinc-400 leading-relaxed">
                            Unlike LSTMs which process data sequentially (like reading a book), Transformers use <strong className="text-white">attention</strong> to look at all 60 days simultaneously.
                            This allows V9 to spot patterns like "price went up on high volume 3 weeks ago, then consolidated, now showing similar setup" without the information fading over time.
                        </p>
                        <div className="mt-3 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700">
                            <code className="text-xs text-emerald-400">
                                Transformer = Better at cross-temporal patterns
                            </code>
                        </div>
                    </section>

                    {/* 12 Features */}
                    <section>
                        <h4 className="text-sm font-bold text-emerald-400 mb-2 uppercase tracking-wider">The 12 Stationary Features</h4>
                        <p className="text-sm text-zinc-400 mb-3">
                            V9 uses only <strong className="text-white">stationary</strong> features — no raw prices. It doesn't know if Apple is $50 or $500, only:
                        </p>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                            <div className="p-2 bg-zinc-800/50 rounded">📈 1-day, 5-day, 20-day returns</div>
                            <div className="p-2 bg-zinc-800/50 rounded">📊 RSI (14-day momentum)</div>
                            <div className="p-2 bg-zinc-800/50 rounded">📐 MACD histogram (trend)</div>
                            <div className="p-2 bg-zinc-800/50 rounded">📦 Volume ratio (vs 20-day avg)</div>
                            <div className="p-2 bg-zinc-800/50 rounded">📏 Distance from SMA20</div>
                            <div className="p-2 bg-zinc-800/50 rounded">🌊 20-day volatility</div>
                            <div className="p-2 bg-zinc-800/50 rounded">🏦 SPY 5-day return</div>
                            <div className="p-2 bg-zinc-800/50 rounded">😰 VIX level (fear gauge)</div>
                            <div className="p-2 bg-zinc-800/50 rounded">🔗 Correlation with SPY</div>
                            <div className="p-2 bg-zinc-800/50 rounded">💪 Relative strength vs SPY</div>
                        </div>
                    </section>

                    {/* Buy/Sell Logic */}
                    <section>
                        <h4 className="text-sm font-bold text-emerald-400 mb-2 uppercase tracking-wider">How V9 Decides to Buy or Sell</h4>
                        <div className="space-y-3 text-sm text-zinc-400">
                            <div className="flex items-start gap-3">
                                <TrendingUp className="text-emerald-500 flex-shrink-0 mt-0.5" size={16} />
                                <div>
                                    <strong className="text-emerald-400">BUY:</strong> Stock ranks in <strong className="text-white">Top 10</strong> among all 60 stocks checked,
                                    AND has <strong className="text-white">correlation &lt; 0.60</strong> with existing holdings (diversification).
                                </div>
                            </div>
                            <div className="flex items-start gap-3">
                                <TrendingDown className="text-rose-500 flex-shrink-0 mt-0.5" size={16} />
                                <div>
                                    <strong className="text-rose-400">SELL:</strong> Stock drops out of the Top 10 ranking OR
                                    a better-ranked, less-correlated stock is available to replace it.
                                </div>
                            </div>
                            <div className="flex items-start gap-3">
                                <AlertTriangle className="text-amber-500 flex-shrink-0 mt-0.5" size={16} />
                                <div>
                                    <strong className="text-amber-400">REBALANCE:</strong> Every 5 trading days, re-rank all stocks and
                                    adjust portfolio to maintain Top 10 with lowest correlation.
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Crash Behaviour */}
                    <section>
                        <h4 className="text-sm font-bold text-emerald-400 mb-2 uppercase tracking-wider">What Happens During Market Crashes?</h4>
                        <div className="space-y-3">
                            <div className="p-4 bg-rose-900/20 rounded-lg border border-rose-800/50">
                                <h5 className="font-bold text-rose-400 mb-2">🛑 VIX Safety Filter (VIX &gt; 30)</h5>
                                <p className="text-sm text-zinc-400">
                                    When VIX exceeds 30 (market panic like March 2020 or 2008), V9 automatically goes to <strong className="text-white">100% cash</strong>.
                                    No trading until VIX drops below threshold. This saved ~40% in the 2020 crash simulation.
                                </p>
                            </div>
                            <div className="p-4 bg-amber-900/20 rounded-lg border border-amber-800/50">
                                <h5 className="font-bold text-amber-400 mb-2">⚡ During Big Spikes</h5>
                                <p className="text-sm text-zinc-400">
                                    V9 doesn't chase momentum blindly. The <strong className="text-white">correlation filter</strong> prevents it from buying the same "hot sector" 10 times.
                                    During the AI bubble, it held NVDA but also kept positions in healthcare and financials.
                                </p>
                            </div>
                            <div className="p-4 bg-emerald-900/20 rounded-lg border border-emerald-800/50">
                                <h5 className="font-bold text-emerald-400 mb-2">📈 Recovery Mode</h5>
                                <p className="text-sm text-zinc-400">
                                    After crashes, relative rankings shift dramatically. V9 naturally rotates into beaten-down leaders
                                    (stocks with strong fundamentals that dropped) because their <strong className="text-white">relative strength</strong> improves vs others still falling.
                                </p>
                            </div>
                        </div>
                    </section>

                    {/* Performance Summary */}
                    <section className="p-4 bg-emerald-900/30 rounded-lg border border-emerald-700/50">
                        <h4 className="text-sm font-bold text-emerald-400 mb-2">📊 5-Year Backtest Summary</h4>
                        <div className="grid grid-cols-3 gap-4 text-center">
                            <div>
                                <div className="text-2xl font-bold text-emerald-400">+93%</div>
                                <div className="text-xs text-zinc-500">Total Return</div>
                            </div>
                            <div>
                                <div className="text-2xl font-bold text-white">~14%</div>
                                <div className="text-xs text-zinc-500">CAGR</div>
                            </div>
                            <div>
                                <div className="text-2xl font-bold text-amber-400">Top 10</div>
                                <div className="text-xs text-zinc-500">Max Holdings</div>
                            </div>
                        </div>
                    </section>
                </div>
            )}
        </div>
    );
}
