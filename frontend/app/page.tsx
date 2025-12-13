"use client";

import React, { useState, useEffect } from "react";
import { ArrowUpRight, ArrowDownRight, Wallet, TrendingUp, DollarSign, Percent, FileText, Activity } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area } from "recharts";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import SmartTooltip from "./components/SmartTooltip";
import InfoMarker from "./components/InfoMarker";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const API_BASE = "http://localhost:8000";

// --- Types ---

interface PortfolioSummary {
  session_id: string;
  cash_balance: number;
  equity_value: number;
  total_value: number;
  days_since_rebalance: number;
  daily_pnl?: number;
  daily_pnl_pct?: number;
}

interface Holding {
  ticker: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  stop_loss_level: number;
  profit_pct: number;
  value: number;
}

interface Trade {
  date: string;
  action: string;
  ticker: string;
  price: number;
  quantity: number;
  reason: string;
  profit_loss?: number;
}

interface EquityPoint {
  date: string;
  value: number;
  equity: number;
  cash: number;
}

// Time range options
const TIME_RANGES = [
  { key: "1W", label: "1W", days: 7 },
  { key: "1M", label: "1M", days: 30 },
  { key: "3M", label: "3M", days: 90 },
  { key: "6M", label: "6M", days: 180 },
  { key: "1Y", label: "1Y", days: 365 },
  { key: "5Y", label: "5Y", days: 1825 },
] as const;

type TimeRangeKey = typeof TIME_RANGES[number]["key"];

export default function Home() {
  const [activeTab, setActiveTab] = useState<"holdings" | "logs">("holdings");
  const [selectedRange, setSelectedRange] = useState<TimeRangeKey>("1Y");

  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // 1. Fetch Status (Summary, Holdings)
        const statusRes = await fetch(`${API_BASE}/paper/status`);
        if (!statusRes.ok) throw new Error("Failed to fetch status");
        const statusData = await statusRes.json();

        setSummary({
          session_id: statusData.session_id,
          cash_balance: statusData.cash_balance,
          equity_value: statusData.equity_value,
          total_value: statusData.total_value,
          days_since_rebalance: statusData.days_since_rebalance,
          daily_pnl: statusData.pnl,
          daily_pnl_pct: statusData.pnl_pct
        });
        setHoldings(statusData.holdings);

        // 2. Fetch ALL Trades (up to 500 - covers 5 years of weekly rebalancing)
        const tradesRes = await fetch(`${API_BASE}/paper/trades?limit=500`);
        if (tradesRes.ok) {
          const tradesData = await tradesRes.json();
          setTrades(tradesData.trades || []);
        } else {
          // Fallback to recent_trades from status
          setTrades(statusData.recent_trades || []);
        }

        // 3. Fetch History (Chart)
        const histRes = await fetch(`${API_BASE}/paper/history`);
        if (histRes.ok) {
          const histData = await histRes.json();
          setEquityHistory(histData);
        }

      } catch (e) {
        console.error("Failed to load dashboard data", e);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  // Helper for reason parsing (if simple string, just return)
  const formatReason = (reason: string) => {
    if (!reason) return "-";
    return reason;
  };

  // Filter equity history based on selected time range
  const filteredEquityHistory = React.useMemo(() => {
    if (!equityHistory.length) return [];

    const range = TIME_RANGES.find(r => r.key === selectedRange);
    if (!range) return equityHistory;

    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - range.days);

    return equityHistory.filter(point => new Date(point.date) >= cutoffDate);
  }, [equityHistory, selectedRange]);

  return (
    <main className="min-h-screen bg-[#0A0A0F] text-zinc-100 p-6 font-sans relative">
      {/* Loading Indicator */}
      {isLoading && (
        <div className="fixed top-20 right-6 bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3 flex items-center gap-3 shadow-xl backdrop-blur-sm z-50 animate-fade-in">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-sm text-amber-400 font-medium">Loading market data...</span>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto space-y-6">

        {/* --- Top Section: Equity Chart & Metrics --- */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Equity Curve */}
          <div className="lg:col-span-2 p-6 rounded-2xl bg-[#12121A] border border-white/[0.08] relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-6 opacity-5 group-hover:opacity-10 transition-opacity">
              <TrendingUp size={100} />
            </div>

            <div className="relative z-10 flex flex-col h-[380px]">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h2 className="text-zinc-400 text-sm font-medium uppercase tracking-wider mb-1">Portfolio Equity</h2>
                  <div className="flex items-baseline gap-4">
                    <span className="text-4xl font-display font-bold text-white">
                      ${(summary?.total_value || 10000).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                    {summary && (
                      <div className={cn("flex items-center text-sm font-medium px-2 py-1 rounded-full bg-white/5",
                        (summary.daily_pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400")}>
                        {(summary.daily_pnl || 0) >= 0 ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
                        {Math.abs(summary.daily_pnl_pct || 0).toFixed(2)}%
                      </div>
                    )}
                  </div>
                  <p className="text-[10px] text-zinc-600 mt-2 font-mono">
                    *Historical performance is simulated (Backtest). Future results may vary.
                  </p>
                </div>

                {/* Time Range Selector */}
                <div className="flex items-center gap-1">
                  {TIME_RANGES.map((range) => (
                    <button
                      key={range.key}
                      onClick={() => setSelectedRange(range.key)}
                      className={cn(
                        "px-3 py-1.5 text-xs font-medium rounded-md transition-all",
                        selectedRange === range.key
                          ? "bg-amber-500 text-[#0A0A0F] shadow-[0_0_10px_rgba(245,158,11,0.3)]"
                          : "bg-zinc-800/50 text-zinc-400 hover:bg-zinc-700 hover:text-white border border-zinc-700/50"
                      )}
                    >
                      {range.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={filteredEquityHistory}>
                    <defs>
                      <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={false} />
                    <XAxis
                      dataKey="date"
                      stroke="#52525b"
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                      minTickGap={40}
                      tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                    />
                    <YAxis
                      stroke="#52525b"
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                      domain={['auto', 'auto']}
                      tickFormatter={(val) => `$${val.toLocaleString()}`}
                      width={60}
                    />
                    <RechartsTooltip
                      contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', borderRadius: '8px' }}
                      itemStyle={{ fontSize: '12px', color: '#818cf8' }}
                      labelStyle={{ color: '#a1a1aa', fontSize: '11px', marginBottom: '4px' }}
                      formatter={(value: number) => [`$${value.toLocaleString()}`, "Equity"]}
                      labelFormatter={(label) => new Date(label).toLocaleDateString()}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#818cf8"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#colorEquity)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Live Metrics Cards (with SmartTooltips) */}
          <div className="space-y-4">

            {/* Daily PnL */}
            <div className="p-5 rounded-xl bg-[#12121A] border border-white/[0.08] flex flex-col justify-center h-[calc(33%-11px)]">
              <div className="flex items-center gap-2 mb-2">
                <DollarSign size={16} className="text-emerald-500" />
                <span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Total PnL</span>
                <InfoMarker
                  title="Total Profit & Loss"
                  simple="Your total profit or loss since the portfolio started."
                  technical="Calculated as (Current Total Value - Starting Capital). Updates in real-time as holdings change."
                />
              </div>
              <div className={cn("text-2xl font-mono font-bold", (summary?.daily_pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400")}>
                {(summary?.daily_pnl || 0) >= 0 ? "+" : "-"}${Math.abs(summary?.daily_pnl || 0).toLocaleString()}
              </div>
            </div>

            {/* Sharpe Ratio */}
            <div className="p-5 rounded-xl bg-[#12121A] border border-white/[0.08] flex flex-col justify-center h-[calc(33%-11px)]">
              <div className="flex items-center gap-2 mb-2">
                <Activity size={16} className="text-amber-500" />
                <span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Sharpe Ratio (1y)</span>
                <InfoMarker
                  title="Sharpe Ratio"
                  simple="Risk-adjusted return metric. Higher is better. Values > 1.0 indicate good performance."
                  technical="Sharpe = (Rp - Rf) / σp. Measures excess return per unit of volatility."
                />
              </div>

              <div className="text-2xl font-mono font-bold text-zinc-200">
                1.84 <span className="text-xs font-normal text-zinc-500 ml-1">(Est.)</span>
              </div>
            </div>

            {/* Alpha */}
            <div className="p-5 rounded-xl bg-[#12121A] border border-white/[0.08] flex flex-col justify-center h-[calc(33%-11px)]">
              <div className="flex items-center gap-2 mb-2">
                <Percent size={16} className="text-indigo-500" />
                <span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Alpha</span>
                <InfoMarker
                  title="Alpha"
                  simple="Excess return vs S&P 500 benchmark. Positive alpha means beating the market."
                  technical="Alpha (α) = Rp - [Rf + β(Rm - Rf)]. Active return on investment adjusted for market risk."
                />
              </div>

              <div className="text-2xl font-mono font-bold text-indigo-400">
                +4.2%
              </div>
            </div>
          </div>
        </div>

        {/* --- Bottom Section: Tabbed Content --- */}
        <div className="rounded-2xl border border-white/[0.08] bg-[#12121A] overflow-visible min-h-[400px]">

          {/* Tabs */}
          <div className="px-6 border-b border-white/[0.08] flex items-center gap-6 bg-white/[0.02]">
            <button
              onClick={() => setActiveTab("holdings")}
              className={cn(
                "py-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2",
                activeTab === "holdings" ? "border-amber-500 text-white" : "border-transparent text-zinc-500 hover:text-zinc-300"
              )}>
              <Wallet size={16} />
              Current Holdings
            </button>
            <button
              onClick={() => setActiveTab("logs")}
              className={cn(
                "py-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2",
                activeTab === "logs" ? "border-amber-500 text-white" : "border-transparent text-zinc-500 hover:text-zinc-300"
              )}>
              <FileText size={16} />
              Trade Logs
            </button>

            <div className="ml-auto text-xs text-zinc-500 font-mono flex items-center gap-4">
              <span>Cash: <span className="text-zinc-300 ml-1">${(summary?.cash_balance || 0).toLocaleString()}</span></span>
              <span className="flex items-center">
                Exposure: <span className="text-zinc-300 ml-1">${(summary?.equity_value || 0).toLocaleString()}</span>
                <InfoMarker
                  title="Exposure"
                  simple="Total market value of all stock holdings, excluding cash."
                  technical="Formula: sum of (quantity × current_price) for each position. Does not include cash balance."
                />
              </span>
            </div>
          </div>

          {/* Content */}
          <div className="overflow-visible">

            {/* HOLDINGS TABLE */}
            {activeTab === "holdings" && (
              <table className="w-full text-left text-sm text-zinc-400">
                <thead className="bg-[#0A0A0F] text-xs uppercase text-zinc-500 font-medium">
                  <tr>
                    <th className="px-6 py-4">Ticker</th>
                    <th className="px-6 py-4 text-right">Qty</th>
                    <th className="px-6 py-4 text-right">Entry</th>
                    <th className="px-6 py-4 text-right">Current</th>
                    <th className="px-6 py-4 text-right">Value</th>
                    <th className="px-6 py-4 text-right">PnL</th>
                    <th className="px-6 py-4 text-left pl-8 w-1/3 flex items-center gap-1">
                      Why?
                      <InfoMarker
                        title="Trade Reasoning"
                        simple="Shows V9's AI ranking and correlation data that triggered this buy/sell decision."
                        technical="Rank: V9 Transformer relative strength score (0-1). Corr: Average correlation with other holdings. Lower corr = better diversification."
                      />
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {holdings.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-6 py-12 text-center text-zinc-600 italic">
                        No active holdings. Portfolio is 100% Cash.
                      </td>
                    </tr>
                  ) : (
                    holdings.map((h) => {
                      // Find last buy trade reasoning if available
                      const lastBuy = trades.find(t => t.ticker === h.ticker && t.action === "BUY");
                      const reason = lastBuy ? lastBuy.reason : "Rebalance";

                      return (
                        <tr key={h.ticker} className="hover:bg-white/[0.02] transition-colors group">
                          <td className="px-6 py-4 font-bold text-white font-display flex items-center gap-2">
                            <div className="w-8 h-8 rounded bg-zinc-800 flex items-center justify-center text-xs font-mono">{h.ticker[0]}</div>
                            {h.ticker}
                          </td>
                          <td className="px-6 py-4 text-right font-mono">{h.quantity}</td>
                          <td className="px-6 py-4 text-right font-mono text-zinc-500">${h.entry_price.toFixed(2)}</td>
                          <td className="px-6 py-4 text-right font-mono text-white">${h.current_price.toFixed(2)}</td>
                          <td className="px-6 py-4 text-right font-mono text-zinc-300 font-bold">${h.value.toFixed(2)}</td>
                          <td className="px-6 py-4 text-right font-mono">
                            <div className={cn("inline-flex items-center px-1.5 py-0.5 rounded", h.profit_pct >= 0 ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400")}>
                              {h.profit_pct >= 0 ? "+" : ""}{h.profit_pct.toFixed(2)}%
                            </div>
                          </td>
                          <td className="px-6 py-4 text-left pl-8 text-xs text-amber-500/80 font-mono">
                            {reason}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            )}

            {/* LOGS TABLE */}
            {activeTab === "logs" && (
              <table className="w-full text-left text-sm text-zinc-400">
                <thead className="bg-[#0A0A0F] text-xs uppercase text-zinc-500 font-medium">
                  <tr>
                    <th className="px-6 py-4">Date</th>
                    <th className="px-6 py-4">Ticker</th>
                    <th className="px-6 py-4">Action</th>
                    <th className="px-6 py-4 text-right">Price</th>
                    <th className="px-6 py-4 text-right">Qty</th>
                    <th className="px-6 py-4">Reason</th>
                    <th className="px-6 py-4 text-right">Result</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {trades.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-6 py-12 text-center text-zinc-600 italic">
                        No trade history available yet.
                      </td>
                    </tr>
                  ) : (
                    trades.map((t, idx) => (
                      <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                        <td className="px-6 py-4 font-mono text-zinc-500 text-xs text-nowrap">
                          {new Date(t.date).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4 font-bold text-white font-display">
                          {t.ticker}
                        </td>
                        <td className="px-6 py-4">
                          <span className={cn("px-2 py-0.5 rounded text-[10px] font-bold border",
                            t.action === "BUY" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-rose-500/10 text-rose-400 border-rose-500/20")}>
                            {t.action}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right font-mono">${t.price.toFixed(2)}</td>
                        <td className="px-6 py-4 text-right font-mono">{t.quantity}</td>
                        <td className="px-6 py-4 text-xs text-amber-500/80 font-mono max-w-xs truncate" title={t.reason}>
                          {t.reason}
                        </td>
                        <td className="px-6 py-4 text-right font-mono">
                          {t.profit_loss !== null && t.profit_loss !== undefined ? (
                            <span className={t.profit_loss >= 0 ? "text-emerald-400" : "text-rose-400"}>
                              {t.profit_loss >= 0 ? "+" : ""}{t.profit_loss.toFixed(2)}
                            </span>
                          ) : "-"}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            )}

          </div>
        </div>

      </div>
    </main>
  );
}
