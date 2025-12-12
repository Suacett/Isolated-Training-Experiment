"use client";

import { useState, useEffect } from "react";
import { TrendingUp, AlertCircle, RefreshCw } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useToast } from "./Toast";

interface Portfolio {
  session_id: string;
  name: string;
  category: string;
  initial_value: number;
  final_value: number;
  total_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  days_simulated: number;
  equity_curve: Array<{ date: string; value: number }>;
}

interface ComparisonData {
  portfolios: Portfolio[];
  statistics: {
    best_return: number;
    worst_return: number;
    avg_return: number;
    std_return: number;
    best_sharpe: number;
    worst_drawdown: number;
    avg_drawdown: number;
  };
  count: number;
}

// Color palette for 10 different portfolios
const PORTFOLIO_COLORS = [
  "#10b981", // Emerald
  "#3b82f6", // Blue
  "#f59e0b", // Amber
  "#ef4444", // Red
  "#8b5cf6", // Purple
  "#06b6d4", // Cyan
  "#ec4899", // Pink
  "#14b8a6", // Teal
  "#f97316", // Orange
  "#6366f1", // Indigo
];

export default function PortfolioComparison() {
  const [data, setData] = useState<ComparisonData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedPortfolios, setSelectedPortfolios] = useState<Set<string>>(new Set());
  const { showToast } = useToast();

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch("http://localhost:8000/portfolio/comparison");

        if (!response.ok) {
          if (response.status === 404) {
            showToast("No portfolio data found. Run the ensemble generator first.", "info");
            setData(null);
          } else {
            showToast("Failed to load portfolio comparison data", "error");
          }
          return;
        }

        const comparisonData = await response.json();
        setData(comparisonData);

        // Default to showing all portfolios
        setSelectedPortfolios(new Set(comparisonData.portfolios.map((p: Portfolio) => p.session_id)));
      } catch (error) {
        console.error("Error loading portfolio data:", error);
        showToast("Error loading portfolio data", "error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [showToast]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="flex flex-col items-center gap-3">
          <RefreshCw className="w-8 h-8 text-amber-400 animate-spin" />
          <p className="text-zinc-400">Loading portfolio comparison...</p>
        </div>
      </div>
    );
  }

  if (!data || data.portfolios.length === 0) {
    return (
      <div className="bg-amber-900/20 border border-amber-700/50 rounded-lg p-6">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-amber-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="text-amber-100 font-semibold mb-1">No Portfolio Data</h3>
            <p className="text-amber-200/80 text-sm">
              Generate the 10-portfolio ensemble first using the backend script to compare strategies.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Prepare data for chart
  const chartData = data.portfolios[0]?.equity_curve || [];
  const portfoliosForChart = data.portfolios.filter(p => selectedPortfolios.has(p.session_id));

  // Toggle portfolio visibility in chart
  const togglePortfolio = (sessionId: string) => {
    const newSelected = new Set(selectedPortfolios);
    if (newSelected.has(sessionId)) {
      newSelected.delete(sessionId);
    } else {
      newSelected.add(sessionId);
    }
    setSelectedPortfolios(newSelected);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 mb-6">
        <TrendingUp className="w-6 h-6 text-amber-400" />
        <h2 className="text-2xl font-bold text-white">10-Portfolio Strategy Comparison</h2>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#1A1A24] border border-white/[0.08] rounded-lg p-4">
          <p className="text-zinc-400 text-xs uppercase mb-1">Best Return</p>
          <p className="text-emerald-400 font-bold text-lg">{data.statistics.best_return.toFixed(2)}%</p>
        </div>
        <div className="bg-[#1A1A24] border border-white/[0.08] rounded-lg p-4">
          <p className="text-zinc-400 text-xs uppercase mb-1">Avg Return</p>
          <p className="text-blue-400 font-bold text-lg">{data.statistics.avg_return.toFixed(2)}%</p>
        </div>
        <div className="bg-[#1A1A24] border border-white/[0.08] rounded-lg p-4">
          <p className="text-zinc-400 text-xs uppercase mb-1">Best Sharpe</p>
          <p className="text-emerald-400 font-bold text-lg">{data.statistics.best_sharpe.toFixed(2)}</p>
        </div>
        <div className="bg-[#1A1A24] border border-white/[0.08] rounded-lg p-4">
          <p className="text-zinc-400 text-xs uppercase mb-1">Avg Drawdown</p>
          <p className="text-red-400 font-bold text-lg">{data.statistics.avg_drawdown.toFixed(2)}%</p>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-[#1A1A24] border border-white/[0.08] rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-white/[0.08] bg-zinc-900/50">
                <th className="px-4 py-3 text-xs font-medium text-zinc-400 uppercase">Strategy</th>
                <th className="px-4 py-3 text-xs font-medium text-zinc-400 uppercase">Category</th>
                <th className="px-4 py-3 text-xs font-medium text-zinc-400 uppercase text-right">Return</th>
                <th className="px-4 py-3 text-xs font-medium text-zinc-400 uppercase text-right">Max DD</th>
                <th className="px-4 py-3 text-xs font-medium text-zinc-400 uppercase text-right">Sharpe</th>
                <th className="px-4 py-3 text-xs font-medium text-zinc-400 uppercase text-right">Final Value</th>
              </tr>
            </thead>
            <tbody>
              {data.portfolios.map((portfolio, idx) => (
                <tr
                  key={portfolio.session_id}
                  className="border-b border-white/[0.08] hover:bg-zinc-800/30 transition-colors cursor-pointer"
                  onClick={() => togglePortfolio(portfolio.session_id)}
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{ backgroundColor: PORTFOLIO_COLORS[idx % PORTFOLIO_COLORS.length] }}
                      />
                      <div className="text-sm font-medium text-white">{portfolio.name}</div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-zinc-400 capitalize">{portfolio.category}</td>
                  <td className="px-4 py-3 text-sm text-right font-medium">
                    <span className={portfolio.total_return >= 0 ? "text-emerald-400" : "text-red-400"}>
                      {portfolio.total_return >= 0 ? "+" : ""}{portfolio.total_return.toFixed(2)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-right text-red-400 font-medium">
                    {portfolio.max_drawdown.toFixed(2)}%
                  </td>
                  <td className="px-4 py-3 text-sm text-right text-blue-400 font-medium">
                    {portfolio.sharpe_ratio.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 text-sm text-right text-zinc-300 font-mono">
                    ${portfolio.final_value.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Multi-Portfolio Equity Curve Chart */}
      <div className="bg-[#1A1A24] border border-white/[0.08] rounded-lg p-6">
        <h3 className="text-white font-semibold mb-4">5-Year Equity Curve Comparison</h3>
        <p className="text-xs text-zinc-400 mb-4">
          Click portfolio rows above to toggle lines on/off. Click legend to show/hide individual portfolios.
        </p>

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#a1a1aa", fontSize: 12 }}
              interval={Math.floor(chartData.length / 10)}
              stroke="#52525b"
            />
            <YAxis
              tick={{ fill: "#a1a1aa", fontSize: 12 }}
              stroke="#52525b"
              label={{ value: "Portfolio Value ($)", angle: -90, position: "insideLeft" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0A0A0F",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px"
              }}
              formatter={(value: any) => `$${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}`}
              labelStyle={{ color: "#fff" }}
            />
            <Legend wrapperStyle={{ paddingTop: "20px" }} />

            {portfoliosForChart.map((portfolio, idx) => (
              <Line
                key={portfolio.session_id}
                type="monotone"
                dataKey={portfolio.session_id}
                stroke={PORTFOLIO_COLORS[idx % PORTFOLIO_COLORS.length]}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                data={portfolio.equity_curve.map(point => ({
                  date: point.date,
                  [portfolio.session_id]: point.value
                }))}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend / Portfolio Selector */}
      <div className="bg-[#1A1A24] border border-white/[0.08] rounded-lg p-4">
        <h3 className="text-white font-semibold mb-3 text-sm">Portfolio Visibility</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
          {data.portfolios.map((portfolio, idx) => (
            <button
              key={portfolio.session_id}
              onClick={() => togglePortfolio(portfolio.session_id)}
              className={`flex items-center gap-2 px-3 py-2 rounded text-xs transition-all ${
                selectedPortfolios.has(portfolio.session_id)
                  ? "bg-zinc-700 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700/50"
              }`}
            >
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: PORTFOLIO_COLORS[idx % PORTFOLIO_COLORS.length] }}
              />
              <span className="truncate">{portfolio.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Chart Note */}
      <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
        <p className="text-blue-200 text-sm">
          💡 <strong>How to interpret:</strong> Each line represents a different V9 portfolio strategy. Monte Carlo variations
          show strategy consistency, while parameter variations show the impact of configuration choices.
        </p>
      </div>
    </div>
  );
}
