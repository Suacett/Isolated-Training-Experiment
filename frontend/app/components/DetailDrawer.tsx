"use client";

import { useState, useEffect } from "react";
import { X, Loader2 } from "lucide-react";
import {
    ComposedChart,
    Line,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Brush,
} from "recharts";
import { Asset } from "./AssetTable";

const API_BASE = "http://localhost:8000";

interface DetailDrawerProps {
    asset: Asset | null;
    onClose: () => void;
}

// API response type
interface HistoryItem {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    predicted_close: number | null;
    intrinsic_value: number;
}

interface ChartDataPoint {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    intrinsic: number;
    prediction: number | null;
    isCorrect: boolean;
    errorPct: number;
}

// --- Custom Candlestick Shape ---
const Candlestick = (props: {
    x?: number;
    width?: number;
    payload?: { open: number; high: number; low: number; close: number };
    yAxis?: { scale: (val: number) => number };
    xAxis?: unknown;
}) => {
    const { x, width, payload, yAxis, xAxis } = props;

    if (!yAxis || !xAxis || !payload || x === undefined || width === undefined) return null;

    const { open, close, high, low } = payload;
    const isBullish = close > open;
    const color = isBullish ? "#10b981" : "#f43f5e";

    const yHigh = yAxis.scale(high);
    const yLow = yAxis.scale(low);
    const yOpen = yAxis.scale(open);
    const yClose = yAxis.scale(close);

    const bodyTop = Math.min(yOpen, yClose);
    const bodyHeight = Math.abs(yOpen - yClose);

    return (
        <g>
            <line
                x1={x + width / 2}
                y1={yHigh}
                x2={x + width / 2}
                y2={yLow}
                stroke={color}
                strokeWidth={1}
            />
            <rect
                x={x}
                y={bodyTop}
                width={width}
                height={Math.max(2, bodyHeight)}
                fill={color}
                stroke="none"
            />
        </g>
    );
};

// --- Custom Tooltip ---
const CustomTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ payload: ChartDataPoint }>;
    label?: string;
}) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="bg-zinc-900 border border-zinc-700 p-3 rounded shadow-xl text-xs font-mono z-50">
                <div className="text-zinc-400 mb-2 font-bold">{label}</div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                    <span className="text-zinc-500">Open:</span>
                    <span className="text-right text-zinc-200">${data.open.toFixed(2)}</span>
                    <span className="text-zinc-500">High:</span>
                    <span className="text-right text-zinc-200">${data.high.toFixed(2)}</span>
                    <span className="text-zinc-500">Low:</span>
                    <span className="text-right text-zinc-200">${data.low.toFixed(2)}</span>
                    <span className="text-zinc-500">Close:</span>
                    <span className="text-right text-zinc-200">${data.close.toFixed(2)}</span>

                    <div className="col-span-2 h-px bg-zinc-800 my-1" />

                    {data.prediction !== null && (
                        <>
                            <span className="text-yellow-500">Pred:</span>
                            <span className="text-right text-yellow-500">${data.prediction.toFixed(2)}</span>
                        </>
                    )}
                    {data.intrinsic > 0 && (
                        <>
                            <span className="text-purple-400">Intr:</span>
                            <span className="text-right text-purple-400">${data.intrinsic.toFixed(2)}</span>
                        </>
                    )}
                    {data.prediction !== null && (
                        <>
                            <span className="text-zinc-500">Error:</span>
                            <span className="text-right text-zinc-300">{data.errorPct.toFixed(2)}%</span>
                        </>
                    )}
                </div>
            </div>
        );
    }
    return null;
};

export default function DetailDrawer({ asset, onClose }: DetailDrawerProps) {
    const [data, setData] = useState<ChartDataPoint[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [accuracy, setAccuracy] = useState<number | null>(null);

    // Fetch history data when asset changes
    useEffect(() => {
        if (!asset) return;

        const fetchHistory = async () => {
            setIsLoading(true);
            try {
                const res = await fetch(`${API_BASE}/dashboard/${asset.ticker}`);
                if (!res.ok) throw new Error("Failed to fetch history");

                const json = await res.json();
                const history: HistoryItem[] = json.history || [];

                // Transform API data to chart format
                const chartData: ChartDataPoint[] = history.map((item, index, arr) => {
                    // Calculate accuracy: was yesterday's prediction correct about today's direction?
                    let isCorrect = false;
                    if (index > 0 && arr[index - 1].predicted_close !== null) {
                        const prevPrediction = arr[index - 1].predicted_close!;
                        const prevClose = arr[index - 1].close;
                        const predictedDirection = prevPrediction > prevClose;
                        const actualDirection = item.close > prevClose;
                        isCorrect = predictedDirection === actualDirection;
                    }

                    const errorPct = item.predicted_close !== null
                        ? Math.abs((item.predicted_close - item.close) / item.close) * 100
                        : 0;

                    return {
                        date: new Date(item.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
                        open: item.open,
                        high: item.high,
                        low: item.low,
                        close: item.close,
                        intrinsic: item.intrinsic_value || 0,
                        prediction: item.predicted_close,
                        isCorrect,
                        errorPct,
                    };
                });

                setData(chartData);

                // Calculate overall accuracy
                const withPredictions = chartData.filter((_, i) => i > 0 && chartData[i - 1].prediction !== null);
                if (withPredictions.length > 0) {
                    const correctCount = withPredictions.filter(d => d.isCorrect).length;
                    setAccuracy(Math.round((correctCount / withPredictions.length) * 100));
                } else {
                    setAccuracy(null);
                }
            } catch (err) {
                console.error("Failed to fetch history:", err);
                setData([]);
            } finally {
                setIsLoading(false);
            }
        };

        fetchHistory();
    }, [asset]);

    if (!asset) return null;

    return (
        <div className="fixed inset-y-0 right-0 w-[800px] bg-zinc-950 border-l border-zinc-800 shadow-2xl transform transition-transform duration-300 ease-in-out z-50 flex flex-col">
            {/* Header */}
            <div className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-900/50 backdrop-blur">
                <div>
                    <h2 className="text-xl font-bold text-white flex items-center gap-3">
                        {asset.ticker}
                        <span className="text-xs font-normal text-zinc-400 px-2 py-0.5 rounded border border-zinc-700 bg-zinc-800">
                            PRO TRADER VIEW
                        </span>
                    </h2>
                    <div className="flex gap-4 mt-1 text-sm">
                        <span className="text-zinc-400">
                            Current: <span className="text-white font-mono">${asset.price.toFixed(2)}</span>
                        </span>
                        <span className="text-zinc-400">
                            Target: <span className="text-emerald-400 font-mono">${asset.prediction.toFixed(2)}</span>
                        </span>
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-white transition-colors"
                >
                    <X size={20} />
                </button>
            </div>

            {/* Main Content */}
            <div className="flex-1 p-4 overflow-hidden flex flex-col">
                {isLoading ? (
                    <div className="flex-1 flex items-center justify-center">
                        <Loader2 className="animate-spin text-zinc-500" size={32} />
                    </div>
                ) : data.length === 0 ? (
                    <div className="flex-1 flex items-center justify-center text-zinc-500">
                        <p>No historical data available. Click "Sync Data" to fetch.</p>
                    </div>
                ) : (
                    <>
                        {/* Chart Container */}
                        <div className="flex-1 bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 flex flex-col min-h-0">
                            <div className="flex justify-between items-center mb-2">
                                <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
                                    Price Action & AI Forecast
                                </h3>
                                <div className="flex gap-2 text-xs">
                                    <div className="flex items-center gap-1">
                                        <div className="w-2 h-2 rounded-full bg-purple-400" />
                                        <span className="text-zinc-400">Intrinsic</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <div className="w-2 h-2 rounded-full bg-yellow-500" />
                                        <span className="text-zinc-400">AI Pred</span>
                                    </div>
                                </div>
                            </div>

                            <div className="flex-1 min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={data}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                        <XAxis
                                            dataKey="date"
                                            stroke="#52525b"
                                            fontSize={10}
                                            tickLine={false}
                                            axisLine={false}
                                            minTickGap={30}
                                        />
                                        <YAxis
                                            stroke="#52525b"
                                            fontSize={10}
                                            domain={['auto', 'auto']}
                                            tickLine={false}
                                            axisLine={false}
                                            tickFormatter={(val) => `$${val.toFixed(0)}`}
                                        />
                                        <Tooltip content={<CustomTooltip />} />

                                        {/* Intrinsic Value Line */}
                                        <Line
                                            type="monotone"
                                            dataKey="intrinsic"
                                            stroke="#a78bfa"
                                            strokeWidth={2}
                                            dot={false}
                                            activeDot={false}
                                        />

                                        {/* Prediction Line */}
                                        <Line
                                            type="monotone"
                                            dataKey="prediction"
                                            stroke="#eab308"
                                            strokeWidth={2}
                                            strokeDasharray="4 4"
                                            dot={false}
                                            activeDot={false}
                                        />

                                        {/* Candlesticks */}
                                        <Bar
                                            dataKey="close"
                                            shape={<Candlestick />}
                                            isAnimationActive={false}
                                        />

                                        <Brush
                                            dataKey="date"
                                            height={30}
                                            stroke="#52525b"
                                            fill="#18181b"
                                            tickFormatter={() => ""}
                                        />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Performance Strip */}
                        <div className="mt-4 h-16 bg-zinc-900/30 rounded-xl border border-zinc-800 p-3 flex flex-col justify-center">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
                                    AI Accuracy Stream (Last {data.length} Days)
                                </span>
                                <span className={`text-xs font-mono ${accuracy !== null && accuracy >= 50 ? "text-emerald-500" : "text-rose-500"}`}>
                                    {accuracy !== null ? `${accuracy}% Correct` : "N/A"}
                                </span>
                            </div>
                            <div className="flex gap-0.5 h-full w-full overflow-hidden">
                                {data.map((d: ChartDataPoint, i: number) => (
                                    <div
                                        key={i}
                                        className={`flex-1 rounded-sm ${d.isCorrect ? "bg-emerald-500/50" : "bg-rose-500/50"
                                            } hover:opacity-100 transition-opacity`}
                                        title={`${d.date}: ${d.isCorrect ? "Correct" : "Incorrect"}`}
                                    />
                                ))}
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
