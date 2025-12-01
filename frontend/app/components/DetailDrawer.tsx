"use client";

import { X, ZoomIn, ZoomOut } from "lucide-react";
import {
    ComposedChart,
    Line,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
    Brush,
    ReferenceLine,
} from "recharts";
import { Asset } from "./AssetTable";

interface DetailDrawerProps {
    asset: Asset | null;
    onClose: () => void;
}

// --- Mock Data Generator (90 Days OHLC) ---
const generateProHistory = (basePrice: number) => {
    const data = [];
    let currentPrice = basePrice;
    const now = new Date();

    for (let i = 90; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);

        const volatility = basePrice * 0.02; // 2% daily volatility
        const change = (Math.random() - 0.5) * volatility;

        const open = currentPrice;
        const close = currentPrice + change;
        const high = Math.max(open, close) + Math.random() * volatility * 0.5;
        const low = Math.min(open, close) - Math.random() * volatility * 0.5;

        // Intrinsic value (slow moving average-ish)
        const intrinsic = basePrice * (1 + Math.sin(i / 10) * 0.1);

        // Prediction (T+1, slightly noisy)
        const prediction = close * (1 + (Math.random() - 0.5) * 0.03);

        // Accuracy (Directional)
        // For simplicity, let's say "correct" if prediction direction matches next day's actual direction
        // But since we generate backwards, let's just randomize "correctness" for the strip
        const isCorrect = Math.random() > 0.4; // 60% accuracy

        data.push({
            date: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
            open,
            high,
            low,
            close,
            intrinsic,
            prediction,
            isCorrect,
            errorPct: Math.abs((prediction - close) / close) * 100,
        });

        currentPrice = close;
    }
    return data;
};

// --- Custom Candlestick Shape ---
const Candlestick = (props: any) => {
    const {
        x,
        y,
        width,
        height,
        low,
        high,
        open,
        close,
    } = props;

    const isBullish = close > open;
    const color = isBullish ? "#10b981" : "#f43f5e"; // Emerald-500 vs Rose-500
    const ratio = Math.abs(height / (open - close)); // Pixels per dollar

    // Calculate y-coordinates for high and low wicks
    // Recharts passes y for the 'top' of the bar (min value if inverted, but here y is top pixel)
    // We need to map values to pixels. 
    // Actually, Recharts custom shape props are a bit tricky. 
    // Let's rely on the passed `y` and `height` which correspond to the bar body (open/close).
    // But we need high/low pixels. 
    // A better approach for Recharts candlesticks is often using ErrorBar or just drawing lines relative to the body if we had the scale.
    // Since we don't have the scale easily in the shape, we can try a simpler approximation or use the `payload` if available.

    // Alternative: Use a standard Bar for the body, and ErrorBar for wicks? 
    // Or just draw the body and a line through the middle.
    // The `y` prop is the top of the bar, `height` is the height.
    // We need to know where High and Low are.
    // Fortunately, we can pass the y-scale or calculate relative positions if we passed them.

    // Let's try a simplified approach: 
    // We will assume the Bar component is rendering the [Min(Open, Close), Max(Open, Close)] range.
    // So `y` is the top of the body, `y + height` is the bottom.
    // We need to draw wicks to High and Low.
    // BUT, the Bar dataKey usually only takes one value. 
    // To do proper candlesticks in Recharts, we often use a ComposedChart with:
    // 1. Bar for the body (Open-Close range? No, Bar takes a value).
    // 2. ErrorBar? 

    // ACTUALLY, the standard way to do Candlesticks in Recharts is a bit hacky.
    // We will use a custom shape on a Bar that has dataKey="high" (max range) but we need to draw the body inside.
    // Let's pass the full payload to the shape.

    // Wait, simpler: 
    // We can use a Bar chart where the data is [min, max] for the range? No.

    // Let's stick to the "Pro" requirement:
    // We will draw the body using the `open` and `close` from `payload`.
    // We need the YAxis scale to convert values to pixels. 
    // Recharts passes `yAxis` to the custom shape!

    const { yAxis, xAxis } = props;
    if (!yAxis || !xAxis) return null;

    const yHigh = yAxis.scale(props.payload.high);
    const yLow = yAxis.scale(props.payload.low);
    const yOpen = yAxis.scale(props.payload.open);
    const yClose = yAxis.scale(props.payload.close);

    const bodyTop = Math.min(yOpen, yClose);
    const bodyHeight = Math.abs(yOpen - yClose);
    const bodyBottom = bodyTop + bodyHeight;

    return (
        <g>
            {/* Wick */}
            <line
                x1={x + width / 2}
                y1={yHigh}
                x2={x + width / 2}
                y2={yLow}
                stroke={color}
                strokeWidth={1}
            />
            {/* Body */}
            <rect
                x={x}
                y={bodyTop}
                width={width}
                height={Math.max(2, bodyHeight)} // Min height 2px so dojis are visible
                fill={color}
                stroke="none"
            />
        </g>
    );
};

// --- Custom Tooltip ---
const CustomTooltip = ({ active, payload, label }: any) => {
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

                    <span className="text-yellow-500">Pred:</span>
                    <span className="text-right text-yellow-500">${data.prediction.toFixed(2)}</span>
                    <span className="text-purple-400">Intr:</span>
                    <span className="text-right text-purple-400">${data.intrinsic.toFixed(2)}</span>
                    <span className="text-zinc-500">Error:</span>
                    <span className="text-right text-zinc-300">{data.errorPct.toFixed(2)}%</span>
                </div>
            </div>
        );
    }
    return null;
};

export default function DetailDrawer({ asset, onClose }: DetailDrawerProps) {
    if (!asset) return null;

    const data = generateProHistory(asset.price);

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
                                    stroke="#a78bfa" // Purple-400
                                    strokeWidth={2}
                                    dot={false}
                                    activeDot={false}
                                />

                                {/* Prediction Line */}
                                <Line
                                    type="monotone"
                                    dataKey="prediction"
                                    stroke="#eab308" // Yellow-500
                                    strokeWidth={2}
                                    strokeDasharray="4 4"
                                    dot={false}
                                    activeDot={false}
                                />

                                {/* Candlesticks (Using Bar with custom shape) */}
                                {/* We bind 'close' to dataKey just to give it a value range, but the shape uses payload */}
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
                        <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">AI Accuracy Stream (Last 90 Days)</span>
                        <span className="text-xs font-mono text-emerald-500">62% Correct</span>
                    </div>
                    <div className="flex gap-0.5 h-full w-full overflow-hidden">
                        {data.map((d, i) => (
                            <div
                                key={i}
                                className={`flex-1 rounded-sm ${d.isCorrect ? "bg-emerald-500/50" : "bg-rose-500/50"
                                    } hover:opacity-100 transition-opacity`}
                                title={`${d.date}: ${d.isCorrect ? "Correct" : "Incorrect"}`}
                            />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
