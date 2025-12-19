import React from "react";
import {
    ComposedChart,
    Line,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Brush,
    ReferenceLine,
} from "recharts";
import { ChartDataPoint, TIME_RANGES } from "./DetailDrawer.types";

// Custom Tooltip Component
export function CustomTooltip({ active, payload, showPercentage }: {
    active?: boolean;
    payload?: Array<{ payload: ChartDataPoint }>;
    showPercentage?: boolean;
}) {
    if (active && payload && payload.length) {
        const d = payload[0].payload;
        const diff = d.prediction !== null ? d.prediction - d.close : 0;
        const diffPct = (d.prediction !== null && d.close > 0) ? (diff / d.close) * 100 : 0;

        return (
            <div className="bg-zinc-900/95 border border-zinc-700 p-3 rounded-lg shadow-xl text-xs backdrop-blur min-w-[200px]">
                <div className="text-zinc-400 mb-2 font-bold border-b border-zinc-700 pb-1 flex justify-between">
                    <span>{d.date}</span>
                    <span className="text-zinc-500 font-normal">{d.fullDate.getFullYear()}</span>
                </div>
                <div className="space-y-1">
                    <div className="flex justify-between gap-4">
                        <span className="text-emerald-400">Price:</span>
                        <span className="font-mono text-white">
                            {showPercentage ? `${d.closePct.toFixed(2)}%` : `$${d.close.toFixed(2)}`}
                        </span>
                    </div>
                    {d.prediction !== null && (
                        <>
                            <div className="flex justify-between gap-4">
                                <span className="text-pink-400">AI Pred:</span>
                                <span className="font-mono text-white">
                                    {showPercentage ? `${d.predictionPct?.toFixed(2)}%` : `$${d.prediction.toFixed(2)}`}
                                </span>
                            </div>
                            <div className="flex justify-between gap-4 pt-1 border-t border-zinc-700/50 mt-1">
                                <span className="text-zinc-400">Diff:</span>
                                <span className={`font-mono ${diff >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {diff >= 0 ? '+' : ''}{diff.toFixed(2)} ({diffPct.toFixed(2)}%)
                                </span>
                            </div>
                            <div className="text-[10px] text-zinc-500 mt-1 italic">
                                {diff > 0 ? "Prediction was higher than actual." : "Prediction was lower than actual."}
                            </div>
                        </>
                    )}
                    {d.spyClose !== null && (
                        <div className="flex justify-between gap-4 pt-1 border-t border-zinc-700/50 mt-1">
                            <span className="text-blue-400">S&P 500:</span>
                            <span className="font-mono text-white">
                                {showPercentage ? `${d.spyPct?.toFixed(2)}%` : `$${d.spyClose.toFixed(2)}`}
                            </span>
                        </div>
                    )}
                    {d.intrinsic !== null && d.intrinsic > 0 && (
                        <div className="flex justify-between gap-4">
                            <span className="text-purple-400">Intrinsic:</span>
                            <span className="font-mono text-white">
                                {showPercentage ? `${d.intrinsicPct?.toFixed(2)}%` : `$${d.intrinsic.toFixed(2)}`}
                            </span>
                        </div>
                    )}
                </div>
            </div>
        );
    }
    return null;
}

// Memoized Chart Component
export const DetailChart = React.memo(({
    data,
    showPercentage,
    visibleLines,
    hasIntrinsic,
    timeRange
}: {
    data: ChartDataPoint[];
    showPercentage: boolean;
    visibleLines: { price: boolean; prediction: boolean; spy: boolean; intrinsic: boolean };
    hasIntrinsic: boolean;
    timeRange: string;
}) => {
    const yDataKey = showPercentage ? (d: ChartDataPoint) => d.closePct : (d: ChartDataPoint) => d.close;
    const predDataKey = showPercentage ? (d: ChartDataPoint) => d.predictionPct : (d: ChartDataPoint) => d.prediction;
    const spyDataKey = showPercentage ? (d: ChartDataPoint) => d.spyPct : (d: ChartDataPoint) => d.spyClose;
    const intrinsicDataKey = showPercentage ? (d: ChartDataPoint) => d.intrinsicPct : (d: ChartDataPoint) => d.intrinsic;

    if (!data || data.length === 0) return null;

    return (
        <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis
                    dataKey="date"
                    stroke="#52525b"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    minTickGap={50}
                    tickFormatter={(value, index) => {
                        const rangeDef = TIME_RANGES.find(r => r.label === timeRange);
                        const isLongRange = rangeDef && rangeDef.days > 365;
                        const isMediumRange = rangeDef && rangeDef.days > 90;

                        if (isLongRange) {
                            const d = data[index];
                            if (d && index % 12 === 0) return d.fullDate.getFullYear().toString();
                            return "";
                        }
                        if (isMediumRange) {
                            const d = data[index];
                            if (d && index % 5 === 0) return `${d.fullDate.getMonth() + 1}/${d.fullDate.getFullYear().toString().slice(2)}`;
                            return "";
                        }
                        return value;
                    }}
                />
                <YAxis
                    stroke="#52525b"
                    fontSize={10}
                    domain={['auto', 'auto']}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(val) => showPercentage ? `${val.toFixed(0)}%` : `$${val.toFixed(0)}`}
                    width={50}
                />
                <Tooltip content={<CustomTooltip showPercentage={showPercentage} />} />

                {showPercentage && <ReferenceLine y={0} stroke="#52525b" strokeDasharray="3 3" />}

                {/* S&P 500 comparison */}
                {visibleLines.spy && (
                    <Line
                        type="monotone"
                        dataKey={spyDataKey}
                        stroke="#3b82f6"
                        strokeWidth={1.5}
                        dot={false}
                        strokeDasharray="4 4"
                        name="S&P 500"
                    />
                )}

                {/* Intrinsic Value */}
                {hasIntrinsic && visibleLines.intrinsic && (
                    <Line
                        type="monotone"
                        dataKey={intrinsicDataKey}
                        stroke="#a78bfa"
                        strokeWidth={2}
                        dot={false}
                        name="Intrinsic"
                    />
                )}

                {/* AI Prediction */}
                {visibleLines.prediction && (
                    <Line
                        type="monotone"
                        dataKey={predDataKey}
                        stroke="#f472b6"
                        strokeWidth={2}
                        strokeDasharray="6 3"
                        dot={false}
                        name="AI Prediction"
                    />
                )}

                {/* Actual Price */}
                {visibleLines.price && (
                    <Area
                        type="monotone"
                        dataKey={yDataKey}
                        stroke="#10b981"
                        strokeWidth={2}
                        fill="url(#priceGradient)"
                        name="Price"
                    />
                )}

                <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                </defs>

                <Brush
                    dataKey="date"
                    height={30}
                    stroke="#52525b"
                    fill="#18181b"
                    tickFormatter={() => ""}
                />
            </ComposedChart>
        </ResponsiveContainer>
    );
});

DetailChart.displayName = "DetailChart";
