import { Info, TrendingUp, TrendingDown } from "lucide-react";

type BaseForecastCardProps = {
    label: string;
    price: number | null;
    changePct: number | null;
    infoText: string;
    showPercentage?: boolean;
    referencePrice?: number;
};

type ResultForecastCardProps = BaseForecastCardProps & {
    isResult: true;
    wasCorrect: boolean;
};

type RegularForecastCardProps = BaseForecastCardProps & {
    isResult?: false;
    wasCorrect?: never;
};

type ForecastCardProps = ResultForecastCardProps | RegularForecastCardProps;

export function ForecastCard(props: ForecastCardProps) {
    const {
        label,
        price,
        changePct,
        infoText,
        isResult,
        wasCorrect,
        showPercentage,
        referencePrice
    } = props;
    const isPositive = (changePct ?? 0) >= 0;

    let displayValue = "N/A";
    if (price !== null) {
        if (showPercentage && referencePrice !== null && referencePrice !== undefined) {
            if (referencePrice === 0) {
                displayValue = "0.00%";
            } else {
                const relPct = ((price - referencePrice) / referencePrice) * 100;
                displayValue = `${relPct > 0 ? '+' : ''}${relPct.toFixed(2)}%`;
            }
        } else {
            displayValue = `$${price.toFixed(2)}`;
        }
    }

    return (
        <div className={`flex-shrink-0 w-28 bg-zinc-800/50 rounded-lg p-3 border ${isResult
            ? (wasCorrect ? 'border-emerald-500/50' : 'border-rose-500/50')
            : 'border-zinc-700/50'
            }`}>
            <div className="flex items-center gap-1 mb-1">
                <span className="text-[10px] text-zinc-500 uppercase font-medium">{label}</span>
                <div className="relative inline-block group">
                    <Info size={10} className="text-zinc-600 cursor-help" />
                    <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-48 bg-zinc-900 border border-zinc-700 p-2 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[500] pointer-events-none">
                        <span className="text-[10px] text-zinc-300">{infoText}</span>
                    </div>
                </div>
            </div>
            {price !== null ? (
                <>
                    <div className="text-sm font-bold text-white">{displayValue}</div>
                    <div className={`text-xs font-medium flex items-center gap-0.5 ${changePct !== null ? (changePct >= 0 ? 'text-emerald-400' : 'text-rose-400') : 'text-zinc-500'}`}>
                        {changePct !== null ? (
                            <>
                                {changePct >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                                {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                            </>
                        ) : (
                            "—"
                        )}
                    </div>
                    {isResult && (
                        <div className={`mt-1 text-[10px] font-bold ${wasCorrect ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {wasCorrect ? '✓ CORRECT' : '✗ WRONG'}
                        </div>
                    )}
                </>
            ) : (
                <div className="text-sm text-zinc-500">N/A</div>
            )}
        </div>
    );
}
