import { Clock, XCircle, CheckCircle } from "lucide-react";

export function ModelEvolutionHistory() {
    return (
        <div className="mt-8 p-6 bg-zinc-900/50 rounded-2xl border border-zinc-800">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Clock size={20} className="text-amber-400" />
                Model Evolution History
            </h3>
            <p className="text-sm text-zinc-500 mb-6">
                The journey from failure to success. Each model taught us something.
            </p>

            <div className="relative">
                {/* Timeline line */}
                <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-zinc-800" />

                {/* V7 */}
                <div className="relative pl-12 pb-6">
                    <div className="absolute left-2.5 w-3 h-3 rounded-full bg-amber-500/50 border-2 border-amber-500" />
                    <div className="flex items-start gap-3">
                        <XCircle size={18} className="text-amber-500 mt-0.5 flex-shrink-0" />
                        <div>
                            <h4 className="font-bold text-amber-400">V7 LSTM Regression</h4>
                            <p className="text-xs text-zinc-500 mt-1">
                                Trained to predict exact prices. Learned to output <code className="bg-zinc-800 px-1 rounded">tomorrow = today</code> to minimize MSE.
                                Zero directional accuracy. <span className="text-rose-400">Failed.</span>
                            </p>
                        </div>
                    </div>
                </div>

                {/* V8 */}
                <div className="relative pl-12 pb-6">
                    <div className="absolute left-2.5 w-3 h-3 rounded-full bg-rose-500/50 border-2 border-rose-500" />
                    <div className="flex items-start gap-3">
                        <XCircle size={18} className="text-rose-500 mt-0.5 flex-shrink-0" />
                        <div>
                            <h4 className="font-bold text-rose-400">V8 Bi-LSTM</h4>
                            <p className="text-xs text-zinc-500 mt-1">
                                Switched to binary classification. Learned to detect volatility spikes.
                                But couldn't tell "panic volatility" from "opportunity volatility". Bought falling knives.
                                <span className="text-rose-400"> Failed.</span>
                            </p>
                        </div>
                    </div>
                </div>

                {/* V9 */}
                <div className="relative pl-12">
                    <div className="absolute left-2.5 w-3 h-3 rounded-full bg-emerald-500/50 border-2 border-emerald-500 animate-pulse" />
                    <div className="flex items-start gap-3">
                        <CheckCircle size={18} className="text-emerald-500 mt-0.5 flex-shrink-0" />
                        <div>
                            <h4 className="font-bold text-emerald-400">V9 Transformer</h4>
                            <p className="text-xs text-zinc-500 mt-1">
                                Transformer architecture. Ranks stocks by <strong className="text-white">Relative Strength vs SPY</strong>.
                                Holds Top 10 with low correlation. Diversifies risk.
                                <span className="text-emerald-400"> Current Production Model.</span>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
