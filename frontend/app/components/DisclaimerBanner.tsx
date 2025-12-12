'use client';
import { useState } from 'react';
import { X, AlertTriangle, Info, AlertCircle } from 'lucide-react';

// ====== EDIT THIS TEXT EASILY ======
const BANNER_CONFIG = {
  enabled: true,
  message: "🚧 System Processing Historical Data - UI may be slow",
  type: "warning" as "info" | "warning" | "error"
};
// ===================================

const BANNER_STYLES = {
  warning: {
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    text: "text-amber-400",
    icon: AlertTriangle
  },
  error: {
    bg: "bg-red-500/10",
    border: "border-red-500/30",
    text: "text-red-400",
    icon: AlertCircle
  },
  info: {
    bg: "bg-blue-500/10",
    border: "border-blue-500/30",
    text: "text-blue-400",
    icon: Info
  }
};

export default function DisclaimerBanner() {
  const [dismissed, setDismissed] = useState(false);

  if (!BANNER_CONFIG.enabled || dismissed) return null;

  const style = BANNER_STYLES[BANNER_CONFIG.type];
  const Icon = style.icon;

  return (
    <div className={`${style.bg} border-b ${style.border} px-4 py-2 flex items-center justify-between animate-fade-in`}>
      <div className={`flex items-center gap-2 ${style.text} text-sm`}>
        <Icon size={16} className="flex-shrink-0" />
        <span>{BANNER_CONFIG.message}</span>
      </div>
      <button
        onClick={() => setDismissed(true)}
        className={`${style.text} hover:text-white transition-colors`}
        title="Dismiss"
      >
        <X size={16} />
      </button>
    </div>
  );
}
