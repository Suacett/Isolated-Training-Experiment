"use client";

import { useState, useEffect, createContext, useContext, useCallback, useRef } from "react";
import { CheckCircle, XCircle, AlertCircle, Info, X } from "lucide-react";

type ToastType = "success" | "error" | "warning" | "info";

interface Toast {
    id: string;
    message: string;
    type: ToastType;
}

interface ToastContextType {
    showToast: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function useToast() {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error("useToast must be used within a ToastProvider");
    }
    return context;
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    // Use a persistent counter for guaranteed unique IDs
    const idCounterRef = useRef(0);

    const showToast = useCallback((message: string, type: ToastType = "info") => {
        idCounterRef.current += 1;
        const id = `toast-${Date.now()}-${idCounterRef.current}-${Math.random().toString(36).substr(2, 9)}`;
        setToasts((prev) => [...prev, { id, message, type }]);
    }, []);

    const removeToast = useCallback((id: string) => {
        setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, []);

    return (
        <ToastContext.Provider value={{ showToast }}>
            {children}
            <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
                {toasts.map((toast) => (
                    <ToastItem
                        key={toast.id}
                        toast={toast}
                        onRemove={() => removeToast(toast.id)}
                    />
                ))}
            </div>
        </ToastContext.Provider>
    );
}

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: () => void }) {
    useEffect(() => {
        const timer = setTimeout(onRemove, 4000);
        return () => clearTimeout(timer);
    }, [onRemove]);

    const icons = {
        success: <CheckCircle className="text-emerald-400" size={18} />,
        error: <XCircle className="text-rose-400" size={18} />,
        warning: <AlertCircle className="text-amber-400" size={18} />,
        info: <Info className="text-blue-400" size={18} />,
    };

    const bgColors = {
        success: "bg-emerald-500/10 border-emerald-500/30",
        error: "bg-rose-500/10 border-rose-500/30",
        warning: "bg-amber-500/10 border-amber-500/30",
        info: "bg-blue-500/10 border-blue-500/30",
    };

    return (
        <div
            className={`pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-lg border backdrop-blur-lg shadow-xl animate-slide-in ${bgColors[toast.type]}`}
        >
            {icons[toast.type]}
            <span className="text-sm text-zinc-200">{toast.message}</span>
            <button
                onClick={onRemove}
                className="ml-2 p-1 hover:bg-white/10 rounded-full transition-colors"
            >
                <X size={14} className="text-zinc-400" />
            </button>
        </div>
    );
}
