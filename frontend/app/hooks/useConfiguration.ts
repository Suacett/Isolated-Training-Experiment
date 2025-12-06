import { useState } from "react";

export type DataSource = "alpaca" | "alpha_vantage";

interface ConfigurationState {
    dataSource: DataSource;
    alpacaApiKey: string;
    alpacaSecretKey: string;
    alphaVantageKey: string;
    status: "idle" | "loading" | "success" | "error";
    message: string;
}

export function useConfiguration(onSuccess?: () => void) {
    const [state, setState] = useState<ConfigurationState>({
        dataSource: "alpaca",
        alpacaApiKey: "",
        alpacaSecretKey: "",
        alphaVantageKey: "",
        status: "idle",
        message: "",
    });

    const setDataSource = (dataSource: DataSource) => setState(prev => ({ ...prev, dataSource }));
    const setAlpacaApiKey = (key: string) => setState(prev => ({ ...prev, alpacaApiKey: key }));
    const setAlpacaSecretKey = (key: string) => setState(prev => ({ ...prev, alpacaSecretKey: key }));
    const setAlphaVantageKey = (key: string) => setState(prev => ({ ...prev, alphaVantageKey: key }));

    const saveConfiguration = async (e: React.FormEvent) => {
        e.preventDefault();

        // Validation
        if (state.dataSource === "alpaca") {
            if (!state.alpacaApiKey.trim() || !state.alpacaSecretKey.trim()) {
                setState(prev => ({ ...prev, status: "error", message: "Please fill in all Alpaca keys." }));
                return;
            }
        } else {
            if (!state.alphaVantageKey.trim()) {
                setState(prev => ({ ...prev, status: "error", message: "Please enter your Alpha Vantage key." }));
                return;
            }
        }

        setState(prev => ({ ...prev, status: "loading", message: "" }));

        try {
            const payload: Record<string, string | null> = {
                ALPACA_API_KEY: null,
                ALPACA_SECRET_KEY: null,
                ALPHA_VANTAGE_KEY: null,
            };

            if (state.dataSource === "alpaca") {
                payload.ALPACA_API_KEY = state.alpacaApiKey;
                payload.ALPACA_SECRET_KEY = state.alpacaSecretKey;
            } else {
                payload.ALPHA_VANTAGE_KEY = state.alphaVantageKey;
            }

            const res = await fetch("http://localhost:8000/settings/keys", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            if (res.ok) {
                setState(prev => ({ ...prev, status: "success", message: "Configuration saved successfully!" }));
                setTimeout(() => {
                    if (onSuccess) {
                        onSuccess();
                    } else {
                        window.location.reload();
                    }
                }, 1500); // Show success message for 1.5s before closing/reloading
            } else {
                setState(prev => ({ ...prev, status: "error", message: "Failed to save configuration." }));
            }
        } catch (error) {
            console.error(error);
            setState(prev => ({ ...prev, status: "error", message: "Network error occurred." }));
        }
    };

    return {
        ...state,
        setDataSource,
        setAlpacaApiKey,
        setAlpacaSecretKey,
        setAlphaVantageKey,
        saveConfiguration,
    };
}
