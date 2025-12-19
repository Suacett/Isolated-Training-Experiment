"use client";

import { useState, useEffect } from "react";

/**
 * Hook to detect if a component has mounted on the client.
 * Useful for avoiding hydration mismatches when rendering
 * client-only content (like icons modified by extensions).
 */
export function useMounted() {
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    return mounted;
}
