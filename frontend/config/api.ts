/**
 * Centralized API configuration for frontend
 * Allows easy switching between local, staging, and production environments
 */

export const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 30000,
} as const;

/**
 * Constructs a full API URL from an endpoint path
 * @param endpoint - The API endpoint (with or without leading slash)
 * @returns Full URL to the API endpoint
 */
export const getApiUrl = (endpoint: string): string => {
  const base = API_CONFIG.baseURL.endsWith('/') ? API_CONFIG.baseURL.slice(0, -1) : API_CONFIG.baseURL;
  const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${base}${path}`;
};

/**
 * Alias for getApiUrl for backward compatibility
 */
export const apiUrl = getApiUrl;
