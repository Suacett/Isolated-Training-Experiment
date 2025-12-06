#!/usr/bin/env python3
"""
Bulk S&P 500 Data Downloader

Downloads historical OHLCV data for all S&P 500 companies
using DIRECT Yahoo Finance API (bypasses yfinance curl_cffi issues).

Usage:
    docker exec proxmox_stock_backend python -m scripts.download_bulk_data
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "training_raw"

# Yahoo Finance Direct API
YAHOO_API = "https://query1.finance.yahoo.com/v8/finance/chart"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    print("📋 Fetching S&P 500 ticker list...")
    
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # Use requests with headers to avoid 403 Forbidden
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        from io import StringIO
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        if 'Symbol' in df.columns:
            tickers = df['Symbol'].tolist()
        else:
            tickers = df.iloc[:, 0].tolist()
        
        # Clean tickers (BRK.B -> BRK-B for Yahoo)
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"  ✅ Found {len(tickers)} S&P 500 tickers")
        return tickers
        
    except Exception as e:
        print(f"  ⚠️ Wikipedia failed: {e}, using fallback list")
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
            "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
            "ACN", "TMO", "ABT", "CRM", "DHR", "BAC", "NFLX", "AMD", "ADBE",
            "NKE", "DIS", "INTC", "VZ", "CMCSA", "TXN", "PM", "WFC", "RTX",
            "UNP", "NEE", "QCOM", "IBM", "HON"
        ]


def download_ticker_direct(ticker: str, start_date: str = "2010-01-01", max_retries: int = 2) -> pd.DataFrame | None:
    """
    Download historical data for a single ticker using DIRECT Yahoo API.
    Bypasses yfinance completely to avoid curl_cffi issues.
    """
    for attempt in range(max_retries):
        try:
            # Calculate timestamps
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int(datetime.now().timestamp())
            
            url = f"{YAHOO_API}/{ticker}"
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
                "events": "history"
            }
            
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
            
            data = response.json()
            result = data.get("chart", {}).get("result")
            
            if not result or len(result) == 0:
                return None
            
            result = result[0]
            timestamps = result.get("timestamp", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]
            
            if not timestamps:
                return None
            
            # Create DataFrame
            df = pd.DataFrame({
                "date": [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps],
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", [])
            })
            
            # Drop rows with NaN close prices
            df = df.dropna(subset=['close'])
            
            if df.empty:
                return None
                
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
    
    return None


def download_all_parallel(tickers: list[str], max_workers: int = 5) -> dict:
    """Download all tickers in parallel with rate limiting."""
    print(f"\n📥 Downloading {len(tickers)} tickers (parallel, max {max_workers} workers)...")
    
    results = {}
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all downloads
        future_to_ticker = {
            executor.submit(download_ticker_direct, ticker): ticker 
            for ticker in tickers
        }
        
        for i, future in enumerate(as_completed(future_to_ticker), 1):
            ticker = future_to_ticker[future]
            
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results[ticker] = df
                else:
                    failed.append(ticker)
            except Exception:
                failed.append(ticker)
            
            # Progress update every 50 tickers
            if i % 50 == 0 or i == len(tickers):
                print(f"  Progress: {i}/{len(tickers)} ({len(results)} success, {len(failed)} failed)")
            
            # Small delay to be polite
            time.sleep(0.2)
    
    return {"data": results, "failed": failed}


def save_to_csv(results: dict, output_dir: Path) -> int:
    """Save each ticker's data to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving data to {output_dir}...")
    
    saved_count = 0
    
    for ticker, df in results["data"].items():
        if df.empty:
            continue
            
        try:
            output_file = output_dir / f"{ticker}.csv"
            df.to_csv(output_file, index=False)
            saved_count += 1
        except Exception as e:
            print(f"  ⚠️ Failed to save {ticker}: {e}")
            continue
    
    return saved_count


def main():
    """Main entry point."""
    print("=" * 60)
    print("🚀 S&P 500 Bulk Data Downloader (Direct API)")
    print("=" * 60)
    
    # Step 1: Get tickers
    tickers = fetch_sp500_tickers()
    
    # Step 2: Download all data in parallel
    results = download_all_parallel(tickers, max_workers=5)
    
    if not results["data"]:
        print("\n❌ No data downloaded. Exiting.")
        sys.exit(1)
    
    # Step 3: Save to CSV files
    saved_count = save_to_csv(results, OUTPUT_DIR)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Successfully downloaded {saved_count} datasets to {OUTPUT_DIR}")
    print("=" * 60)
    
    if results["failed"]:
        print(f"⚠️ Failed: {len(results['failed'])} tickers")
    
    # Stats
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
    print(f"📊 Total files: {len(csv_files)}")
    print(f"📁 Total size: {total_size:.2f} MB")
    
    return saved_count


if __name__ == "__main__":
    main()
