"""
Insider trading data collection from SEC EDGAR
Ported from legacy/LSTM_AI_Stock_Predictor/TrainingData/featuresPy/insiderbuying.py

This module fetches and parses SEC Form 4 filings to extract insider buying/selling activity.
"""

import os
import logging
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sec_edgar_downloader import Downloader
    SEC_EDGAR_AVAILABLE = True
except ImportError:
    SEC_EDGAR_AVAILABLE = False
    logger.warning("sec-edgar-downloader not installed. Insider trading features disabled.")


class InsiderDataCollector:
    """Collects insider trading data from SEC EDGAR Form 4 filings"""

    def __init__(self, cache_dir: str = "backend/cache/insider"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = "ProxmoxStockPredictor/1.0 (Educational Research)"

    def download_form4s(
        self,
        ticker: str,
        after: str,
        before: str = None
    ) -> Optional[Path]:
        """
        Download Form 4 filings for a ticker.

        Args:
            ticker: Stock ticker symbol
            after: Start date (YYYY-MM-DD)
            before: End date (YYYY-MM-DD), defaults to today

        Returns:
            Path to downloaded filings directory, or None if failed
        """
        if not SEC_EDGAR_AVAILABLE:
            logger.error("sec-edgar-downloader not installed")
            return None

        try:
            dl = Downloader(self.user_agent, str(self.cache_dir))
            if before:
                dl.get("4", ticker, after=after, before=before)
            else:
                dl.get("4", ticker, after=after)

            filings_dir = self.cache_dir / ticker.lower() / "4"
            return filings_dir if filings_dir.exists() else None
        except Exception as e:
            logger.error(f"Failed to download Form 4 for {ticker}: {e}")
            return None

    def extract_xml(self, filepath: Path) -> Optional[str]:
        """Extract XML content from Form 4 filing"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            start = content.find('<?xml')
            end = content.rfind('</ownershipDocument>') + len('</ownershipDocument>')

            if start != -1 and end != -1:
                return content[start:end]
            return None
        except Exception as e:
            logger.error(f"Failed to extract XML from {filepath}: {e}")
            return None

    def parse_form4(self, xml_str: str) -> List[Tuple[str, float, float, int]]:
        """
        Parse Form 4 XML to extract transactions.

        Returns:
            List of tuples: (date, shares, amount, buy_flag)
            buy_flag: 1 = buy (P), 0 = sell (S)
        """
        try:
            root = ET.fromstring(xml_str)
        except Exception as e:
            logger.error(f"XML parsing failed: {e}")
            return []

        # Handle namespace
        ns = {}
        if '}' in root.tag:
            ns = {'ns': root.tag.split('}')[0].strip('{')}

        transactions = []

        # Parse non-derivative transactions
        for txn in root.findall('.//nonDerivativeTransaction', ns):
            try:
                code_elem = txn.find('.//transactionCoding/transactionCode', ns)
                if code_elem is None:
                    continue

                code = code_elem.text.upper()
                if code not in ['P', 'S']:  # P = Purchase, S = Sale
                    continue

                buy_flag = 1 if code == 'P' else 0
                date = txn.find('.//transactionDate/value', ns).text
                shares = float(txn.find('.//transactionShares/value', ns).text)

                price_el = txn.find('.//transactionPricePerShare/value', ns)
                price = float(price_el.text) if price_el is not None else 0.0
                amount = shares * price

                transactions.append((date, shares, amount, buy_flag))
            except Exception as e:
                logger.debug(f"Error extracting transaction: {e}")
                continue

        return transactions

    def aggregate_by_day(self, transactions: List[Tuple]) -> List[Tuple]:
        """Aggregate transactions by date and buy/sell flag"""
        daily = {}

        for date, shares, amount, flag in transactions:
            key = (date, flag)
            if key not in daily:
                daily[key] = {"shares": 0.0, "amount": 0.0}
            daily[key]["shares"] += shares
            daily[key]["amount"] += amount

        # Sort by date
        return sorted([
            (d, s["shares"], s["amount"], b)
            for (d, b), s in daily.items()
        ])

    async def fetch_insider_trades(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch and process insider trades for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for fetching (default: 2 years ago)
            end_date: End date for fetching (default: today)

        Returns:
            DataFrame with columns: [date, shares, amount, buy_flag]
        """
        if not SEC_EDGAR_AVAILABLE:
            logger.warning(f"SEC EDGAR unavailable, returning empty insider data for {ticker}")
            return pd.DataFrame(columns=['date', 'shares', 'amount', 'buy_flag'])

        # Default to last 2 years
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()

        # Download Form 4 filings
        after_str = start_date.strftime("%Y-%m-%d")
        before_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching insider trades for {ticker} from {after_str} to {before_str}")

        # Run download in executor to avoid blocking
        loop = asyncio.get_event_loop()
        filings_dir = await loop.run_in_executor(
            None,
            self.download_form4s,
            ticker,
            after_str,
            before_str
        )

        if not filings_dir or not filings_dir.exists():
            logger.warning(f"No Form 4 filings found for {ticker}")
            return pd.DataFrame(columns=['date', 'shares', 'amount', 'buy_flag'])

        # Parse all filings
        all_transactions = []
        for filing in filings_dir.glob("*.txt"):
            xml_content = self.extract_xml(filing)
            if xml_content:
                transactions = self.parse_form4(xml_content)
                all_transactions.extend(transactions)

        if not all_transactions:
            logger.warning(f"No insider transactions found in filings for {ticker}")
            return pd.DataFrame(columns=['date', 'shares', 'amount', 'buy_flag'])

        # Aggregate by day
        aggregated = self.aggregate_by_day(all_transactions)

        # Create DataFrame
        df = pd.DataFrame(aggregated, columns=['date', 'shares', 'amount', 'buy_flag'])
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Found {len(df)} days of insider activity for {ticker}")
        return df

    async def get_daily_insider_summary(
        self,
        ticker: str,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get daily aggregated insider trading summary (net buying/selling per day).

        This is the format expected by the feature engineering pipeline.

        Returns:
            DataFrame with columns: [date, insider_shares, insider_amount, insider_buy_flag]
            insider_buy_flag: 1 if net buying, 0 if net selling, -1 if no activity
        """
        df = await self.fetch_insider_trades(ticker, start_date)

        if df.empty:
            return pd.DataFrame(columns=['date', 'insider_shares', 'insider_amount', 'insider_buy_flag'])

        # Group by date and compute net activity
        grouped = df.groupby('date').agg({
            'shares': lambda x: sum(x if f == 1 else -x for x, f in zip(x, df.loc[x.index, 'buy_flag'])),
            'amount': lambda x: sum(x if f == 1 else -x for x, f in zip(x, df.loc[x.index, 'buy_flag']))
        }).reset_index()

        # Determine net buy flag
        grouped['insider_buy_flag'] = grouped['shares'].apply(
            lambda s: 1 if s > 0 else (0 if s < 0 else -1)
        )

        # Rename columns
        grouped = grouped.rename(columns={
            'shares': 'insider_shares',
            'amount': 'insider_amount'
        })

        return grouped[['date', 'insider_shares', 'insider_amount', 'insider_buy_flag']]


# Global instance
insider_collector = InsiderDataCollector()
