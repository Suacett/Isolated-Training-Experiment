"""
Stocks Router - For managing watchlist and stock data.

Uses Yahoo Finance for ticker validation and data ingestion.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from pydantic import BaseModel
import logging

from services.db import (
    get_watchlist, 
    add_watchlist_item, 
    remove_watchlist_item, 
    delete_stock_data,
    get_favorites,
    set_favorite,
    get_watchlist_with_favorites,
    get_unique_tickers
)
from services.data_ingest import YahooFinanceClient, sync_ingest_hybrid_data

router = APIRouter(
    tags=["stocks"]
)

logger = logging.getLogger(__name__)

# Popular stocks for Browse Stocks feature (grouped by category)
BROWSE_STOCKS = {
    "Popular": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B"],
    "Tech": ["AMD", "INTC", "CRM", "ADBE", "ORCL", "CSCO", "IBM", "QCOM"],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP"],
    "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    "Defense": ["LMT", "RTX", "BA", "NOC", "GD", "HII", "LHX"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "Consumer": ["WMT", "HD", "COST", "TGT", "LOW", "SBUX", "MCD", "NKE"],
    "ETFs": ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "ARKK", "XLF"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD"],
}


class StockCreate(BaseModel):
    ticker: str
    is_favorite: bool = False  # New stocks don't get Alpha Vantage calls by default


class FavoriteUpdate(BaseModel):
    is_favorite: bool


@router.get("/", response_model=List[str])
async def list_stocks():
    """
    List all tickers in the watchlist.
    """
    return await get_watchlist()


@router.get("/with-favorites")
async def list_stocks_with_favorites():
    """
    List all tickers with their favorite status.
    """
    return await get_watchlist_with_favorites()


@router.get("/tickers")
async def list_tickers():
    """
    List all unique tickers found in the historical data.
    """
    tickers = await get_unique_tickers()
    return {"tickers": tickers}


@router.get("/favorites", response_model=List[str])
async def list_favorites():
    """
    List only favorite tickers (these get Alpha Vantage intrinsic value calls).
    """
    return await get_favorites()


@router.get("/browse")
async def browse_stocks():
    """
    Get categorized list of popular stocks for browsing.
    Returns stocks grouped by category with their availability status.
    """
    watchlist = await get_watchlist()
    result = {}
    
    for category, tickers in BROWSE_STOCKS.items():
        result[category] = [
            {"ticker": ticker, "in_watchlist": ticker in watchlist}
            for ticker in tickers
        ]
    
    return result


@router.post("/")
async def add_stock(stock: StockCreate, background_tasks: BackgroundTasks):
    """
    Add a new ticker to the watchlist.
    Validates ticker existence via Yahoo Finance.
    Triggers hybrid data ingestion (OHLCV + Sentiment).
    """
    ticker = stock.ticker.upper()
    

    # 1. Validate Ticker via Yahoo Finance
    try:
        yf_client = YahooFinanceClient()
        if not yf_client.verify_ticker(ticker):
            raise HTTPException(status_code=400, detail=f"Invalid ticker: {ticker}")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Ticker validation warning for {ticker}: {e}")
        # Continue anyway - Yahoo Finance sometimes fails on valid tickers

    # 2. Add to DB with favorite status
    try:
        await add_watchlist_item(ticker, is_favorite=stock.is_favorite)
    except Exception as e:
        logger.error(f"Failed to add {ticker} to DB: {e}")
        raise HTTPException(status_code=500, detail="Failed to add ticker to watchlist.")

    # 3. Trigger Hybrid Background Ingestion (OHLCV + Sentiment)
    def run_hybrid_ingest():
        try:
            result = sync_ingest_hybrid_data(ticker)
            logger.info(f"✅ Hybrid ingestion completed for {ticker}: {result}")
        except Exception as e:
            logger.error(f"❌ Hybrid ingestion failed for {ticker}: {e}")
    
    background_tasks.add_task(run_hybrid_ingest)
    
    return {
        "message": f"Added {ticker} to watchlist.", 
        "source": "hybrid",
        "is_favorite": stock.is_favorite
    }


@router.patch("/{ticker}/favorite")
async def update_favorite(ticker: str, update: FavoriteUpdate):
    """
    Toggle favorite status for a ticker.
    Favorites get Alpha Vantage intrinsic value calculations.
    """
    ticker = ticker.upper()
    
    success = await set_favorite(ticker, update.is_favorite)
    if not success:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in watchlist")
    
    return {
        "ticker": ticker, 
        "is_favorite": update.is_favorite,
        "message": f"{'Added to' if update.is_favorite else 'Removed from'} favorites"
    }


@router.delete("/{ticker}")
async def delete_stock(ticker: str, cascade: bool = False):
    """
    Remove a ticker from the watchlist.
    If cascade=True, also deletes all associated data (prices, predictions).
    """
    ticker = ticker.upper()
    
    try:
        await remove_watchlist_item(ticker)
        
        if cascade:
            await delete_stock_data(ticker)
            return {"message": f"Removed {ticker} and deleted all associated data."}
            
        return {"message": f"Removed {ticker} from watchlist."}
        
    except Exception as e:
        logger.error(f"Failed to delete {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete {ticker}")
