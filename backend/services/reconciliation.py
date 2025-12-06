
import logging
import numpy as np
from datetime import datetime
from sqlalchemy import select, and_
from services.db import AsyncSessionLocal, Prediction, StockPrice

logger = logging.getLogger(__name__)

class ReconciliationService:
    """
    Service to grade past predictions against actual market data.
    """

    @staticmethod
    async def reconcile_predictions(ticker: str = None):
        """
        Check pending predictions and update them with actual results.
        
        Args:
            ticker: Optional ticker to filter by. If None, checks all pending.
        """
        logger.info(f"Starting reconciliation for {ticker if ticker else 'ALL tickers'}...")
        
        async with AsyncSessionLocal() as session:
            # 1. Find pending predictions (target_date passed, but no actual_value)
            now = datetime.now()
            query = select(Prediction).where(
                and_(
                    Prediction.target_date <= now,
                    Prediction.actual_value.is_(None)
                )
            )
            
            if ticker:
                query = query.where(Prediction.ticker == ticker)
                
            result = await session.execute(query)
            pending_preds = result.scalars().all()
            
            if not pending_preds:
                logger.info("No pending predictions to reconcile.")
                return

            logger.info(f"Found {len(pending_preds)} pending predictions to reconcile.")
            
            updated_count = 0
            
            for pred in pending_preds:
                # 2. Get actual price for the target date
                # We look for a price entry on the target date (or closest available?)
                # For strict accuracy, we want the exact date.
                
                # Note: StockPrice timestamp is typically market close (e.g. 16:00 or 00:00)
                # We need to match the date part.
                
                price_query = select(StockPrice).where(
                    and_(
                        StockPrice.ticker == pred.ticker,
                        # Cast to date to match regardless of time
                        # SQLAlchemy generic function or just range check
                        StockPrice.timestamp >= pred.target_date.replace(hour=0, minute=0, second=0),
                        StockPrice.timestamp < pred.target_date.replace(hour=23, minute=59, second=59)
                    )
                ).order_by(StockPrice.timestamp.desc()).limit(1)
                
                price_result = await session.execute(price_query)
                actual_price_row = price_result.scalar_one_or_none()
                
                if actual_price_row:
                    # 3. Grade the prediction
                    actual_price = actual_price_row.close
                    pred.actual_value = actual_price
                    
                    # Direction logic
                    # We need the price at prediction time to know the direction
                    # pred.current_price should be stored at prediction time
                    
                    if pred.current_price:
                        # Determine actual direction
                        actual_return = np.log(actual_price / pred.current_price)
                        actual_direction = actual_return > 0
                        
                        # Determine predicted direction (predicted_value is log return)
                        predicted_direction = pred.predicted_value > 0
                        
                        # Correct if directions match
                        pred.is_correct = (actual_direction == predicted_direction)
                        
                        # Calculate error (percentage difference in price)
                        predicted_price_val = pred.current_price * np.exp(pred.predicted_value)
                        error = abs((predicted_price_val - actual_price) / actual_price) * 100
                        
                        logger.info(f"Graded {pred.ticker} {pred.target_date.date()}: PredReturn={pred.predicted_value:.4f}, ActReturn={actual_return:.4f}, Correct={pred.is_correct}")
                        updated_count += 1
                    else:
                        logger.warning(f"Prediction {pred.id} missing current_price, cannot grade direction.")
            
            if updated_count > 0:
                await session.commit()
                logger.info(f"Successfully reconciled {updated_count} predictions.")
            else:
                logger.info("No predictions could be matched with actual prices.")
