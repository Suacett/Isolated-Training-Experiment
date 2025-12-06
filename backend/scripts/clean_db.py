
import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_duplicates():
    conn = psycopg2.connect('postgresql://postgres:password@db:5432/stock_predictor')
    conn.autocommit = True
    cur = conn.cursor()
    
    logger.info("Identifying duplicates...")
    
    # CTE to identify duplicates: same ticker, same day, different timestamp
    # We want to keep the one with the latest timestamp (likely market close)
    
    query = """
    DELETE FROM stock_prices a USING (
        SELECT ticker, timestamp
        FROM (
            SELECT ticker, timestamp,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, date(timestamp) 
                       ORDER BY timestamp DESC
                   ) as r_num
            FROM stock_prices
        ) t
        WHERE t.r_num > 1
    ) b
    WHERE a.ticker = b.ticker AND a.timestamp = b.timestamp;
    """
    
    cur.execute("SELECT count(*) FROM stock_prices")
    before = cur.fetchone()[0]
    logger.info(f"Total records before: {before}")
    
    logger.info("Deleting duplicates (keeping latest per day)...")
    cur.execute(query)
    deleted = cur.rowcount
    logger.info(f"Deleted {deleted} duplicate records")
    
    cur.execute("SELECT count(*) FROM stock_prices")
    after = cur.fetchone()[0]
    logger.info(f"Total records after: {after}")
    
    conn.close()

if __name__ == "__main__":
    clean_duplicates()
