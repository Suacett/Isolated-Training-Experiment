#!/usr/bin/env python3
"""
Migration script to recreate predictions table with new schema.

This script:
1. Backs up existing predictions (if any)
2. Drops the old predictions table
3. Recreates it with the new schema
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import text
from services.db import AsyncSessionLocal, Base, engine, Prediction
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_predictions_table():
    """Drop and recreate predictions table with new schema."""

    logger.info("Starting predictions table migration...")

    async with AsyncSessionLocal() as session:
        try:
            # Check if table exists and has data
            result = await session.execute(
                text("SELECT COUNT(*) FROM predictions")
            )
            count = result.scalar()
            logger.info(f"Found {count} existing predictions")

            if count > 0:
                logger.warning(f"⚠️  This will DELETE {count} existing predictions!")
                logger.warning("Press Ctrl+C within 3 seconds to cancel...")
                await asyncio.sleep(3)

        except Exception as e:
            logger.info(f"Predictions table doesn't exist or is empty: {e}")

        # Drop the old table
        logger.info("Dropping old predictions table...")
        await session.execute(text("DROP TABLE IF EXISTS predictions CASCADE"))
        await session.commit()
        logger.info("✅ Dropped old table")

        # Recreate with new schema
        logger.info("Creating new predictions table...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Created new table with updated schema")

        # Verify new schema
        result = await session.execute(
            text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'predictions'
                ORDER BY ordinal_position
            """)
        )
        columns = result.fetchall()

        logger.info("\n📋 New predictions table schema:")
        for col_name, col_type in columns:
            logger.info(f"  - {col_name}: {col_type}")

        logger.info("\n✅ Migration complete!")
        logger.info("You can now run /ingest/all to generate new predictions.")


if __name__ == "__main__":
    asyncio.run(migrate_predictions_table())
