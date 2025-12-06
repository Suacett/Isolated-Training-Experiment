"""
Database Migration: Add is_favorite column to watchlist table.

Run this once to update existing database schema.
"""

import asyncio
import asyncpg
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@timescaledb:5432/stock_db")

# Convert to asyncpg format
DB_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

async def migrate():
    print("🔄 Starting database migration...")
    
    conn = await asyncpg.connect(DB_URL)
    
    try:
        # Check if column exists
        result = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'watchlist' AND column_name = 'is_favorite'
            )
        """)
        
        if result:
            print("✅ Column 'is_favorite' already exists. Skipping migration.")
        else:
            print("📝 Adding 'is_favorite' column to watchlist table...")
            await conn.execute("""
                ALTER TABLE watchlist 
                ADD COLUMN is_favorite BOOLEAN DEFAULT TRUE
            """)
            print("✅ Added 'is_favorite' column.")
        
        # Check for added_at column
        result = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'watchlist' AND column_name = 'added_at'
            )
        """)
        
        if result:
            print("✅ Column 'added_at' already exists. Skipping.")
        else:
            print("📝 Adding 'added_at' column to watchlist table...")
            await conn.execute("""
                ALTER TABLE watchlist 
                ADD COLUMN added_at TIMESTAMP DEFAULT NOW()
            """)
            print("✅ Added 'added_at' column.")
        
        # Set all existing items as favorites
        updated = await conn.execute("""
            UPDATE watchlist SET is_favorite = TRUE WHERE is_favorite IS NULL
        """)
        print(f"✅ Set existing items as favorites: {updated}")
        
        print("🎉 Migration complete!")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
