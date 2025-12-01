import asyncio
from services.db import engine, Base

async def init_db():
    async with engine.begin() as conn:
        # Create all tables defined in Base (including Watchlist)
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created successfully.")

if __name__ == "__main__":
    asyncio.run(init_db())
