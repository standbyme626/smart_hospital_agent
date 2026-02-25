
import asyncio
from sqlalchemy import text
from app.db.session import engine
from app.db.base_class import Base
from app.db.models.medical import Consultation, Prescription  # Import to register models

async def init_db():
    print("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully.")

if __name__ == "__main__":
    asyncio.run(init_db())
