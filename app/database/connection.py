import asyncio
import asyncpg
import logging
from typing import Optional
from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager for PostgreSQL using asyncpg"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = settings.database_url
        
    async def connect(self) -> bool:
        """
        Establish connection to PostgreSQL database
        Returns True if connection successful, False otherwise
        """
        try:
            logger.info("Attempting to connect to PostgreSQL database...")
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            
            # Test connection
            async with self.pool.acquire() as connection:
                result = await connection.fetchval('SELECT version()')
                logger.info(f"✅ Database connected successfully!")
                logger.info(f"PostgreSQL version: {result}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            return False
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection closed")
    
    async def get_connection(self):
        """Get a connection from the pool"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.pool.acquire()
    
    async def test_connection(self) -> bool:
        """Test if database connection is working"""
        try:
            if not self.pool:
                return False
                
            async with self.pool.acquire() as connection:
                await connection.fetchval('SELECT 1')
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False

# Global database instance
db = DatabaseConnection()

async def init_database():
    """Initialize database connection"""
    return await db.connect()

async def close_database():
    """Close database connection"""
    await db.disconnect()

async def get_database():
    """Get database connection (dependency injection)"""
    return db