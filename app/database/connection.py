import asyncio
import asyncpg
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager for PostgreSQL using asyncpg"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = settings.database_url
        logger.info(f"Database URL from settings: {self.database_url}")
        
        # Fallback to environment variable if settings.database_url is None
        if not self.database_url:
            import os
            self.database_url = os.getenv("DB_URL_POSTGRES")
            logger.info(f"Using fallback DB_URL_POSTGRES: {self.database_url}")
        
        if not self.database_url:
            logger.error("No database URL found! Check .env file or DB_URL_POSTGRES environment variable")
        
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

class MongoDBConnection:
    """Database connection manager for MongoDB using Motor"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.mongodb_url = settings.mongodb_url
        self.database_name = settings.mongodb_database
        logger.info(f"MongoDB URL configured: {self.mongodb_url[:50]}...")
        
    async def connect(self) -> bool:
        """
        Establish connection to MongoDB database
        Returns True if connection successful, False otherwise
        """
        try:
            logger.info("Attempting to connect to MongoDB database...")
            
            # Create MongoDB client
            self.client = AsyncIOMotorClient(self.mongodb_url)
            self.database = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Get server info
            server_info = await self.client.server_info()
            logger.info(f"✅ MongoDB connected successfully!")
            logger.info(f"MongoDB version: {server_info.get('version', 'Unknown')}")
            logger.info(f"Database: {self.database_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            return False
    
    async def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_database(self):
        """Get MongoDB database instance"""
        if not self.database:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self.database
    
    def get_collection(self, collection_name: str):
        """Get a specific collection from MongoDB"""
        return self.get_database()[collection_name]
    
    async def test_connection(self) -> bool:
        """Test if MongoDB connection is working"""
        try:
            if not self.client:
                return False
                
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {str(e)}")
            return False

# Global MongoDB instance
mongodb = MongoDBConnection()

async def init_database():
    """Initialize database connections"""
    postgres_success = await db.connect()
    mongodb_success = await mongodb.connect()
    return postgres_success and mongodb_success

async def close_database():
    """Close database connections"""
    await db.disconnect()
    await mongodb.disconnect()

async def get_database():
    """Get PostgreSQL database connection (dependency injection)"""
    return db

async def get_mongodb():
    """Get MongoDB database (dependency injection)"""
    return mongodb