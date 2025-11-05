from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class MongoDBConnection:
    """MongoDB connection manager"""
    
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.database = None
        self.sync_client: MongoClient = None
        self.sync_database = None
    
    async def connect(self):
        """Create async connection to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            self.database = self.client[settings.mongodb_database]
            
            # Test the connection
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB (async)")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB (async): {e}")
            raise
    
    def connect_sync(self):
        """Create sync connection to MongoDB"""
        try:
            self.sync_client = MongoClient(settings.mongodb_url)
            self.sync_database = self.sync_client[settings.mongodb_database]
            
            # Test the connection
            self.sync_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB (sync)")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB (sync): {e}")
            raise
    
    async def disconnect(self):
        """Close async connection"""
        if self.client is not None:
            self.client.close()
            logger.info("Disconnected from MongoDB (async)")
    
    def disconnect_sync(self):
        """Close sync connection"""
        if self.sync_client is not None:
            self.sync_client.close()
            logger.info("Disconnected from MongoDB (sync)")
    
    def get_collection(self, collection_name: str):
        """Get async collection"""
        if self.database is None:
            raise Exception("Database not connected. Call connect() first.")
        return self.database[collection_name]
    
    def get_collection_sync(self, collection_name: str):
        """Get sync collection"""
        if self.sync_database is None:
            raise Exception("Database not connected. Call connect_sync() first.")
        return self.sync_database[collection_name]

# Global instance
mongodb_connection = MongoDBConnection()

# Convenience functions
async def get_mongodb():
    """Get MongoDB async connection"""
    if mongodb_connection.database is None:
        await mongodb_connection.connect()
    return mongodb_connection.database

def get_mongodb_sync():
    """Get MongoDB sync connection"""
    if mongodb_connection.sync_database is None:
        mongodb_connection.connect_sync()
    return mongodb_connection.sync_database

async def get_collection(collection_name: str):
    """Get async collection by name"""
    db = await get_mongodb()
    return db[collection_name]

def get_collection_sync(collection_name: str):
    """Get sync collection by name"""
    db = get_mongodb_sync()
    return db[collection_name]