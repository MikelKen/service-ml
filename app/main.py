from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import strawberry
from strawberry.fastapi import GraphQLRouter
import uvicorn
import logging

from app.config.settings import settings
from app.graphql.schema import schema
from app.config.connection import init_database, close_database
from app.services.sync_service import auto_sync_service

# Importar routers disponibles de forma segura
from app.routers import health
try:
    from app.routers import sync  # router de sincronización
except Exception:
    sync = None
    logging.getLogger(__name__).warning("Router opcional 'sync' no encontrado. Se omitirá su registro.")
try:
    from app.routers import ml_database  # opcional
except Exception:
    ml_database = None
    logging.getLogger(__name__).warning("Router opcional 'ml_database' no encontrado. Se omitirá su registro.")
try:
    from app.routers import clustering  # opcional
except Exception:  # ImportError u otros
    clustering = None
    logging.getLogger(__name__).warning("Router opcional 'clustering' no encontrado. Se omitirá su registro.")

try:
    from app.routers import database  # opcional
except Exception:
    database = None
    logging.getLogger(__name__).warning("Router opcional 'database' no encontrado. Se omitirá su registro.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML Hiring Service API",
    description="Machine Learning microservice for hiring prediction and candidate clustering analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create GraphQL schema - using the centralized schema
graphql_app = GraphQLRouter(schema)

# Add GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")

# Add health check routes
app.include_router(health.router, prefix="/api")

# Add sync routes
if sync is not None:
    app.include_router(sync.router, prefix="/api")

# Add clustering routes (si el módulo existe)
if clustering is not None:
    app.include_router(clustering.router, prefix="/api")

# Add database query routes (si el módulo existe)
if database is not None:
    app.include_router(database.router, prefix="/api/db")

# Add ML database routes (si el módulo existe)
if ml_database is not None:
    app.include_router(ml_database.router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting ML Hiring Service...")
    
    # Initialize database connection
    db_connected = await init_database()
    if db_connected:
        logger.info("✅ Database connection established successfully!")
        
        # Iniciar servicio de sincronización automática
        await auto_sync_service.start()
    else:
        logger.error("❌ Failed to connect to database!")
    
    logger.info("🎯 ML Hiring Service started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("🛑 Shutting down ML Hiring Service...")
    
    # Detener servicio de sincronización
    await auto_sync_service.stop()
    
    await close_database()
    logger.info("👋 ML Hiring Service shutdown complete!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ML Hiring Service",
        "status": "running",
        "version": "1.0.0",
        "description": "Machine Learning microservice for hiring prediction and analytics",
        "endpoints": {
            "graphql": "/graphql",
            "graphql_playground": "/graphql",
            "health": "/api/health",
            "sync_status": "/api/sync/status",
            "sync_force": "/api/sync/force-sync",
            "sync_start": "/api/sync/start",
            "sync_stop": "/api/sync/stop",
            "clustering": "/api/clustering",
            "database_queries": "/api/db",
            "ml_database": "/api/ml/database",
            "ml_training_data": "/api/ml/database/training-data",
            "ml_train_model": "/api/ml/database/train-model",
            "ml_predictions": "/api/ml/database/predict-applications",
            "database_status": "/api/db/db-status",
            "empresas": "/api/db/empresas",
            "ofertas": "/api/db/ofertas",
            "postulaciones": "/api/db/postulaciones",
            "entrevistas": "/api/db/entrevistas",
            "evaluaciones": "/api/db/evaluaciones",
            "estadisticas": "/api/db/estadisticas",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with database status"""
    from app.config.connection import db, mongodb
    
    postgres_status = await db.test_connection()
    mongodb_status = await mongodb.test_connection()
    
    return {
        "status": "healthy" if (postgres_status and mongodb_status) else "unhealthy",
        "service": "ml-hiring-service",
        "version": "1.0.0",
        "databases": {
            "postgresql": {
                "status": "connected" if postgres_status else "disconnected",
                "type": "PostgreSQL"
            },
            "mongodb": {
                "status": "connected" if mongodb_status else "disconnected",
                "type": "MongoDB"
            }
        }
    }


@app.get("/info")
async def get_service_info():
    """Service information endpoint"""
    return {
        "name": "ML Hiring Service",
        "description": "Machine Learning microservice for hiring prediction",
        "version": "1.0.0",
        "features": [
            "Hiring probability prediction",
            "Candidate evaluation",
            "Candidate clustering by profile similarity",
            "Similar candidate search",
            "ML analytics",
            "GraphQL API",
            "REST API"
        ],
        "technologies": [
            "FastAPI",
            "Strawberry GraphQL",
            "scikit-learn",
            "pandas",
            "numpy"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )