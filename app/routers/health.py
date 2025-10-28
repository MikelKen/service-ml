from fastapi import APIRouter
from app.config.settings import settings
import os

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verificar si el modelo ML está disponible
        model_path = "trained_models/simple_hiring_model.pkl"
        model_loaded = os.path.exists(model_path)
        
        return {
            "status": "healthy",
            "service": "ml-hiring-service",
            "version": "1.0.0",
            "model_loaded": model_loaded,
            "message": "ML Hiring Service is running"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "ml-hiring-service",
            "version": "1.0.0",
            "model_loaded": False,
            "error": str(e)
        }


@router.get("/")
async def service_info():
    """Service information endpoint"""
    return {
        "service": "ML Hiring Service",
        "version": "1.0.0",
        "description": "Machine Learning microservice for hiring prediction",
        "status": "running",
        "endpoints": {
            "graphql": "/graphql",
            "health": "/api/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "features": [
            "Hiring probability prediction",
            "Candidate evaluation",
            "GraphQL API"
        ]
    }