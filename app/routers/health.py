from fastapi import APIRouter
from app.config.settings import settings
from app.services.product_service import product_service

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    health_data = product_service.get_health_status()
    return health_data


@router.get("/")
async def service_info():
    """Service information endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "description": settings.description,
        "status": "running",
        "endpoints": {
            "graphql": settings.graphql_path,
            "health": "/health",
            "products_rest": f"{settings.api_prefix}/products"
        },
        "ml_models_loaded": product_service.ml_models_loaded,
        "total_products": len(product_service.products)
    }