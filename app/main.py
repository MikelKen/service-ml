from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.tools import merge_types
import uvicorn

from app.config.settings import settings
from app.graphql.queries import Query
from app.graphql.mutations import Mutation
from app.routers import products

# Create FastAPI app
app = FastAPI(
    title="ML Service API",
    description="Machine Learning microservice for product recommendations and analytics",
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

# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create GraphQL router
graphql_app = GraphQLRouter(schema)

# Add GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")

# Add REST API routes
app.include_router(products.router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ML Service",
        "status": "running",
        "version": "1.0.0",
        "description": "Machine Learning microservice for product recommendations and analytics",
        "endpoints": {
            "graphql": "/graphql",
            "graphql_playground": "/graphql",
            "rest_api": "/api",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ml-service",
        "version": "1.0.0"
    }


@app.get("/info")
async def get_service_info():
    """Service information endpoint"""
    return {
        "name": "ML Service",
        "description": "Machine Learning microservice",
        "version": "1.0.0",
        "features": [
            "Product recommendations",
            "Demand prediction",
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