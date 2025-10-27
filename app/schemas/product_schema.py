import strawberry
from typing import List, Optional
from datetime import datetime


@strawberry.type
class Product:
    id: str = strawberry.field(description="Unique product identifier")
    name: str = strawberry.field(description="Product name")
    price: float = strawberry.field(description="Product price")
    category: Optional[str] = strawberry.field(description="Product category")
    description: Optional[str] = strawberry.field(description="Product description")
    stock: int = strawberry.field(description="Available stock")
    active: bool = strawberry.field(description="Product availability status")
    tags: List[str] = strawberry.field(description="Product tags")
    ml_score: Optional[float] = strawberry.field(description="ML recommendation score")
    created_at: str = strawberry.field(description="Creation timestamp")
    updated_at: Optional[str] = strawberry.field(description="Last update timestamp")


@strawberry.input
class ProductInput:
    name: str = strawberry.field(description="Product name")
    price: float = strawberry.field(description="Product price")
    category: Optional[str] = strawberry.field(default=None, description="Product category")
    description: Optional[str] = strawberry.field(default=None, description="Product description")
    stock: int = strawberry.field(default=0, description="Available stock")
    active: bool = strawberry.field(default=True, description="Product availability status")
    tags: List[str] = strawberry.field(default_factory=list, description="Product tags")


@strawberry.input
class ProductUpdateInput:
    name: Optional[str] = strawberry.field(default=None, description="Product name")
    price: Optional[float] = strawberry.field(default=None, description="Product price")
    category: Optional[str] = strawberry.field(default=None, description="Product category")
    description: Optional[str] = strawberry.field(default=None, description="Product description")
    stock: Optional[int] = strawberry.field(default=None, description="Available stock")
    active: Optional[bool] = strawberry.field(default=None, description="Product availability status")
    tags: Optional[List[str]] = strawberry.field(default=None, description="Product tags")


@strawberry.type
class ProductRecommendation:
    product_id: str = strawberry.field(description="Product ID")
    score: float = strawberry.field(description="Recommendation score")
    reason: str = strawberry.field(description="Recommendation reason")
    user_id: Optional[str] = strawberry.field(description="Target user ID")


@strawberry.type
class MLPrediction:
    product_id: str = strawberry.field(description="Product ID")
    predicted_demand: float = strawberry.field(description="Predicted demand")
    confidence: float = strawberry.field(description="Prediction confidence")
    factors: List[str] = strawberry.field(description="Key factors influencing prediction")


@strawberry.type
class HealthStatus:
    status: str = strawberry.field(description="Service health status")
    service: str = strawberry.field(description="Service name")
    version: str = strawberry.field(description="Service version")
    timestamp: str = strawberry.field(description="Health check timestamp")
    ml_models_loaded: bool = strawberry.field(description="ML models status")