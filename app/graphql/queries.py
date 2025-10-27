import strawberry
from typing import List, Optional
from app.schemas.product_schema import Product, ProductRecommendation, MLPrediction, HealthStatus
from app.services.product_service import product_service


@strawberry.type
class Query:
    
    @strawberry.field(description="Get service health status")
    def health(self) -> HealthStatus:
        health_data = product_service.get_health_status()
        return HealthStatus(
            status=health_data["status"],
            service=health_data["service"],
            version=health_data["version"],
            timestamp=health_data["timestamp"],
            ml_models_loaded=health_data["ml_models_loaded"]
        )
    
    @strawberry.field(description="Get all products")
    def products(self) -> List[Product]:
        products_data = product_service.get_all_products()
        return [
            Product(
                id=p["id"],
                name=p["name"],
                price=p["price"],
                category=p.get("category"),
                description=p.get("description"),
                stock=p["stock"],
                active=p["active"],
                tags=p.get("tags", []),
                ml_score=p.get("ml_score"),
                created_at=p["created_at"],
                updated_at=p.get("updated_at")
            )
            for p in products_data
        ]
    
    @strawberry.field(description="Get product by ID")
    def product(self, id: str) -> Optional[Product]:
        product_data = product_service.get_product_by_id(id)
        if not product_data:
            return None
        
        return Product(
            id=product_data["id"],
            name=product_data["name"],
            price=product_data["price"],
            category=product_data.get("category"),
            description=product_data.get("description"),
            stock=product_data["stock"],
            active=product_data["active"],
            tags=product_data.get("tags", []),
            ml_score=product_data.get("ml_score"),
            created_at=product_data["created_at"],
            updated_at=product_data.get("updated_at")
        )
    
    @strawberry.field(description="Get products by category")
    def products_by_category(self, category: str) -> List[Product]:
        products_data = product_service.get_products_by_category(category)
        return [
            Product(
                id=p["id"],
                name=p["name"],
                price=p["price"],
                category=p.get("category"),
                description=p.get("description"),
                stock=p["stock"],
                active=p["active"],
                tags=p.get("tags", []),
                ml_score=p.get("ml_score"),
                created_at=p["created_at"],
                updated_at=p.get("updated_at")
            )
            for p in products_data
        ]
    
    @strawberry.field(description="Get only active products")
    def active_products(self) -> List[Product]:
        products_data = product_service.get_active_products()
        return [
            Product(
                id=p["id"],
                name=p["name"],
                price=p["price"],
                category=p.get("category"),
                description=p.get("description"),
                stock=p["stock"],
                active=p["active"],
                tags=p.get("tags", []),
                ml_score=p.get("ml_score"),
                created_at=p["created_at"],
                updated_at=p.get("updated_at")
            )
            for p in products_data
        ]
    
    @strawberry.field(description="Get ML-based product recommendations for a user")
    def product_recommendations(self, user_id: str, limit: int = 5) -> List[ProductRecommendation]:
        recommendations_data = product_service.get_product_recommendations(user_id, limit)
        return [
            ProductRecommendation(
                product_id=r["product_id"],
                score=r["score"],
                reason=r["reason"],
                user_id=r.get("user_id")
            )
            for r in recommendations_data
        ]
    
    @strawberry.field(description="Predict demand for a specific product")
    def predict_product_demand(self, product_id: str) -> Optional[MLPrediction]:
        prediction_data = product_service.predict_demand(product_id)
        if not prediction_data:
            return None
        
        return MLPrediction(
            product_id=prediction_data["product_id"],
            predicted_demand=prediction_data["predicted_demand"],
            confidence=prediction_data["confidence"],
            factors=prediction_data["factors"]
        )