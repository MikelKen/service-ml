import uuid
import random
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductService:
    def __init__(self):
        # In-memory storage for demonstration
        self.products: Dict[str, Dict[str, Any]] = {}
        self.ml_models_loaded = False
        self._initialize_sample_data()
        self._initialize_ml_components()

    def _initialize_sample_data(self):
        """Initialize with sample product data"""
        sample_products = [
            {
                "name": "MacBook Pro 16\"",
                "price": 2499.99,
                "category": "Electronics",
                "description": "Powerful laptop with M2 chip for professionals",
                "stock": 25,
                "active": True,
                "tags": ["laptop", "apple", "professional", "m2"],
                "ml_score": 0.95
            },
            {
                "name": "iPhone 15 Pro",
                "price": 999.99,
                "category": "Electronics",
                "description": "Latest iPhone with advanced camera system",
                "stock": 50,
                "active": True,
                "tags": ["smartphone", "apple", "camera", "5g"],
                "ml_score": 0.92
            },
            {
                "name": "AirPods Pro",
                "price": 249.99,
                "category": "Electronics",
                "description": "Wireless earbuds with active noise cancellation",
                "stock": 75,
                "active": True,
                "tags": ["audio", "wireless", "apple", "noise-cancellation"],
                "ml_score": 0.88
            },
            {
                "name": "Samsung Galaxy S24",
                "price": 899.99,
                "category": "Electronics",
                "description": "Android flagship smartphone with AI features",
                "stock": 30,
                "active": True,
                "tags": ["smartphone", "samsung", "android", "ai"],
                "ml_score": 0.85
            },
            {
                "name": "Gaming Chair Pro",
                "price": 299.99,
                "category": "Furniture",
                "description": "Ergonomic gaming chair with lumbar support",
                "stock": 15,
                "active": True,
                "tags": ["gaming", "chair", "ergonomic", "furniture"],
                "ml_score": 0.78
            },
            {
                "name": "Wireless Mouse",
                "price": 49.99,
                "category": "Electronics",
                "description": "Bluetooth wireless mouse with precision tracking",
                "stock": 100,
                "active": False,
                "tags": ["mouse", "wireless", "bluetooth", "computer"],
                "ml_score": 0.65
            }
        ]

        for product_data in sample_products:
            product_id = str(uuid.uuid4())
            product = {
                "id": product_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": None,
                **product_data
            }
            self.products[product_id] = product

        logger.info(f"🛍️ Initialized {len(self.products)} sample products")

    def _initialize_ml_components(self):
        """Initialize ML components for recommendations"""
        try:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            self.ml_models_loaded = True
            logger.info("🤖 ML components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML components: {e}")
            self.ml_models_loaded = False

    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products"""
        logger.info("📦 Fetching all products")
        return list(self.products.values())

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID"""
        logger.info(f"📦 Fetching product with ID: {product_id}")
        return self.products.get(product_id)

    def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get products by category"""
        logger.info(f"📦 Fetching products by category: {category}")
        return [
            product for product in self.products.values()
            if product.get("category", "").lower() == category.lower()
        ]

    def get_active_products(self) -> List[Dict[str, Any]]:
        """Get only active products"""
        logger.info("✅ Fetching active products")
        return [
            product for product in self.products.values()
            if product.get("active", False)
        ]

    def create_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new product"""
        product_id = str(uuid.uuid4())
        logger.info(f"➕ Creating new product: {product_data.get('name')}")

        # Calculate ML score based on product features
        ml_score = self._calculate_ml_score(product_data)

        product = {
            "id": product_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": None,
            "ml_score": ml_score,
            **product_data
        }

        self.products[product_id] = product
        logger.info(f"✅ Product created successfully with ID: {product_id}")
        return product

    def update_product(self, product_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing product"""
        if product_id not in self.products:
            logger.warning(f"❌ Product not found: {product_id}")
            return None

        logger.info(f"🔄 Updating product: {product_id}")
        product = self.products[product_id]

        # Update fields
        for key, value in update_data.items():
            if value is not None:
                product[key] = value

        product["updated_at"] = datetime.now().isoformat()

        # Recalculate ML score if relevant fields changed
        if any(key in update_data for key in ["name", "description", "tags", "category"]):
            product["ml_score"] = self._calculate_ml_score(product)

        self.products[product_id] = product
        logger.info(f"✅ Product updated successfully: {product_id}")
        return product

    def delete_product(self, product_id: str) -> bool:
        """Delete a product (soft delete by marking inactive)"""
        if product_id not in self.products:
            logger.warning(f"❌ Product not found: {product_id}")
            return False

        logger.info(f"🗑️ Soft deleting product: {product_id}")
        self.products[product_id]["active"] = False
        self.products[product_id]["updated_at"] = datetime.now().isoformat()
        logger.info(f"✅ Product soft deleted: {product_id}")
        return True

    def get_product_recommendations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate ML-based product recommendations"""
        logger.info(f"🤖 Generating recommendations for user: {user_id}")

        active_products = self.get_active_products()
        if not active_products:
            return []

        recommendations = []
        for product in active_products[:limit]:
            score = product.get("ml_score", 0.5) + random.uniform(-0.1, 0.1)
            score = max(0.0, min(1.0, score))  # Clamp between 0 and 1

            recommendations.append({
                "product_id": product["id"],
                "score": round(score, 3),
                "reason": f"ML recommendation based on {product.get('category', 'product')} popularity and user preferences",
                "user_id": user_id
            })

        # Sort by score descending
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"✅ Generated {len(recommendations)} recommendations")
        return recommendations

    def predict_demand(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Predict product demand using ML"""
        product = self.get_product_by_id(product_id)
        if not product:
            return None

        logger.info(f"📊 Predicting demand for product: {product_id}")

        # Simulate ML prediction
        base_demand = product.get("stock", 0) * 0.1
        ml_factor = product.get("ml_score", 0.5)
        predicted_demand = base_demand * (1 + ml_factor) + random.uniform(0, 10)

        prediction = {
            "product_id": product_id,
            "predicted_demand": round(predicted_demand, 2),
            "confidence": round(ml_factor * 0.9 + 0.1, 3),
            "factors": [
                "Historical sales data",
                "Seasonal trends",
                "Product popularity score",
                "Market demand patterns"
            ]
        }

        logger.info(f"✅ Demand prediction completed for {product_id}")
        return prediction

    def _calculate_ml_score(self, product_data: Dict[str, Any]) -> float:
        """Calculate ML score based on product features"""
        try:
            score = 0.5  # Base score

            # Price factor (normalized)
            price = product_data.get("price", 0)
            if price > 0:
                # Higher score for mid-range prices
                price_factor = 1.0 - abs(price - 500) / 1000
                score += price_factor * 0.2

            # Category factor
            popular_categories = ["Electronics", "Computers", "Mobile"]
            if product_data.get("category") in popular_categories:
                score += 0.15

            # Tags factor
            tags = product_data.get("tags", [])
            if len(tags) > 2:
                score += 0.1

            # Description factor
            description = product_data.get("description", "")
            if len(description) > 50:
                score += 0.05

            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"⚠️ Error calculating ML score: {e}")
            return 0.5

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "status": "healthy",
            "service": "service_ml",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "ml_models_loaded": self.ml_models_loaded,
            "total_products": len(self.products),
            "active_products": len(self.get_active_products())
        }


# Global service instance
product_service = ProductService()