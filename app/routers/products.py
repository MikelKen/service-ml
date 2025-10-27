from fastapi import APIRouter, HTTPException
from typing import List, Optional
from app.models.product import Product, ProductInput, ProductUpdate
from app.services.product_service import product_service

router = APIRouter(prefix="/products", tags=["products"])


@router.get("/", response_model=List[Product])
async def get_all_products():
    """Get all products"""
    products_data = product_service.get_all_products()
    return products_data


@router.get("/{product_id}", response_model=Product)
async def get_product(product_id: str):
    """Get product by ID"""
    product = product_service.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.get("/category/{category}", response_model=List[Product])
async def get_products_by_category(category: str):
    """Get products by category"""
    products = product_service.get_products_by_category(category)
    return products


@router.get("/active/list", response_model=List[Product])
async def get_active_products():
    """Get only active products"""
    products = product_service.get_active_products()
    return products


@router.post("/", response_model=Product)
async def create_product(product_input: ProductInput):
    """Create a new product"""
    product_data = product_input.dict()
    product = product_service.create_product(product_data)
    return product


@router.put("/{product_id}", response_model=Product)
async def update_product(product_id: str, product_update: ProductUpdate):
    """Update an existing product"""
    update_data = product_update.dict(exclude_unset=True)
    product = product_service.update_product(product_id, update_data)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.delete("/{product_id}")
async def delete_product(product_id: str):
    """Delete a product (soft delete)"""
    success = product_service.delete_product(product_id)
    if not success:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"message": "Product deleted successfully"}


@router.patch("/{product_id}/toggle-status", response_model=Product)
async def toggle_product_status(product_id: str):
    """Toggle product active status"""
    product = product_service.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    current_status = product.get("active", True)
    update_data = {"active": not current_status}
    updated_product = product_service.update_product(product_id, update_data)
    return updated_product


@router.get("/{product_id}/recommendations")
async def get_product_recommendations(product_id: str, user_id: str, limit: int = 5):
    """Get ML-based recommendations for a product/user"""
    recommendations = product_service.get_product_recommendations(user_id, limit)
    return {
        "product_id": product_id,
        "user_id": user_id,
        "recommendations": recommendations
    }


@router.get("/{product_id}/demand-prediction")
async def predict_product_demand(product_id: str):
    """Predict demand for a specific product"""
    prediction = product_service.predict_demand(product_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Product not found")
    return prediction