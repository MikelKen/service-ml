import strawberry
from typing import Optional
from app.schemas.product_schema import Product, ProductInput, ProductUpdateInput
from app.services.product_service import product_service
from app.graphql.ml_mutations import MLMutation


@strawberry.type
class Mutation(MLMutation):
    """Mutaciones principales del microservicio - ahora incluye ML"""
    
    @strawberry.mutation(description="Create a new product")
    def create_product(self, input: ProductInput) -> Product:
        product_data = {
            "name": input.name,
            "price": input.price,
            "category": input.category,
            "description": input.description,
            "stock": input.stock,
            "active": input.active,
            "tags": input.tags
        }
        
        created_product = product_service.create_product(product_data)
        
        return Product(
            id=created_product["id"],
            name=created_product["name"],
            price=created_product["price"],
            category=created_product.get("category"),
            description=created_product.get("description"),
            stock=created_product["stock"],
            active=created_product["active"],
            tags=created_product.get("tags", []),
            ml_score=created_product.get("ml_score"),
            created_at=created_product["created_at"],
            updated_at=created_product.get("updated_at")
        )
    
    @strawberry.mutation(description="Update an existing product")
    def update_product(self, id: str, input: ProductUpdateInput) -> Optional[Product]:
        update_data = {}
        
        if input.name is not None:
            update_data["name"] = input.name
        if input.price is not None:
            update_data["price"] = input.price
        if input.category is not None:
            update_data["category"] = input.category
        if input.description is not None:
            update_data["description"] = input.description
        if input.stock is not None:
            update_data["stock"] = input.stock
        if input.active is not None:
            update_data["active"] = input.active
        if input.tags is not None:
            update_data["tags"] = input.tags
        
        updated_product = product_service.update_product(id, update_data)
        
        if not updated_product:
            return None
        
        return Product(
            id=updated_product["id"],
            name=updated_product["name"],
            price=updated_product["price"],
            category=updated_product.get("category"),
            description=updated_product.get("description"),
            stock=updated_product["stock"],
            active=updated_product["active"],
            tags=updated_product.get("tags", []),
            ml_score=updated_product.get("ml_score"),
            created_at=updated_product["created_at"],
            updated_at=updated_product.get("updated_at")
        )
    
    @strawberry.mutation(description="Delete a product (soft delete)")
    def delete_product(self, id: str) -> bool:
        return product_service.delete_product(id)
    
    @strawberry.mutation(description="Toggle product active status")
    def toggle_product_status(self, id: str) -> Optional[Product]:
        product_data = product_service.get_product_by_id(id)
        if not product_data:
            return None
        
        current_status = product_data.get("active", True)
        update_data = {"active": not current_status}
        
        updated_product = product_service.update_product(id, update_data)
        
        if not updated_product:
            return None
        
        return Product(
            id=updated_product["id"],
            name=updated_product["name"],
            price=updated_product["price"],
            category=updated_product.get("category"),
            description=updated_product.get("description"),
            stock=updated_product["stock"],
            active=updated_product["active"],
            tags=updated_product.get("tags", []),
            ml_score=updated_product.get("ml_score"),
            created_at=updated_product["created_at"],
            updated_at=updated_product.get("updated_at")
        )