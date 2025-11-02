"""Esquema GraphQL central de la aplicación.

Incluye consultas para leer las colecciones ERP en MongoDB.
"""
import strawberry
from typing import List, Optional

from app.graphql.types.erp_types import (
    EmpresaFeature, OfertaFeature, PostulanteFeature,
    EmpresaFilter, OfertaFilter, PostulanteFilter,
)
from app.graphql.resolvers import erp_resolvers


@strawberry.type
class Query:
    @strawberry.field(description="Empresas desde MongoDB")
    async def empresas(self, filter: Optional[EmpresaFilter] = None, limit: Optional[int] = 100) -> List[EmpresaFeature]:
        data = await erp_resolvers.get_empresas(filter.__dict__ if filter else None, limit)
        return [EmpresaFeature(**d) for d in data]

    @strawberry.field(description="Ofertas de trabajo desde MongoDB")
    async def ofertas(self, filter: Optional[OfertaFilter] = None, limit: Optional[int] = 100) -> List[OfertaFeature]:
        data = await erp_resolvers.get_ofertas(filter.__dict__ if filter else None, limit)
        return [OfertaFeature(**d) for d in data]

    @strawberry.field(description="Postulantes (features) desde MongoDB")
    async def postulantes(self, filter: Optional[PostulanteFilter] = None, limit: Optional[int] = 100) -> List[PostulanteFeature]:
        data = await erp_resolvers.get_postulantes(filter.__dict__ if filter else None, limit)
        return [PostulanteFeature(**d) for d in data]


schema = strawberry.Schema(query=Query)
