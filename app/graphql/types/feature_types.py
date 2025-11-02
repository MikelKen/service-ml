"""
Tipos GraphQL para consultar datos de features desde MongoDB
"""
import strawberry
from typing import List, Optional
from datetime import datetime


@strawberry.type
class CandidateFeature:
    """Características de un candidato desde MongoDB"""
    id: str = strawberry.field(name="_id")
    postulante_id: Optional[str] = None
    anios_experiencia: Optional[int] = None
    nivel_educacion: Optional[str] = None
    habilidades: Optional[str] = None
    idiomas: Optional[str] = None
    certificaciones: Optional[str] = None
    puesto_actual: Optional[str] = None
    oferta_id: Optional[str] = None
    oferta_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@strawberry.type
class JobOfferFeature:
    """Características de una oferta laboral desde MongoDB"""
    id: str = strawberry.field(name="_id")
    oferta_id: Optional[str] = None
    titulo: Optional[str] = None
    salario: Optional[float] = None
    ubicacion: Optional[str] = None
    requisitos: Optional[str] = None
    empresa_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@strawberry.type
class CompanyFeature:
    """Características de una empresa desde MongoDB"""
    id: str = strawberry.field(name="_id")
    empresa_id: Optional[str] = None
    nombre: Optional[str] = None
    rubro: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@strawberry.input
class FeatureQueryInput:
    """Input para consultas de features"""
    limit: Optional[int] = 10
    skip: Optional[int] = 0
    search: Optional[str] = None


@strawberry.type
class CandidateFeatureList:
    """Lista de candidatos con paginación"""
    items: List[CandidateFeature]
    total: int
    has_more: bool


@strawberry.type
class JobOfferFeatureList:
    """Lista de ofertas con paginación"""
    items: List[JobOfferFeature]
    total: int
    has_more: bool


@strawberry.type
class CompanyFeatureList:
    """Lista de empresas con paginación"""
    items: List[CompanyFeature]
    total: int
    has_more: bool


@strawberry.type
class FeatureCollectionInfo:
    """Información sobre una colección de features"""
    collection_name: str
    total_documents: int
    sample_fields: List[str]
    last_updated: Optional[str] = None