"""Tipos GraphQL para colecciones ERP en MongoDB"""
import strawberry
from typing import Optional, List


@strawberry.type
class EmpresaFeature:
    empresa_id: str
    nombre: str
    rubro: Optional[str]


@strawberry.type
class OfertaFeature:
    oferta_id: str
    titulo: str
    salario: Optional[float]
    ubicacion: Optional[str]
    requisitos: Optional[str]
    empresa_id: str


@strawberry.type
class PostulanteFeature:
    postulante_id: str
    anios_experiencia: Optional[int]
    nivel_educacion: Optional[str]
    habilidades: Optional[str]
    idiomas: Optional[str]
    certificaciones: Optional[str]
    puesto_actual: Optional[str]
    oferta_id: str


@strawberry.input
class EmpresaFilter:
    empresa_id: Optional[str] = None
    nombre_contains: Optional[str] = None


@strawberry.input
class OfertaFilter:
    oferta_id: Optional[str] = None
    empresa_id: Optional[str] = None
    titulo_contains: Optional[str] = None


@strawberry.input
class PostulanteFilter:
    postulante_id: Optional[str] = None
    oferta_id: Optional[str] = None
    nivel_educacion: Optional[str] = None
    min_anios_experiencia: Optional[int] = None
