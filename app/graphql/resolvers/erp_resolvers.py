"""Resolvers para consultar colecciones ERP en MongoDB"""
from typing import List, Dict, Any
from app.config.connection import mongodb

COLL_POSTULANTES = "candidates_features"
COLL_OFERTAS = "job_offers_features"
COLL_EMPRESAS = "companies_features"


def _project_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Eliminar _id interno de Mongo para evitar conflictos con Strawberry
    if "_id" in doc:
        doc = dict(doc)
        doc.pop("_id", None)
    return doc


async def get_empresas(filter: Dict[str, Any] | None = None, limit: int | None = None) -> List[Dict[str, Any]]:
    await mongodb.connect()
    coll = mongodb.get_collection(COLL_EMPRESAS)
    query: Dict[str, Any] = {}

    if filter:
        if fid := filter.get("empresa_id"):
            query["empresa_id"] = fid
        if name := filter.get("nombre_contains"):
            query["nombre"] = {"$regex": name, "$options": "i"}

    cursor = coll.find(query)
    if limit:
        cursor = cursor.limit(int(limit))
    return [_project_id(d) async for d in cursor]


async def get_ofertas(filter: Dict[str, Any] | None = None, limit: int | None = None) -> List[Dict[str, Any]]:
    await mongodb.connect()
    coll = mongodb.get_collection(COLL_OFERTAS)
    query: Dict[str, Any] = {}

    if filter:
        if oid := filter.get("oferta_id"):
            query["oferta_id"] = oid
        if eid := filter.get("empresa_id"):
            query["empresa_id"] = eid
        if t := filter.get("titulo_contains"):
            query["titulo"] = {"$regex": t, "$options": "i"}

    cursor = coll.find(query)
    if limit:
        cursor = cursor.limit(int(limit))
    return [_project_id(d) async for d in cursor]


async def get_postulantes(filter: Dict[str, Any] | None = None, limit: int | None = None) -> List[Dict[str, Any]]:
    await mongodb.connect()
    coll = mongodb.get_collection(COLL_POSTULANTES)
    query: Dict[str, Any] = {}

    if filter:
        if pid := filter.get("postulante_id"):
            query["postulante_id"] = pid
        if oid := filter.get("oferta_id"):
            query["oferta_id"] = oid
        if edu := filter.get("nivel_educacion"):
            query["nivel_educacion"] = edu
        if minexp := filter.get("min_anios_experiencia"):
            query["anios_experiencia"] = {"$gte": int(minexp)}

    cursor = coll.find(query)
    if limit:
        cursor = cursor.limit(int(limit))
    return [_project_id(d) async for d in cursor]
