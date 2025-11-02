"""
Resolvers GraphQL para consultar datos de features desde MongoDB
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.graphql.types.feature_types import (
    CandidateFeature, JobOfferFeature, CompanyFeature,
    CandidateFeatureList, JobOfferFeatureList, CompanyFeatureList,
    FeatureQueryInput, FeatureCollectionInfo
)
from app.config.mongodb_connection import get_collection_sync

logger = logging.getLogger(__name__)


def _convert_mongo_doc_to_candidate(doc: Dict[str, Any]) -> CandidateFeature:
    """Convierte documento MongoDB a tipo CandidateFeature"""
    return CandidateFeature(
        id=str(doc.get('_id', '')),
        postulante_id=doc.get('postulante_id'),
        anios_experiencia=doc.get('anios_experiencia'),
        nivel_educacion=doc.get('nivel_educacion'),
        habilidades=doc.get('habilidades'),
        idiomas=doc.get('idiomas'),
        certificaciones=doc.get('certificaciones'),
        puesto_actual=doc.get('puesto_actual'),
        oferta_id=doc.get('oferta_id'),
        created_at=doc.get('created_at'),
        updated_at=doc.get('updated_at')
    )


def _convert_mongo_doc_to_job_offer(doc: Dict[str, Any]) -> JobOfferFeature:
    """Convierte documento MongoDB a tipo JobOfferFeature"""
    return JobOfferFeature(
        id=str(doc.get('_id', '')),
        oferta_id=doc.get('oferta_id'),
        titulo=doc.get('titulo'),
        salario=float(doc.get('salario', 0)) if doc.get('salario') is not None else None,
        ubicacion=doc.get('ubicacion'),
        requisitos=doc.get('requisitos'),
        empresa_id=doc.get('empresa_id'),
        created_at=doc.get('created_at'),
        updated_at=doc.get('updated_at')
    )


def _convert_mongo_doc_to_company(doc: Dict[str, Any]) -> CompanyFeature:
    """Convierte documento MongoDB a tipo CompanyFeature"""
    return CompanyFeature(
        id=str(doc.get('_id', '')),
        empresa_id=doc.get('empresa_id'),
        nombre=doc.get('nombre'),
        rubro=doc.get('rubro'),
        created_at=doc.get('created_at'),
        updated_at=doc.get('updated_at')
    )


async def get_candidates_features(query_input: Optional[FeatureQueryInput] = None) -> CandidateFeatureList:
    """Obtiene lista de candidatos con sus características"""
    
    if query_input is None:
        query_input = FeatureQueryInput()
    
    try:
        from app.config.mongodb_connection import get_mongodb_sync
        db = get_mongodb_sync()
        collection = db["candidates_features"]
        
        # Construir filtro de búsqueda
        filter_query = {}
        if query_input.search:
            filter_query = {
                "$or": [
                    {"habilidades": {"$regex": query_input.search, "$options": "i"}},
                    {"nivel_educacion": {"$regex": query_input.search, "$options": "i"}},
                    {"puesto_actual": {"$regex": query_input.search, "$options": "i"}},
                    {"idiomas": {"$regex": query_input.search, "$options": "i"}}
                ]
            }
        
        # Contar total
        total = collection.count_documents(filter_query)
        
        # Obtener documentos con paginación
        cursor = collection.find(filter_query).skip(query_input.skip or 0).limit(query_input.limit or 10)
        docs = list(cursor)
        
        # Convertir a tipos GraphQL
        items = [_convert_mongo_doc_to_candidate(doc) for doc in docs]
        
        has_more = total > (query_input.skip or 0) + len(items)
        
        return CandidateFeatureList(
            items=items,
            total=total,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo candidatos: {e}")
        return CandidateFeatureList(items=[], total=0, has_more=False)


async def get_job_offers_features(query_input: Optional[FeatureQueryInput] = None) -> JobOfferFeatureList:
    """Obtiene lista de ofertas laborales con sus características"""
    
    if query_input is None:
        query_input = FeatureQueryInput()
    
    try:
        from app.config.mongodb_connection import get_mongodb_sync
        db = get_mongodb_sync()
        collection = db["job_offers_features"]
        
        # Construir filtro de búsqueda
        filter_query = {}
        if query_input.search:
            filter_query = {
                "$or": [
                    {"titulo": {"$regex": query_input.search, "$options": "i"}},
                    {"ubicacion": {"$regex": query_input.search, "$options": "i"}},
                    {"requisitos": {"$regex": query_input.search, "$options": "i"}}
                ]
            }
        
        # Contar total
        total = collection.count_documents(filter_query)
        
        # Obtener documentos con paginación
        cursor = collection.find(filter_query).skip(query_input.skip or 0).limit(query_input.limit or 10)
        docs = list(cursor)
        
        # Convertir a tipos GraphQL
        items = [_convert_mongo_doc_to_job_offer(doc) for doc in docs]
        
        has_more = total > (query_input.skip or 0) + len(items)
        
        return JobOfferFeatureList(
            items=items,
            total=total,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo ofertas: {e}")
        return JobOfferFeatureList(items=[], total=0, has_more=False)


async def get_companies_features(query_input: Optional[FeatureQueryInput] = None) -> CompanyFeatureList:
    """Obtiene lista de empresas con sus características"""
    
    if query_input is None:
        query_input = FeatureQueryInput()
    
    try:
        collection = get_collection_sync("companies_features")
        
        # Construir filtro de búsqueda
        filter_query = {}
        if query_input.search:
            filter_query = {
                "$or": [
                    {"nombre": {"$regex": query_input.search, "$options": "i"}},
                    {"rubro": {"$regex": query_input.search, "$options": "i"}}
                ]
            }
        
        # Contar total
        total = collection.count_documents(filter_query)
        
        # Obtener documentos con paginación
        cursor = collection.find(filter_query).skip(query_input.skip or 0).limit(query_input.limit or 10)
        docs = list(cursor)
        
        # Convertir a tipos GraphQL
        items = [_convert_mongo_doc_to_company(doc) for doc in docs]
        
        has_more = total > (query_input.skip or 0) + len(items)
        
        return CompanyFeatureList(
            items=items,
            total=total,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo empresas: {e}")
        return CompanyFeatureList(items=[], total=0, has_more=False)


async def get_candidate_by_id(candidate_id: str) -> Optional[CandidateFeature]:
    """Obtiene un candidato específico por ID"""
    
    try:
        collection = get_collection_sync("candidates_features")
        
        # Buscar por _id o postulante_id
        doc = collection.find_one({"$or": [{"_id": candidate_id}, {"postulante_id": candidate_id}]})
        
        if doc:
            return _convert_mongo_doc_to_candidate(doc)
        
        return None
        
    except Exception as e:
        logger.error(f"Error obteniendo candidato {candidate_id}: {e}")
        return None


async def get_job_offer_by_id(offer_id: str) -> Optional[JobOfferFeature]:
    """Obtiene una oferta específica por ID"""
    
    try:
        collection = get_collection_sync("job_offers_features")
        
        # Buscar por _id o oferta_id
        doc = collection.find_one({"$or": [{"_id": offer_id}, {"oferta_id": offer_id}]})
        
        if doc:
            return _convert_mongo_doc_to_job_offer(doc)
        
        return None
        
    except Exception as e:
        logger.error(f"Error obteniendo oferta {offer_id}: {e}")
        return None


async def get_company_by_id(company_id: str) -> Optional[CompanyFeature]:
    """Obtiene una empresa específica por ID"""
    
    try:
        collection = get_collection_sync("companies_features")
        
        # Buscar por _id o empresa_id
        doc = collection.find_one({"$or": [{"_id": company_id}, {"empresa_id": company_id}]})
        
        if doc:
            return _convert_mongo_doc_to_company(doc)
        
        return None
        
    except Exception as e:
        logger.error(f"Error obteniendo empresa {company_id}: {e}")
        return None


async def get_collection_info(collection_name: str) -> FeatureCollectionInfo:
    """Obtiene información sobre una colección de features"""
    
    try:
        collection = get_collection_sync(collection_name)
        
        # Contar documentos
        total = collection.count_documents({})
        
        # Obtener campos de muestra
        sample_fields = []
        if total > 0:
            sample_doc = collection.find_one({})
            if sample_doc:
                sample_fields = list(sample_doc.keys())
        
        # Buscar última actualización
        last_updated = None
        try:
            latest_doc = collection.find_one({}, sort=[("updated_at", -1)])
            if latest_doc and latest_doc.get("updated_at"):
                last_updated = latest_doc["updated_at"]
        except:
            pass
        
        return FeatureCollectionInfo(
            collection_name=collection_name,
            total_documents=total,
            sample_fields=sample_fields,
            last_updated=last_updated
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo info de colección {collection_name}: {e}")
        return FeatureCollectionInfo(
            collection_name=collection_name,
            total_documents=0,
            sample_fields=[],
            last_updated=None
        )


async def get_candidates_by_offer_id(offer_id: str, limit: int = 10) -> List[CandidateFeature]:
    """Obtiene candidatos que aplicaron a una oferta específica"""
    
    try:
        collection = get_collection_sync("candidates_features")
        
        # Buscar candidatos para esta oferta
        cursor = collection.find({"oferta_id": offer_id}).limit(limit)
        docs = list(cursor)
        
        return [_convert_mongo_doc_to_candidate(doc) for doc in docs]
        
    except Exception as e:
        logger.error(f"Error obteniendo candidatos para oferta {offer_id}: {e}")
        return []


async def get_offers_by_company_id(company_id: str, limit: int = 10) -> List[JobOfferFeature]:
    """Obtiene ofertas de una empresa específica"""
    
    try:
        collection = get_collection_sync("job_offers_features")
        
        # Buscar ofertas de esta empresa
        cursor = collection.find({"empresa_id": company_id}).limit(limit)
        docs = list(cursor)
        
        return [_convert_mongo_doc_to_job_offer(doc) for doc in docs]
        
    except Exception as e:
        logger.error(f"Error obteniendo ofertas para empresa {company_id}: {e}")
        return []