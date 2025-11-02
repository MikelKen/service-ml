from fastapi import APIRouter, HTTPException
from typing import Any
import logging

from app.database.queries import db_queries
from app.database.connection import db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/db-status")
async def db_status() -> Any:
    """Return basic DB connection status"""
    try:
        status = await db.test_connection()
        return {"connected": bool(status)}
    except Exception as e:
        logger.error(f"Error checking DB status: {e}")
        raise HTTPException(status_code=500, detail="Error checking database status")


@router.get("/empresas")
async def list_empresas() -> Any:
    try:
        empresas = await db_queries.get_all_empresas()
        return empresas
    except Exception as e:
        logger.error(f"Error listing empresas: {e}")
        raise HTTPException(status_code=500, detail="Error listing empresas")


@router.get("/empresas/{empresa_id}")
async def get_empresa(empresa_id: str) -> Any:
    try:
        empresa = await db_queries.get_empresa_by_id(empresa_id)
        if not empresa:
            raise HTTPException(status_code=404, detail="Empresa not found")
        return empresa
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting empresa {empresa_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting empresa")


@router.get("/ofertas")
async def list_ofertas() -> Any:
    try:
        ofertas = await db_queries.get_all_ofertas()
        return ofertas
    except Exception as e:
        logger.error(f"Error listing ofertas: {e}")
        raise HTTPException(status_code=500, detail="Error listing ofertas")


@router.get("/ofertas/{oferta_id}")
async def get_oferta(oferta_id: str) -> Any:
    try:
        oferta = await db_queries.get_oferta_by_id(oferta_id)
        if not oferta:
            raise HTTPException(status_code=404, detail="Oferta not found")
        return oferta
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting oferta {oferta_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting oferta")


@router.get("/postulaciones")
async def list_postulaciones() -> Any:
    try:
        postulaciones = await db_queries.get_all_postulaciones()
        return postulaciones
    except Exception as e:
        logger.error(f"Error listing postulaciones: {e}")
        raise HTTPException(status_code=500, detail="Error listing postulaciones")


@router.get("/postulaciones/{postulacion_id}")
async def get_postulacion(postulacion_id: str) -> Any:
    try:
        postulacion = await db_queries.get_postulacion_by_id(postulacion_id)
        if not postulacion:
            raise HTTPException(status_code=404, detail="Postulacion not found")
        return postulacion
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting postulacion {postulacion_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting postulacion")


@router.get("/entrevistas")
async def list_entrevistas() -> Any:
    try:
        entrevistas = await db_queries.get_all_entrevistas()
        return entrevistas
    except Exception as e:
        logger.error(f"Error listing entrevistas: {e}")
        raise HTTPException(status_code=500, detail="Error listing entrevistas")


@router.get("/entrevistas/{entrevista_id}")
async def get_entrevista(entrevista_id: str) -> Any:
    try:
        entrevista = await db_queries.get_entrevista_by_id(entrevista_id)
        if not entrevista:
            raise HTTPException(status_code=404, detail="Entrevista not found")
        return entrevista
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entrevista {entrevista_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting entrevista")


@router.get("/evaluaciones")
async def list_evaluaciones() -> Any:
    try:
        evaluaciones = await db_queries.get_all_evaluaciones()
        return evaluaciones
    except Exception as e:
        logger.error(f"Error listing evaluaciones: {e}")
        raise HTTPException(status_code=500, detail="Error listing evaluaciones")


@router.get("/estadisticas")
async def estadisticas_generales() -> Any:
    try:
        stats = await db_queries.get_estadisticas_generales()
        return stats
    except Exception as e:
        logger.error(f"Error getting estadisticas generales: {e}")
        raise HTTPException(status_code=500, detail="Error getting estadisticas")


@router.get("/estadisticas/empresa/{empresa_id}")
async def estadisticas_por_empresa(empresa_id: str) -> Any:
    try:
        stats = await db_queries.get_estadisticas_empresa(empresa_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting estadisticas for empresa {empresa_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting estadisticas empresa")
