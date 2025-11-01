"""
Router for database queries endpoints
Exposes database query functionality via REST API
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

from app.database.queries import db_queries

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Database Queries"])

# ==================== EMPRESA ENDPOINTS ====================

@router.get("/empresas", response_model=List[Dict[str, Any]])
async def get_empresas():
    """Get all companies"""
    try:
        empresas = await db_queries.get_all_empresas()
        return empresas
    except Exception as e:
        logger.error(f"Error in get_empresas: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving companies")

@router.get("/empresas/{empresa_id}", response_model=Dict[str, Any])
async def get_empresa(empresa_id: int):
    """Get company by ID"""
    try:
        empresa = await db_queries.get_empresa_by_id(empresa_id)
        if not empresa:
            raise HTTPException(status_code=404, detail="Company not found")
        return empresa
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_empresa: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving company")

@router.get("/empresas/{empresa_id}/estadisticas", response_model=Dict[str, Any])
async def get_empresa_estadisticas(empresa_id: int):
    """Get company statistics"""
    try:
        stats = await db_queries.get_estadisticas_empresa(empresa_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Company not found or no data available")
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_empresa_estadisticas: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving company statistics")

# ==================== OFERTAS TRABAJO ENDPOINTS ====================

@router.get("/ofertas", response_model=List[Dict[str, Any]])
async def get_ofertas(empresa_id: Optional[int] = Query(None, description="Filter by company ID")):
    """Get job offers, optionally filtered by company"""
    try:
        if empresa_id:
            ofertas = await db_queries.get_ofertas_by_empresa(empresa_id)
        else:
            ofertas = await db_queries.get_all_ofertas()
        return ofertas
    except Exception as e:
        logger.error(f"Error in get_ofertas: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving job offers")

@router.get("/ofertas/{oferta_id}", response_model=Dict[str, Any])
async def get_oferta(oferta_id: int):
    """Get job offer by ID"""
    try:
        oferta = await db_queries.get_oferta_by_id(oferta_id)
        if not oferta:
            raise HTTPException(status_code=404, detail="Job offer not found")
        return oferta
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_oferta: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving job offer")

@router.get("/ofertas/{oferta_id}/visualizaciones", response_model=List[Dict[str, Any]])
async def get_oferta_visualizaciones(oferta_id: int):
    """Get job offer views"""
    try:
        visualizaciones = await db_queries.get_visualizaciones_by_oferta(oferta_id)
        return visualizaciones
    except Exception as e:
        logger.error(f"Error in get_oferta_visualizaciones: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving job offer views")

# ==================== POSTULACIONES ENDPOINTS ====================

@router.get("/postulaciones", response_model=List[Dict[str, Any]])
async def get_postulaciones(oferta_id: Optional[int] = Query(None, description="Filter by job offer ID")):
    """Get applications, optionally filtered by job offer"""
    try:
        if oferta_id:
            postulaciones = await db_queries.get_postulaciones_by_oferta(oferta_id)
        else:
            postulaciones = await db_queries.get_all_postulaciones()
        return postulaciones
    except Exception as e:
        logger.error(f"Error in get_postulaciones: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving applications")

@router.get("/postulaciones/{postulacion_id}", response_model=Dict[str, Any])
async def get_postulacion(postulacion_id: int):
    """Get application by ID"""
    try:
        postulacion = await db_queries.get_postulacion_by_id(postulacion_id)
        if not postulacion:
            raise HTTPException(status_code=404, detail="Application not found")
        return postulacion
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_postulacion: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving application")

# ==================== ENTREVISTAS ENDPOINTS ====================

@router.get("/entrevistas", response_model=List[Dict[str, Any]])
async def get_entrevistas(postulacion_id: Optional[int] = Query(None, description="Filter by application ID")):
    """Get interviews, optionally filtered by application"""
    try:
        if postulacion_id:
            entrevistas = await db_queries.get_entrevistas_by_postulacion(postulacion_id)
        else:
            entrevistas = await db_queries.get_all_entrevistas()
        return entrevistas
    except Exception as e:
        logger.error(f"Error in get_entrevistas: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving interviews")

@router.get("/entrevistas/{entrevista_id}", response_model=Dict[str, Any])
async def get_entrevista(entrevista_id: int):
    """Get interview by ID"""
    try:
        entrevista = await db_queries.get_entrevista_by_id(entrevista_id)
        if not entrevista:
            raise HTTPException(status_code=404, detail="Interview not found")
        return entrevista
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_entrevista: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving interview")

# ==================== EVALUACIONES ENDPOINTS ====================

@router.get("/evaluaciones", response_model=List[Dict[str, Any]])
async def get_evaluaciones(entrevista_id: Optional[int] = Query(None, description="Filter by interview ID")):
    """Get evaluations, optionally filtered by interview"""
    try:
        if entrevista_id:
            evaluaciones = await db_queries.get_evaluaciones_by_entrevista(entrevista_id)
        else:
            evaluaciones = await db_queries.get_all_evaluaciones()
        return evaluaciones
    except Exception as e:
        logger.error(f"Error in get_evaluaciones: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving evaluations")

# ==================== ESTADÍSTICAS ENDPOINTS ====================

@router.get("/estadisticas", response_model=Dict[str, Any])
async def get_estadisticas_generales():
    """Get general statistics"""
    try:
        stats = await db_queries.get_estadisticas_generales()
        return stats
    except Exception as e:
        logger.error(f"Error in get_estadisticas_generales: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving general statistics")

# ==================== HEALTH CHECK ====================

@router.get("/db-status")
async def check_database_status():
    """Check database connection status"""
    try:
        from app.database.connection import db
        db_status = await db.test_connection()
        
        if db_status:
            # Try a simple query
            empresas = await db_queries.get_all_empresas()
            return {
                "status": "connected",
                "message": "Database is accessible",
                "sample_data": {
                    "empresas_count": len(empresas)
                }
            }
        else:
            return {
                "status": "disconnected",
                "message": "Database connection failed"
            }
    except Exception as e:
        logger.error(f"Error checking database status: {e}")
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }