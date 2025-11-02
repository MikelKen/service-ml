"""
Router para endpoints de sincronización
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from app.services.sync_service import auto_sync_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["Synchronization"])


@router.post("/force-sync")
async def force_synchronization():
    """
    Fuerza una sincronización inmediata entre PostgreSQL y MongoDB
    """
    try:
        stats = await auto_sync_service.force_sync()
        
        return {
            "status": "success",
            "message": "Sincronización forzada completada",
            "stats": stats,
            "total_synced": sum(stats.values())
        }
    
    except Exception as e:
        logger.error(f"Error en sincronización forzada: {e}")
        raise HTTPException(status_code=500, detail=f"Error en sincronización: {str(e)}")


@router.get("/status")
async def get_sync_status():
    """
    Obtiene el estado del servicio de sincronización
    """
    return {
        "status": "running" if auto_sync_service.is_running else "stopped",
        "sync_interval_seconds": auto_sync_service.sync_interval,
        "last_sync_timestamps": auto_sync_service.last_sync_timestamps,
        "service_info": {
            "description": "Servicio de sincronización automática PostgreSQL -> MongoDB",
            "collections": ["companies_features", "job_offers_features", "candidates_features"]
        }
    }


@router.post("/start")
async def start_sync_service():
    """
    Inicia el servicio de sincronización (si está detenido)
    """
    try:
        if auto_sync_service.is_running:
            return {
                "status": "warning",
                "message": "El servicio de sincronización ya está ejecutándose"
            }
        
        await auto_sync_service.start()
        
        return {
            "status": "success",
            "message": "Servicio de sincronización iniciado"
        }
    
    except Exception as e:
        logger.error(f"Error iniciando servicio de sincronización: {e}")
        raise HTTPException(status_code=500, detail=f"Error iniciando servicio: {str(e)}")


@router.post("/stop")
async def stop_sync_service():
    """
    Detiene el servicio de sincronización
    """
    try:
        if not auto_sync_service.is_running:
            return {
                "status": "warning",
                "message": "El servicio de sincronización ya está detenido"
            }
        
        await auto_sync_service.stop()
        
        return {
            "status": "success",
            "message": "Servicio de sincronización detenido"
        }
    
    except Exception as e:
        logger.error(f"Error deteniendo servicio de sincronización: {e}")
        raise HTTPException(status_code=500, detail=f"Error deteniendo servicio: {str(e)}")


@router.put("/interval/{seconds}")
async def update_sync_interval(seconds: int):
    """
    Actualiza el intervalo de sincronización (en segundos)
    
    Args:
        seconds: Nuevo intervalo en segundos (mínimo 10, máximo 3600)
    """
    try:
        if seconds < 10 or seconds > 3600:
            raise HTTPException(
                status_code=400, 
                detail="El intervalo debe estar entre 10 y 3600 segundos"
            )
        
        was_running = auto_sync_service.is_running
        
        # Detener si está ejecutándose
        if was_running:
            await auto_sync_service.stop()
        
        # Actualizar intervalo
        auto_sync_service.sync_interval = seconds
        
        # Reiniciar si estaba ejecutándose
        if was_running:
            await auto_sync_service.start()
        
        return {
            "status": "success",
            "message": f"Intervalo de sincronización actualizado a {seconds} segundos",
            "new_interval": seconds,
            "service_restarted": was_running
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error actualizando intervalo: {e}")
        raise HTTPException(status_code=500, detail=f"Error actualizando intervalo: {str(e)}")