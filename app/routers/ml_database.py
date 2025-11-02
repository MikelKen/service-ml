"""
Router de FastAPI para endpoints específicos de base de datos ML
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging

from app.database.connection import get_database
from app.database.ml_queries import ml_db_queries
from app.services.ml_service import ml_service, get_database_dataset_info, predict_new_applications

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml/database", tags=["ML Database"])


@router.get("/training-data", response_model=Dict[str, Any])
async def get_training_data():
    """
    Obtiene los datos de entrenamiento desde la base de datos
    """
    try:
        data = await ml_db_queries.get_training_data_aggregated()
        
        return {
            "status": "success",
            "total_records": len(data),
            "data": data[:10] if data else [],  # Solo mostrar primeros 10 registros
            "message": f"Se obtuvieron {len(data)} registros de entrenamiento"
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo datos de entrenamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo datos: {str(e)}")


@router.get("/dataset-info", response_model=Dict[str, Any])
async def get_dataset_info_from_db():
    """
    Obtiene información estadística del dataset desde la base de datos
    """
    try:
        info = await get_database_dataset_info()
        return {
            "status": "success",
            "dataset_info": info
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo info del dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo info: {str(e)}")


@router.get("/feature-statistics", response_model=Dict[str, Any])
async def get_feature_statistics():
    """
    Obtiene estadísticas detalladas de las features
    """
    try:
        stats = await ml_db_queries.get_feature_statistics()
        
        return {
            "status": "success",
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.post("/train-model", response_model=Dict[str, Any])
async def train_model_from_database():
    """
    Entrena el modelo usando datos de la base de datos
    """
    try:
        # Verificar si ya hay entrenamiento en progreso
        current_status = ml_service.get_training_status()
        if current_status['is_training']:
            return {
                "status": "warning",
                "message": "Ya hay un entrenamiento en progreso",
                "training_status": current_status
            }
        
        # Iniciar entrenamiento
        success = await ml_service.train_model_from_database_async()
        
        if success:
            return {
                "status": "success",
                "message": "Entrenamiento completado exitosamente",
                "model_info": ml_service.get_model_info()
            }
        else:
            return {
                "status": "error",
                "message": "Error durante el entrenamiento"
            }
    
    except Exception as e:
        logger.error(f"Error entrenando modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")


@router.get("/predict-applications", response_model=Dict[str, Any])
async def predict_applications_endpoint(
    empresa_id: Optional[str] = None,
    oferta_id: Optional[str] = None,
    limit: Optional[int] = 50
):
    """
    Realiza predicciones para postulaciones desde la base de datos
    
    Args:
        empresa_id: ID de la empresa para filtrar (opcional)
        oferta_id: ID de la oferta para filtrar (opcional)
        limit: Límite de resultados (opcional)
    """
    try:
        if not ml_service.is_model_loaded:
            raise HTTPException(
                status_code=400, 
                detail="Modelo no cargado. Entrene o cargue un modelo primero."
            )
        
        predictions = await predict_new_applications(empresa_id, oferta_id)
        
        # Limitar resultados si se especifica
        if limit and len(predictions) > limit:
            predictions = predictions[:limit]
        
        return {
            "status": "success",
            "total_predictions": len(predictions),
            "empresa_id": empresa_id,
            "oferta_id": oferta_id,
            "predictions": predictions
        }
    
    except Exception as e:
        logger.error(f"Error realizando predicciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error realizando predicciones: {str(e)}")


@router.get("/recent-applications", response_model=Dict[str, Any])
async def get_recent_applications(days: int = 30):
    """
    Obtiene aplicaciones recientes para monitoreo del modelo
    
    Args:
        days: Número de días hacia atrás para buscar aplicaciones
    """
    try:
        applications = await ml_db_queries.get_recent_applications_for_monitoring(days)
        
        return {
            "status": "success",
            "days_back": days,
            "total_applications": len(applications),
            "applications": applications
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo aplicaciones recientes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo aplicaciones: {str(e)}")


@router.get("/health-check", response_model=Dict[str, Any])
async def health_check_database():
    """
    Verifica la conexión a la base de datos y el estado del servicio ML
    """
    try:
        from app.database.connection import db
        
        # Verificar conexión a la base de datos
        db_connected = await db.test_connection()
        
        # Verificar estado del modelo ML
        model_info = ml_service.get_model_info()
        
        return {
            "status": "healthy" if db_connected and model_info['is_loaded'] else "warning",
            "database_connected": db_connected,
            "model_loaded": model_info['is_loaded'],
            "model_info": model_info,
            "timestamp": "2024-11-01T00:00:00Z"
        }
    
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "2024-11-01T00:00:00Z"
        }


@router.get("/validate-data", response_model=Dict[str, Any])
async def validate_training_data():
    """
    Valida la calidad y consistencia de los datos de entrenamiento
    """
    try:
        # Obtener datos y estadísticas
        stats = await ml_db_queries.get_feature_statistics()
        training_data = await ml_db_queries.get_training_data_aggregated()
        
        if not training_data:
            return {
                "status": "error",
                "message": "No se encontraron datos de entrenamiento",
                "validations": []
            }
        
        validations = []
        
        # Validación 1: Verificar balance de clases
        general_stats = stats.get('general_stats', {})
        contacted = general_stats.get('contacted_candidates', 0)
        not_contacted = general_stats.get('not_contacted_candidates', 0)
        total = contacted + not_contacted
        
        if total > 0:
            balance_ratio = contacted / total
            if balance_ratio < 0.1 or balance_ratio > 0.9:
                validations.append({
                    "type": "warning",
                    "message": f"Desbalance de clases detectado: {balance_ratio:.2%} contactados",
                    "recommendation": "Considerar técnicas de balanceado de datos"
                })
            else:
                validations.append({
                    "type": "success",
                    "message": f"Balance de clases aceptable: {balance_ratio:.2%} contactados"
                })
        
        # Validación 2: Tamaño del dataset
        if total < 100:
            validations.append({
                "type": "error",
                "message": f"Dataset muy pequeño: {total} registros",
                "recommendation": "Se necesitan al menos 100 registros para entrenamiento confiable"
            })
        elif total < 500:
            validations.append({
                "type": "warning",
                "message": f"Dataset pequeño: {total} registros",
                "recommendation": "Se recomiendan al menos 500 registros para mejores resultados"
            })
        else:
            validations.append({
                "type": "success",
                "message": f"Tamaño de dataset adecuado: {total} registros"
            })
        
        # Validación 3: Diversidad de features
        education_levels = len(stats.get('education_distribution', []))
        sectors = len(stats.get('sector_distribution', []))
        
        if education_levels < 3:
            validations.append({
                "type": "warning",
                "message": f"Poca diversidad en niveles educativos: {education_levels}",
                "recommendation": "Buscar más diversidad en perfiles educativos"
            })
        
        if sectors < 3:
            validations.append({
                "type": "warning",
                "message": f"Poca diversidad en sectores industriales: {sectors}",
                "recommendation": "Incluir más sectores industriales en los datos"
            })
        
        return {
            "status": "success",
            "total_records": total,
            "validations": validations,
            "statistics_summary": stats
        }
    
    except Exception as e:
        logger.error(f"Error validando datos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validando datos: {str(e)}")