"""
Router REST para Clustering de Candidatos
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Optional, Dict, Any
import logging

from app.services.clustering_service import (
    clustering_service, train_candidate_clustering,
    get_clustering_results, find_similar_candidates_to_profile,
    get_clustering_analytics
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clustering", tags=["Clustering"])


@router.get("/", summary="Información del servicio de clustering")
async def clustering_info():
    """Información general del servicio de clustering"""
    return {
        "service": "Candidate Clustering Service",
        "description": "Agrupa candidatos por similitud de perfil usando Machine Learning",
        "features": [
            "Clustering automático de candidatos",
            "Búsqueda de candidatos similares",
            "Análisis de perfiles por clusters",
            "Múltiples algoritmos de clustering"
        ],
        "algorithms": ["K-Means", "Clustering Jerárquico", "DBSCAN"],
        "is_model_trained": clustering_service.is_model_trained
    }


@router.get("/status", summary="Estado del modelo de clustering")
async def clustering_status():
    """Obtiene el estado actual del modelo de clustering"""
    try:
        return {
            "is_trained": clustering_service.is_model_trained,
            "num_features": len(clustering_service.feature_names) if clustering_service.feature_names else 0,
            "silhouette_score": clustering_service.silhouette_score_value,
            "model_type": type(clustering_service.clustering_model).__name__ if clustering_service.clustering_model else None,
            "data_size": len(clustering_service.original_data) if clustering_service.original_data is not None else 0,
            "num_clusters": len(set(clustering_service.cluster_labels)) if clustering_service.cluster_labels is not None else 0
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", summary="Entrena modelo de clustering")
async def train_clustering(
    algorithm: str = Query("kmeans", description="Algoritmo de clustering"),
    n_clusters: Optional[int] = Query(None, description="Número de clusters"),
    eps: Optional[float] = Query(0.5, description="Epsilon para DBSCAN"),
    min_samples: Optional[int] = Query(5, description="Mínimo de muestras para DBSCAN"),
    linkage: Optional[str] = Query("ward", description="Linkage para clustering jerárquico")
):
    """Entrena el modelo de clustering con los parámetros especificados"""
    try:
        kwargs = {}
        
        if algorithm.lower() == "dbscan":
            kwargs.update({'eps': eps, 'min_samples': min_samples})
        elif algorithm.lower() == "hierarchical":
            kwargs.update({'linkage': linkage})
        
        logger.info(f"Iniciando entrenamiento: {algorithm}")
        result = train_candidate_clustering(algorithm, n_clusters, **kwargs)
        
        return {
            "message": "Modelo entrenado exitosamente",
            "algorithm": algorithm,
            "results": result
        }
        
    except Exception as e:
        logger.error(f"Error entrenando modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results", summary="Obtiene resultados del clustering")
async def get_clustering_results_endpoint():
    """Obtiene los resultados del clustering actual"""
    try:
        if not clustering_service.is_model_trained:
            raise HTTPException(
                status_code=400, 
                detail="Modelo no entrenado. Entrene el modelo primero."
            )
        
        results = get_clustering_results()
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo resultados: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters", summary="Obtiene información de todos los clusters")
async def get_clusters():
    """Obtiene información detallada de todos los clusters"""
    try:
        if not clustering_service.is_model_trained:
            raise HTTPException(
                status_code=400, 
                detail="Modelo no entrenado. Entrene el modelo primero."
            )
        
        clusters = clustering_service.analyze_clusters()
        return {"clusters": clusters}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo clusters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{cluster_id}", summary="Obtiene información de un cluster específico")
async def get_cluster_by_id(cluster_id: int):
    """Obtiene información detallada de un cluster específico"""
    try:
        if not clustering_service.is_model_trained:
            raise HTTPException(
                status_code=400, 
                detail="Modelo no entrenado. Entrene el modelo primero."
            )
        
        clusters = clustering_service.analyze_clusters()
        cluster = next((c for c in clusters if c['cluster_id'] == cluster_id), None)
        
        if not cluster:
            raise HTTPException(
                status_code=404, 
                detail=f"Cluster {cluster_id} no encontrado"
            )
        
        return cluster
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo cluster {cluster_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{cluster_id}/candidates", summary="Obtiene candidatos de un cluster")
async def get_candidates_in_cluster(cluster_id: int):
    """Obtiene todos los candidatos asignados a un cluster específico"""
    try:
        if not clustering_service.is_model_trained:
            raise HTTPException(
                status_code=400, 
                detail="Modelo no entrenado. Entrene el modelo primero."
            )
        
        assignments = clustering_service.get_candidate_cluster_assignments()
        cluster_candidates = [
            assignment for assignment in assignments 
            if assignment['cluster_id'] == cluster_id
        ]
        
        if not cluster_candidates:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontraron candidatos en el cluster {cluster_id}"
            )
        
        return {
            "cluster_id": cluster_id,
            "candidate_count": len(cluster_candidates),
            "candidates": cluster_candidates
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo candidatos del cluster {cluster_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/find-similar", summary="Encuentra candidatos similares")
async def find_similar_candidates(
    candidate_profile: Dict[str, Any] = Body(..., description="Perfil del candidato de referencia"),
    max_results: int = Query(10, description="Número máximo de resultados")
):
    """Encuentra candidatos similares a un perfil dado"""
    try:
        if not clustering_service.is_model_trained:
            raise HTTPException(
                status_code=400, 
                detail="Modelo no entrenado. Entrene el modelo primero."
            )
        
        # Validar campos requeridos
        required_fields = [
            'nombre', 'años_experiencia', 'area_especialidad', 
            'habilidades_tecnicas', 'habilidades_blandas', 'salario_esperado'
        ]
        
        missing_fields = [field for field in required_fields if field not in candidate_profile]
        if missing_fields:
            raise HTTPException(
                status_code=400, 
                detail=f"Campos requeridos faltantes: {missing_fields}"
            )
        
        # Completar campos opcionales con valores por defecto
        defaults = {
            'edad': 30,
            'nivel_educacion': 'Licenciatura',
            'idiomas': 'Español,Inglés',
            'certificaciones': '',
            'disponibilidad_viajar': 'No',
            'modalidad_trabajo': 'Híbrido',
            'ubicacion': 'Santa Cruz',
            'industria_experiencia': 'Tecnología',
            'puesto_actual': candidate_profile.get('area_especialidad', 'Profesional'),
            'liderazgo_equipos': 0,
            'proyectos_completados': candidate_profile.get('años_experiencia', 0) * 3,
            'educacion_continua': 'Sí',
            'redes_profesionales': 'LinkedIn'
        }
        
        for key, value in defaults.items():
            if key not in candidate_profile:
                candidate_profile[key] = value
        
        result = find_similar_candidates_to_profile(candidate_profile, max_results)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error encontrando candidatos similares: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics", summary="Obtiene analíticas de clustering")
async def get_analytics():
    """Obtiene analíticas y estadísticas del clustering"""
    try:
        if not clustering_service.is_model_trained:
            raise HTTPException(
                status_code=400, 
                detail="Modelo no entrenado. Entrene el modelo primero."
            )
        
        analytics = get_clustering_analytics()
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo analíticas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset", summary="Resetea el modelo de clustering")
async def reset_clustering():
    """Resetea el modelo de clustering actual"""
    try:
        clustering_service.clustering_model = None
        clustering_service.processed_data = None
        clustering_service.original_data = None
        clustering_service.cluster_labels = None
        clustering_service.cluster_centers = None
        clustering_service.is_model_trained = False
        clustering_service.silhouette_score_value = 0.0
        
        logger.info("Modelo de clustering reseteado")
        return {"message": "Modelo reseteado exitosamente"}
        
    except Exception as e:
        logger.error(f"Error reseteando modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset/info", summary="Información del dataset")
async def get_dataset_info():
    """Obtiene información sobre el dataset de candidatos"""
    try:
        df = clustering_service.load_data()
        
        return {
            "total_candidates": len(df),
            "columns": df.columns.tolist(),
            "sample_data": df.head(3).to_dict('records'),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "file_path": clustering_service.data_path
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo info del dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-cluster", summary="Predice cluster para un candidato")
async def predict_cluster_for_candidate(
    candidate_profile: Dict[str, Any] = Body(..., description="Perfil del candidato")
):
    """Predice a qué cluster pertenecería un nuevo candidato"""
    try:
        if not clustering_service.is_model_trained:
            raise HTTPException(
                status_code=400, 
                detail="Modelo no entrenado. Entrene el modelo primero."
            )
        
        # Validar y completar perfil
        required_fields = [
            'nombre', 'años_experiencia', 'area_especialidad', 
            'habilidades_tecnicas', 'habilidades_blandas', 'salario_esperado'
        ]
        
        missing_fields = [field for field in required_fields if field not in candidate_profile]
        if missing_fields:
            raise HTTPException(
                status_code=400, 
                detail=f"Campos requeridos faltantes: {missing_fields}"
            )
        
        # Usar el servicio de búsqueda de similares que ya implementa la predicción
        similar_results = find_similar_candidates_to_profile(candidate_profile, max_results=1)
        
        if similar_results['similar_candidates']:
            predicted_cluster = similar_results['similar_candidates'][0]['cluster_id']
            cluster_name = similar_results['similar_candidates'][0]['cluster_name']
        else:
            predicted_cluster = -1
            cluster_name = "Sin cluster asignado"
        
        return {
            "candidate_name": candidate_profile['nombre'],
            "predicted_cluster_id": predicted_cluster,
            "predicted_cluster_name": cluster_name,
            "confidence": 0.85,  # Valor estimado
            "similar_candidates_found": len(similar_results['similar_candidates'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error prediciendo cluster: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))