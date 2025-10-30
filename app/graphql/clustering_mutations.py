"""
GraphQL Mutations para Clustering de Candidatos
"""
import strawberry
from typing import Optional, List
import logging
import asyncio

from app.schemas.ml_schemas import (
    ClusteringResult, ClusteringParameters, CandidateProfileInput,
    ModelParameter
)
from app.services.clustering_service import (
    train_candidate_clustering, clustering_service
)

logger = logging.getLogger(__name__)


@strawberry.type
class ClusteringMutations:
    """Mutations relacionadas con clustering de candidatos"""
    
    def _convert_model_parameters(self, params_dict: dict) -> List[ModelParameter]:
        """Convierte diccionario a lista de ModelParameter"""
        return [
            ModelParameter(key=str(k), value=str(v))
            for k, v in params_dict.items()
        ]
    
    @strawberry.mutation
    async def train_clustering_model(self, 
                                   parameters: Optional[ClusteringParameters] = None) -> Optional[ClusteringResult]:
        """Entrena el modelo de clustering de candidatos"""
        try:
            # Parámetros por defecto
            algorithm = "kmeans"
            n_clusters = None
            kwargs = {}
            
            if parameters:
                algorithm = parameters.algorithm or "kmeans"
                n_clusters = parameters.n_clusters
                
                if algorithm.lower() == "dbscan":
                    kwargs['eps'] = parameters.eps or 0.5
                    kwargs['min_samples'] = parameters.min_samples or 5
                elif algorithm.lower() == "hierarchical":
                    kwargs['linkage'] = parameters.linkage or 'ward'
                
                kwargs['max_clusters'] = parameters.max_clusters or 10
            
            logger.info(f"Iniciando entrenamiento de clustering: {algorithm}")
            
            # Ejecutar entrenamiento en un hilo separado para no bloquear
            loop = asyncio.get_event_loop()
            result_data = await loop.run_in_executor(
                None, 
                train_candidate_clustering,
                algorithm,
                n_clusters,
                **kwargs
            )
            
            # Convertir resultado a objetos Strawberry
            from app.schemas.ml_schemas import ClusterInfo, CandidateClusterResult
            
            clusters = [
                ClusterInfo(
                    cluster_id=cluster['cluster_id'],
                    cluster_name=cluster['cluster_name'],
                    candidate_count=cluster['candidate_count'],
                    description=cluster['description'],
                    key_characteristics=cluster['key_characteristics'],
                    avg_experience_years=cluster['avg_experience_years'],
                    avg_salary_expectation=cluster['avg_salary_expectation'],
                    common_skills=cluster['common_skills'],
                    common_industries=cluster['common_industries'],
                    education_levels=cluster['education_levels']
                )
                for cluster in result_data['clusters']
            ]
            
            candidate_assignments = [
                CandidateClusterResult(
                    candidate_id=assignment['candidate_id'],
                    candidate_name=assignment['candidate_name'],
                    cluster_id=assignment['cluster_id'],
                    cluster_name=assignment['cluster_name'],
                    similarity_score=assignment['similarity_score'],
                    distance_to_centroid=assignment['distance_to_centroid'],
                    profile_summary=assignment['profile_summary']
                )
                for assignment in result_data['candidate_assignments']
            ]
            
            logger.info(f"Clustering completado: {result_data['num_clusters']} clusters")
            
            return ClusteringResult(
                total_candidates=result_data['total_candidates'],
                num_clusters=result_data['num_clusters'],
                silhouette_score=result_data['silhouette_score'],
                clusters=clusters,
                candidate_assignments=candidate_assignments,
                processing_time_ms=result_data['processing_time_ms'],
                model_parameters=self._convert_model_parameters(result_data['model_parameters'])
            )
            
        except Exception as e:
            logger.error(f"Error entrenando modelo de clustering: {str(e)}")
            return None
    
    @strawberry.mutation
    def retrain_clustering_with_kmeans(self, n_clusters: Optional[int] = None) -> Optional[ClusteringResult]:
        """Reentrena el modelo usando K-Means con número específico de clusters"""
        try:
            parameters = ClusteringParameters(
                algorithm="kmeans",
                n_clusters=n_clusters,
                max_clusters=10
            )
            
            # Llamar a la función de entrenamiento síncrona para esta versión simple
            result_data = train_candidate_clustering(
                algorithm="kmeans",
                n_clusters=n_clusters
            )
            
            # Convertir resultado
            from app.schemas.ml_schemas import ClusterInfo, CandidateClusterResult
            
            clusters = [
                ClusterInfo(
                    cluster_id=cluster['cluster_id'],
                    cluster_name=cluster['cluster_name'],
                    candidate_count=cluster['candidate_count'],
                    description=cluster['description'],
                    key_characteristics=cluster['key_characteristics'],
                    avg_experience_years=cluster['avg_experience_years'],
                    avg_salary_expectation=cluster['avg_salary_expectation'],
                    common_skills=cluster['common_skills'],
                    common_industries=cluster['common_industries'],
                    education_levels=cluster['education_levels']
                )
                for cluster in result_data['clusters']
            ]
            
            candidate_assignments = [
                CandidateClusterResult(
                    candidate_id=assignment['candidate_id'],
                    candidate_name=assignment['candidate_name'],
                    cluster_id=assignment['cluster_id'],
                    cluster_name=assignment['cluster_name'],
                    similarity_score=assignment['similarity_score'],
                    distance_to_centroid=assignment['distance_to_centroid'],
                    profile_summary=assignment['profile_summary']
                )
                for assignment in result_data['candidate_assignments']
            ]
            
            return ClusteringResult(
                total_candidates=result_data['total_candidates'],
                num_clusters=result_data['num_clusters'],
                silhouette_score=result_data['silhouette_score'],
                clusters=clusters,
                candidate_assignments=candidate_assignments,
                processing_time_ms=result_data['processing_time_ms'],
                model_parameters=self._convert_model_parameters(result_data['model_parameters'])
            )
            
        except Exception as e:
            logger.error(f"Error reentrenando modelo K-Means: {str(e)}")
            return None
    
    @strawberry.mutation
    def train_hierarchical_clustering(self, 
                                    n_clusters: Optional[int] = None,
                                    linkage: Optional[str] = "ward") -> Optional[ClusteringResult]:
        """Entrena modelo usando clustering jerárquico"""
        try:
            result_data = train_candidate_clustering(
                algorithm="hierarchical",
                n_clusters=n_clusters,
                linkage=linkage or "ward"
            )
            
            # Convertir resultado
            from app.schemas.ml_schemas import ClusterInfo, CandidateClusterResult
            
            clusters = [
                ClusterInfo(
                    cluster_id=cluster['cluster_id'],
                    cluster_name=cluster['cluster_name'],
                    candidate_count=cluster['candidate_count'],
                    description=cluster['description'],
                    key_characteristics=cluster['key_characteristics'],
                    avg_experience_years=cluster['avg_experience_years'],
                    avg_salary_expectation=cluster['avg_salary_expectation'],
                    common_skills=cluster['common_skills'],
                    common_industries=cluster['common_industries'],
                    education_levels=cluster['education_levels']
                )
                for cluster in result_data['clusters']
            ]
            
            candidate_assignments = [
                CandidateClusterResult(
                    candidate_id=assignment['candidate_id'],
                    candidate_name=assignment['candidate_name'],
                    cluster_id=assignment['cluster_id'],
                    cluster_name=assignment['cluster_name'],
                    similarity_score=assignment['similarity_score'],
                    distance_to_centroid=assignment['distance_to_centroid'],
                    profile_summary=assignment['profile_summary']
                )
                for assignment in result_data['candidate_assignments']
            ]
            
            return ClusteringResult(
                total_candidates=result_data['total_candidates'],
                num_clusters=result_data['num_clusters'],
                silhouette_score=result_data['silhouette_score'],
                clusters=clusters,
                candidate_assignments=candidate_assignments,
                processing_time_ms=result_data['processing_time_ms'],
                model_parameters=self._convert_model_parameters(result_data['model_parameters'])
            )
            
        except Exception as e:
            logger.error(f"Error entrenando clustering jerárquico: {str(e)}")
            return None
    
    @strawberry.mutation
    def train_dbscan_clustering(self, 
                              eps: Optional[float] = 0.5,
                              min_samples: Optional[int] = 5) -> Optional[ClusteringResult]:
        """Entrena modelo usando DBSCAN"""
        try:
            result_data = train_candidate_clustering(
                algorithm="dbscan",
                eps=eps or 0.5,
                min_samples=min_samples or 5
            )
            
            # Convertir resultado
            from app.schemas.ml_schemas import ClusterInfo, CandidateClusterResult
            
            clusters = [
                ClusterInfo(
                    cluster_id=cluster['cluster_id'],
                    cluster_name=cluster['cluster_name'],
                    candidate_count=cluster['candidate_count'],
                    description=cluster['description'],
                    key_characteristics=cluster['key_characteristics'],
                    avg_experience_years=cluster['avg_experience_years'],
                    avg_salary_expectation=cluster['avg_salary_expectation'],
                    common_skills=cluster['common_skills'],
                    common_industries=cluster['common_industries'],
                    education_levels=cluster['education_levels']
                )
                for cluster in result_data['clusters']
            ]
            
            candidate_assignments = [
                CandidateClusterResult(
                    candidate_id=assignment['candidate_id'],
                    candidate_name=assignment['candidate_name'],
                    cluster_id=assignment['cluster_id'],
                    cluster_name=assignment['cluster_name'],
                    similarity_score=assignment['similarity_score'],
                    distance_to_centroid=assignment['distance_to_centroid'],
                    profile_summary=assignment['profile_summary']
                )
                for assignment in result_data['candidate_assignments']
            ]
            
            return ClusteringResult(
                total_candidates=result_data['total_candidates'],
                num_clusters=result_data['num_clusters'],
                silhouette_score=result_data['silhouette_score'],
                clusters=clusters,
                candidate_assignments=candidate_assignments,
                processing_time_ms=result_data['processing_time_ms'],
                model_parameters=self._convert_model_parameters(result_data['model_parameters'])
            )
            
        except Exception as e:
            logger.error(f"Error entrenando DBSCAN: {str(e)}")
            return None
    
    @strawberry.mutation
    def reset_clustering_model(self) -> bool:
        """Resetea el modelo de clustering"""
        try:
            clustering_service.clustering_model = None
            clustering_service.processed_data = None
            clustering_service.original_data = None
            clustering_service.cluster_labels = None
            clustering_service.cluster_centers = None
            clustering_service.is_model_trained = False
            clustering_service.silhouette_score_value = 0.0
            
            logger.info("Modelo de clustering reseteado")
            return True
            
        except Exception as e:
            logger.error(f"Error reseteando modelo: {str(e)}")
            return False
    
    @strawberry.mutation
    def add_candidate_to_dataset(self, candidate: CandidateProfileInput) -> bool:
        """Agrega un nuevo candidato al dataset (simulado)"""
        try:
            # En una implementación real, esto agregaría el candidato al CSV o base de datos
            # Por ahora, solo simulamos que se agregó correctamente
            logger.info(f"Candidato {candidate.nombre} agregado al dataset (simulado)")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando candidato: {str(e)}")
            return False