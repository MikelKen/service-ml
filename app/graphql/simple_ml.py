"""
GraphQL simple para predicciones de contratación y clustering
"""
import strawberry
from typing import Optional, List
import logging

# Import clustering schemas and functions
from app.schemas.ml_schemas import (
    ClusteringResult, SimilarCandidates, ClusterAnalytics,
    ClusterInfo, CandidateClusterResult, ClusteringParameters, ClusteringModelInfo
)
from app.graphql.clustering_queries import ClusteringQueries
from app.graphql.clustering_mutations import ClusteringMutations

logger = logging.getLogger(__name__)


@strawberry.type
class HiringPrediction:
    """Resultado de predicción de contratación"""
    prediction: int
    probability: float
    confidence_level: str
    recommendation: str
    model_used: str


@strawberry.type
class ModelStatus:
    """Estado del modelo"""
    is_loaded: bool
    model_name: str
    version: str


@strawberry.type
class Query:
    """Consultas GraphQL"""
    
    # ============ HIRING PREDICTION QUERIES ============
    @strawberry.field
    def model_status(self) -> ModelStatus:
        """Obtiene el estado del modelo"""
        try:
            from app.services.ml_service import ml_service
            info = ml_service.get_model_info()
            return ModelStatus(
                is_loaded=info.get('is_loaded', False),
                model_name=info.get('model_name', 'Unknown'),
                version=info.get('version', '1.0.0')
            )
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return ModelStatus(
                is_loaded=False,
                model_name="Error",
                version="0.0.0"
            )
    
    @strawberry.field
    def health_check(self) -> str:
        """Verificación de salud"""
        return "ML Service is healthy"
    
    # ============ CLUSTERING QUERIES ============
    @strawberry.field
    def clustering_results(self) -> Optional[ClusteringResult]:
        """Obtiene resultados del clustering actual"""
        clustering_queries = ClusteringQueries()
        return clustering_queries.get_clustering_results()
    
    @strawberry.field
    def clusters_info(self) -> List[ClusterInfo]:
        """Obtiene información de todos los clusters"""
        clustering_queries = ClusteringQueries()
        return clustering_queries.get_clusters_info()
    
    @strawberry.field
    def cluster_by_id(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Obtiene información de un cluster específico"""
        clustering_queries = ClusteringQueries()
        return clustering_queries.get_cluster_by_id(cluster_id)
    
    @strawberry.field
    def candidates_in_cluster(self, cluster_id: int) -> List[CandidateClusterResult]:
        """Obtiene candidatos en un cluster específico"""
        clustering_queries = ClusteringQueries()
        return clustering_queries.get_candidates_in_cluster(cluster_id)
    
    @strawberry.field
    def similar_candidates(self, 
                          candidate_name: str,
                          experience_years: int,
                          specialty_area: str,
                          technical_skills: str,
                          soft_skills: str,
                          education_level: str,
                          expected_salary: float,
                          max_results: Optional[int] = 10) -> Optional[SimilarCandidates]:
        """Encuentra candidatos similares"""
        clustering_queries = ClusteringQueries()
        return clustering_queries.find_similar_candidates(
            candidate_name=candidate_name,
            experience_years=experience_years,
            specialty_area=specialty_area,
            technical_skills=technical_skills,
            soft_skills=soft_skills,
            education_level=education_level,
            expected_salary=expected_salary,
            max_results=max_results
        )
    
    @strawberry.field
    def clustering_analytics(self) -> Optional[ClusterAnalytics]:
        """Obtiene analíticas de clustering"""
        clustering_queries = ClusteringQueries()
        return clustering_queries.get_clustering_analytics()
    
    @strawberry.field
    def is_clustering_trained(self) -> bool:
        """Verifica si el modelo de clustering está entrenado"""
        clustering_queries = ClusteringQueries()
        return clustering_queries.is_clustering_model_trained()
    
    @strawberry.field
    def clustering_model_info(self) -> ClusteringModelInfo:
        """Obtiene información del modelo de clustering"""
        clustering_queries = ClusteringQueries()
        return clustering_queries.get_clustering_model_info()


@strawberry.type
class Mutation:
    """Mutaciones GraphQL"""
    
    # ============ HIRING PREDICTION MUTATIONS ============
    @strawberry.mutation
    def predict_hiring(
        self,
        nombre: str,
        experience_years: int,
        nivel_educacion: str,
        habilidades: str,
        idiomas: str,
        certificaciones: Optional[str] = None,
        puesto_actual: Optional[str] = None,
        industria: Optional[str] = None,
        titulo: str = "Desarrollador",
        descripcion: str = "Desarrollo de software",
        salario: float = 1000,
        ubicacion: str = "Santa Cruz",
        requisitos: str = "Experiencia en desarrollo",
        fecha_postulacion: Optional[str] = None,
        fecha_publicacion: Optional[str] = None
    ) -> HiringPrediction:
        """Predice la probabilidad de contratación"""
        
        try:
            # Importar el predictor simple
            from simple_predictor import SimpleHiringPredictor
            import os
            
            # Usar modelo entrenado si existe, sino usar datos por defecto
            model_path = "trained_models/simple_hiring_model.pkl"
            
            if not os.path.exists(model_path):
                logger.warning("Modelo no encontrado, usando predicción por defecto")
                return HiringPrediction(
                    prediction=1,
                    probability=0.75,
                    confidence_level="Alta",
                    recommendation="Recomendado para entrevista",
                    model_used="DefaultPredictor"
                )
            
            # Crear datos de entrada
            application_data = {
                'nombre': nombre,
                'años_experiencia': experience_years,
                'nivel_educacion': nivel_educacion,
                'habilidades': habilidades,
                'idiomas': idiomas,
                'certificaciones': certificaciones or "",
                'puesto_actual': puesto_actual or "Desarrollador",
                'industria': industria or "Tecnología",
                'titulo': titulo,
                'descripcion': descripcion,
                'salario': salario,
                'ubicacion': ubicacion,
                'requisitos': requisitos,
                'fecha_postulacion': fecha_postulacion or '2024-01-15',
                'fecha_publicacion': fecha_publicacion or '2024-01-10'
            }
            
            # Realizar predicción
            predictor = SimpleHiringPredictor(model_path)
            result = predictor.predict(application_data)
            
            return HiringPrediction(
                prediction=result['prediction'],
                probability=result['probability'],
                confidence_level=result['confidence_level'],
                recommendation=result['recommendation'],
                model_used=result['model_used']
            )
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            # Retornar predicción por defecto en caso de error
            return HiringPrediction(
                prediction=0,
                probability=0.30,
                confidence_level="Baja",
                recommendation=f"Error en predicción: {str(e)}",
                model_used="ErrorHandler"
            )
    
    # ============ CLUSTERING MUTATIONS ============
    @strawberry.mutation
    async def train_clustering(self, 
                              parameters: Optional[ClusteringParameters] = None) -> Optional[ClusteringResult]:
        """Entrena el modelo de clustering"""
        clustering_mutations = ClusteringMutations()
        return await clustering_mutations.train_clustering_model(parameters)
    
    @strawberry.mutation
    def train_kmeans_clustering(self, n_clusters: Optional[int] = None) -> Optional[ClusteringResult]:
        """Entrena clustering con K-Means"""
        clustering_mutations = ClusteringMutations()
        return clustering_mutations.retrain_clustering_with_kmeans(n_clusters)
    
    @strawberry.mutation
    def train_hierarchical_clustering(self, 
                                    n_clusters: Optional[int] = None,
                                    linkage: Optional[str] = "ward") -> Optional[ClusteringResult]:
        """Entrena clustering jerárquico"""
        clustering_mutations = ClusteringMutations()
        return clustering_mutations.train_hierarchical_clustering(n_clusters, linkage)
    
    @strawberry.mutation
    def train_dbscan_clustering(self, 
                              eps: Optional[float] = 0.5,
                              min_samples: Optional[int] = 5) -> Optional[ClusteringResult]:
        """Entrena clustering con DBSCAN"""
        clustering_mutations = ClusteringMutations()
        return clustering_mutations.train_dbscan_clustering(eps, min_samples)
    
    @strawberry.mutation
    def reset_clustering(self) -> bool:
        """Resetea el modelo de clustering"""
        clustering_mutations = ClusteringMutations()
        return clustering_mutations.reset_clustering_model()