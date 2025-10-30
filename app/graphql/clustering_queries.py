"""
GraphQL Queries para Clustering de Candidatos
"""
import strawberry
from typing import List, Optional
import logging

from app.schemas.ml_schemas import (
    ClusteringResult, SimilarCandidates, ClusterAnalytics,
    CandidateProfile, ClusterInfo, CandidateClusterResult, ClusteringModelInfo,
    ModelParameter, DistributionItem
)
from app.services.clustering_service import (
    get_clustering_results, find_similar_candidates_to_profile,
    get_clustering_analytics, clustering_service
)

logger = logging.getLogger(__name__)


@strawberry.type
class ClusteringQueries:
    """Queries relacionadas con clustering de candidatos"""
    
    @strawberry.field
    def get_clustering_results(self) -> Optional[ClusteringResult]:
        """Obtiene los resultados del clustering actual"""
        try:
            if not clustering_service.is_model_trained:
                logger.warning("Modelo de clustering no entrenado")
                return None
            
            result_data = get_clustering_results()
            
            # Convertir a objetos Strawberry
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
                model_parameters=[
                    ModelParameter(key=str(k), value=str(v))
                    for k, v in result_data['model_parameters'].items()
                ]
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo resultados de clustering: {str(e)}")
            return None
    
    @strawberry.field
    def get_clusters_info(self) -> List[ClusterInfo]:
        """Obtiene información de todos los clusters"""
        try:
            if not clustering_service.is_model_trained:
                logger.warning("Modelo de clustering no entrenado")
                return []
            
            clusters_data = clustering_service.analyze_clusters()
            
            return [
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
                for cluster in clusters_data
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo información de clusters: {str(e)}")
            return []
    
    @strawberry.field
    def get_cluster_by_id(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Obtiene información de un cluster específico"""
        try:
            clusters = self.get_clusters_info()
            for cluster in clusters:
                if cluster.cluster_id == cluster_id:
                    return cluster
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo cluster {cluster_id}: {str(e)}")
            return None
    
    @strawberry.field
    def get_candidates_in_cluster(self, cluster_id: int) -> List[CandidateClusterResult]:
        """Obtiene candidatos en un cluster específico"""
        try:
            if not clustering_service.is_model_trained:
                logger.warning("Modelo de clustering no entrenado")
                return []
            
            assignments = clustering_service.get_candidate_cluster_assignments()
            
            cluster_candidates = [
                CandidateClusterResult(
                    candidate_id=assignment['candidate_id'],
                    candidate_name=assignment['candidate_name'],
                    cluster_id=assignment['cluster_id'],
                    cluster_name=assignment['cluster_name'],
                    similarity_score=assignment['similarity_score'],
                    distance_to_centroid=assignment['distance_to_centroid'],
                    profile_summary=assignment['profile_summary']
                )
                for assignment in assignments
                if assignment['cluster_id'] == cluster_id
            ]
            
            return cluster_candidates
            
        except Exception as e:
            logger.error(f"Error obteniendo candidatos del cluster {cluster_id}: {str(e)}")
            return []
    
    @strawberry.field
    def find_similar_candidates(self, 
                               candidate_name: str,
                               experience_years: int,
                               specialty_area: str,
                               technical_skills: str,
                               soft_skills: str,
                               education_level: str,
                               expected_salary: float,
                               max_results: Optional[int] = 10) -> Optional[SimilarCandidates]:
        """Encuentra candidatos similares a un perfil dado"""
        try:
            if not clustering_service.is_model_trained:
                logger.warning("Modelo de clustering no entrenado")
                return None
            
            # Crear perfil de búsqueda
            candidate_profile = {
                'nombre': candidate_name,
                'edad': 30,  # Valor por defecto
                'años_experiencia': experience_years,
                'nivel_educacion': education_level,
                'area_especialidad': specialty_area,
                'habilidades_tecnicas': technical_skills,
                'habilidades_blandas': soft_skills,
                'idiomas': 'Español,Inglés',  # Valor por defecto
                'certificaciones': '',
                'salario_esperado': expected_salary,
                'disponibilidad_viajar': 'No',  # Valor por defecto
                'modalidad_trabajo': 'Híbrido',  # Valor por defecto
                'ubicacion': 'Santa Cruz',  # Valor por defecto
                'industria_experiencia': 'Tecnología',  # Valor por defecto
                'puesto_actual': specialty_area,
                'liderazgo_equipos': 0,  # Valor por defecto
                'proyectos_completados': experience_years * 3,  # Estimación
                'educacion_continua': 'Sí',  # Valor por defecto
                'redes_profesionales': 'LinkedIn'  # Valor por defecto
            }
            
            result_data = find_similar_candidates_to_profile(
                candidate_profile, max_results or 10
            )
            
            # Convertir reference_candidate
            reference = result_data['reference_candidate']
            reference_candidate = CandidateProfile(
                id=reference['id'],
                nombre=reference['nombre'],
                edad=reference['edad'],
                experience_years=reference['años_experiencia'],
                nivel_educacion=reference['nivel_educacion'],
                area_especialidad=reference['area_especialidad'],
                habilidades_tecnicas=reference['habilidades_tecnicas'],
                habilidades_blandas=reference['habilidades_blandas'],
                idiomas=reference['idiomas'],
                certificaciones=reference['certificaciones'],
                salario_esperado=reference['salario_esperado'],
                disponibilidad_viajar=reference['disponibilidad_viajar'],
                modalidad_trabajo=reference['modalidad_trabajo'],
                ubicacion=reference['ubicacion'],
                industria_experiencia=reference['industria_experiencia'],
                puesto_actual=reference['puesto_actual'],
                liderazgo_equipos=reference['liderazgo_equipos'],
                proyectos_completados=reference['proyectos_completados'],
                educacion_continua=reference['educacion_continua'],
                redes_profesionales=reference['redes_profesionales']
            )
            
            # Convertir similar_candidates
            similar_candidates = [
                CandidateClusterResult(
                    candidate_id=candidate['candidate_id'],
                    candidate_name=candidate['candidate_name'],
                    cluster_id=candidate['cluster_id'],
                    cluster_name=candidate['cluster_name'],
                    similarity_score=candidate['similarity_score'],
                    distance_to_centroid=candidate['distance_to_centroid'],
                    profile_summary=candidate['profile_summary']
                )
                for candidate in result_data['similar_candidates']
            ]
            
            return SimilarCandidates(
                reference_candidate=reference_candidate,
                similar_candidates=similar_candidates,
                similarity_criteria=result_data['similarity_criteria'],
                total_found=result_data['total_found']
            )
            
        except Exception as e:
            logger.error(f"Error encontrando candidatos similares: {str(e)}")
            return None
    
    @strawberry.field
    def get_clustering_analytics(self) -> Optional[ClusterAnalytics]:
        """Obtiene analíticas de clustering"""
        try:
            if not clustering_service.is_model_trained:
                logger.warning("Modelo de clustering no entrenado")
                return None
            
            analytics_data = get_clustering_analytics()
            
            # Convertir diccionarios a listas de objetos
            cluster_distribution = [
                DistributionItem(name=k, count=v) 
                for k, v in analytics_data['cluster_distribution'].items()
            ]
            
            skill_frequency = [
                DistributionItem(name=k, count=v)
                for k, v in analytics_data['skill_frequency'].items()
            ]
            
            industry_distribution = [
                DistributionItem(name=k, count=v)
                for k, v in analytics_data['industry_distribution'].items()
            ]
            
            education_distribution = [
                DistributionItem(name=k, count=v)
                for k, v in analytics_data['education_distribution'].items()
            ]
            
            salary_ranges = [
                ModelParameter(key=k, value=v)
                for k, v in analytics_data['salary_ranges_by_cluster'].items()
            ]
            
            experience_ranges = [
                ModelParameter(key=k, value=v)
                for k, v in analytics_data['experience_ranges_by_cluster'].items()
            ]
            
            return ClusterAnalytics(
                cluster_distribution=cluster_distribution,
                skill_frequency=skill_frequency,
                industry_distribution=industry_distribution,
                education_distribution=education_distribution,
                salary_ranges_by_cluster=salary_ranges,
                experience_ranges_by_cluster=experience_ranges
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo analíticas: {str(e)}")
            return None
    
    @strawberry.field
    def is_clustering_model_trained(self) -> bool:
        """Verifica si el modelo de clustering está entrenado"""
        return clustering_service.is_model_trained
    
    @strawberry.field
    def get_clustering_model_info(self) -> ClusteringModelInfo:
        """Obtiene información del modelo de clustering"""
        try:
            return ClusteringModelInfo(
                is_trained=clustering_service.is_model_trained,
                num_features=len(clustering_service.feature_names) if clustering_service.feature_names else 0,
                silhouette_score=clustering_service.silhouette_score_value,
                model_type=type(clustering_service.clustering_model).__name__ if clustering_service.clustering_model else 'None',
                data_size=len(clustering_service.original_data) if clustering_service.original_data is not None else 0
            )
        except Exception as e:
            logger.error(f"Error obteniendo info del modelo: {str(e)}")
            return ClusteringModelInfo(
                is_trained=False,
                num_features=0,
                silhouette_score=0.0,
                model_type='None',
                data_size=0
            )