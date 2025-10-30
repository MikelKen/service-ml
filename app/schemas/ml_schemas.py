"""
Schemas para el sistema de predicción de contratación
"""
import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime


@strawberry.type
class JobApplication:
    """Esquema para una postulación de trabajo"""
    nombre: str
    experience_years: int
    nivel_educacion: str
    habilidades: str
    idiomas: str
    certificaciones: Optional[str] = None
    puesto_actual: str
    industria: str
    url_cv: Optional[str] = None
    fecha_postulacion: Optional[datetime] = None


@strawberry.type
class JobOffer:
    """Esquema para una oferta de trabajo"""
    titulo: str
    descripcion: str
    salario: float
    ubicacion: str
    requisitos: str
    fecha_publicacion: Optional[datetime] = None


@strawberry.type
class HiringPrediction:
    """Resultado de predicción de contratación"""
    prediction: int
    probability: float
    confidence_level: str
    recommendation: str
    model_used: str


@strawberry.type
class FeatureImportance:
    """Importancia de una feature"""
    feature_name: str
    importance: float


@strawberry.type
class PredictionResult:
    """Resultado completo de predicción"""
    hiring_prediction: HiringPrediction
    feature_importance: List[FeatureImportance]
    processing_time_ms: float


@strawberry.type
class ModelInfo:
    """Información del modelo"""
    model_name: str
    is_loaded: bool
    last_trained: Optional[str] = None
    version: str = "1.0.0"


@strawberry.type
class BatchPredictionResult:
    """Resultado de predicción en lote"""
    total_applications: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[PredictionResult]


@strawberry.input
class JobApplicationInput:
    """Input para postulación de trabajo"""
    nombre: str
    experience_years: int
    nivel_educacion: str
    habilidades: str
    idiomas: str
    certificaciones: Optional[str] = ""
    puesto_actual: str
    industria: str
    url_cv: Optional[str] = ""
    fecha_postulacion: Optional[str] = None


@strawberry.input
class JobOfferInput:
    """Input para oferta de trabajo"""
    titulo: str
    descripcion: str
    salario: float
    ubicacion: str
    requisitos: str
    fecha_publicacion: Optional[str] = None


@strawberry.input
class PredictionInput:
    """Input completo para predicción"""
    application: JobApplicationInput
    job_offer: JobOfferInput


@strawberry.type
class TrainingStatus:
    """Estado del entrenamiento del modelo"""
    is_training: bool
    progress: float
    status_message: str
    estimated_completion: Optional[str] = None


@strawberry.type
class ModelMetrics:
    """Métricas del modelo"""
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float


@strawberry.type
class DatasetInfo:
    """Información del dataset"""
    total_records: int
    positive_class_count: int
    negative_class_count: int
    class_balance_ratio: float
    last_updated: Optional[str] = None


# ============== CLUSTERING SCHEMAS ==============

@strawberry.type
class ModelParameter:
    """Parámetro del modelo"""
    key: str
    value: str


@strawberry.type
class DistributionItem:
    """Item de distribución"""
    name: str
    count: int


@strawberry.type
class CandidateProfile:
    """Perfil de candidato para clustering"""
    id: int
    nombre: str
    edad: int
    experience_years: int
    nivel_educacion: str
    area_especialidad: str
    habilidades_tecnicas: str
    habilidades_blandas: str
    idiomas: str
    certificaciones: str
    salario_esperado: float
    disponibilidad_viajar: str
    modalidad_trabajo: str
    ubicacion: str
    industria_experiencia: str
    puesto_actual: str
    liderazgo_equipos: int
    proyectos_completados: int
    educacion_continua: str
    redes_profesionales: str


@strawberry.type
class ClusterInfo:
    """Información de un cluster"""
    cluster_id: int
    cluster_name: str
    candidate_count: int
    description: str
    key_characteristics: List[str]
    avg_experience_years: float
    avg_salary_expectation: float
    common_skills: List[str]
    common_industries: List[str]
    education_levels: List[str]


@strawberry.type
class CandidateClusterResult:
    """Resultado del clustering para un candidato"""
    candidate_id: int
    candidate_name: str
    cluster_id: int
    cluster_name: str
    similarity_score: float
    distance_to_centroid: float
    profile_summary: str


@strawberry.type
class ClusteringResult:
    """Resultado completo del clustering"""
    total_candidates: int
    num_clusters: int
    silhouette_score: float
    clusters: List[ClusterInfo]
    candidate_assignments: List[CandidateClusterResult]
    processing_time_ms: float
    model_parameters: List[ModelParameter]


@strawberry.type
class SimilarCandidates:
    """Candidatos similares a uno dado"""
    reference_candidate: CandidateProfile
    similar_candidates: List[CandidateClusterResult]
    similarity_criteria: List[str]
    total_found: int


@strawberry.type
class ClusterAnalytics:
    """Analíticas de clustering"""
    cluster_distribution: List[DistributionItem]
    skill_frequency: List[DistributionItem]
    industry_distribution: List[DistributionItem]
    education_distribution: List[DistributionItem]
    salary_ranges_by_cluster: List[ModelParameter]
    experience_ranges_by_cluster: List[ModelParameter]


@strawberry.type
class ClusteringModelInfo:
    """Información del modelo de clustering"""
    is_trained: bool
    num_features: int
    silhouette_score: float
    model_type: str
    data_size: int


@strawberry.input
class CandidateProfileInput:
    """Input para perfil de candidato"""
    nombre: str
    edad: int
    experience_years: int
    nivel_educacion: str
    area_especialidad: str
    habilidades_tecnicas: str
    habilidades_blandas: str
    idiomas: str
    certificaciones: Optional[str] = ""
    salario_esperado: float
    disponibilidad_viajar: str
    modalidad_trabajo: str
    ubicacion: str
    industria_experiencia: str
    puesto_actual: str
    liderazgo_equipos: int
    proyectos_completados: int
    educacion_continua: str
    redes_profesionales: str


@strawberry.input
class ClusteringParameters:
    """Parámetros para clustering"""
    n_clusters: Optional[int] = None  # Si es None, se determina automáticamente
    algorithm: Optional[str] = "kmeans"  # kmeans, hierarchical, dbscan
    max_clusters: Optional[int] = 10
    min_samples: Optional[int] = 5  # Para DBSCAN
    eps: Optional[float] = 0.5  # Para DBSCAN
    linkage: Optional[str] = "ward"  # Para clustering jerárquico


@strawberry.input
class SimilaritySearchInput:
    """Input para búsqueda de similitud"""
    candidate_profile: CandidateProfileInput
    max_results: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.7
    weight_factors: Optional[List[ModelParameter]] = None  # Pesos para diferentes características