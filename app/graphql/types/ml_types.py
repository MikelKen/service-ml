"""
Tipos de GraphQL para Machine Learning - Sistema de Compatibilidad
"""
import strawberry
from typing import Dict, List, Optional, Any
from datetime import datetime


@strawberry.input
class CustomCandidateData:
    """Datos personalizados de un candidato para predicción"""
    anios_experiencia: int
    nivel_educacion: str
    habilidades: str
    idiomas: Optional[str] = ""
    certificaciones: Optional[str] = ""
    puesto_actual: Optional[str] = ""


@strawberry.input
class CustomJobOfferData:
    """Datos personalizados de una oferta laboral para predicción"""
    titulo: str
    salario: float
    ubicacion: str
    requisitos: str


@strawberry.input
class CustomCompatibilityPredictionInput:
    """Input para predicción con datos personalizados (no desde BD)"""
    candidate_data: CustomCandidateData
    offer_data: CustomJobOfferData


@strawberry.input
class CompatibilityPredictionInput:
    """Input para predicción de compatibilidad candidato-oferta"""
    candidate_id: str
    offer_id: str


@strawberry.input
class BatchCompatibilityInput:
    """Input para predicción batch de compatibilidad"""
    pairs: List[CompatibilityPredictionInput]


@strawberry.input
class TopCandidatesInput:
    """Input para obtener top candidatos para una oferta"""
    offer_id: str
    top_n: Optional[int] = 10


@strawberry.type
class CompatibilityPrediction:
    """Resultado detallado de predicción de compatibilidad"""
    candidate_id: str
    offer_id: str
    probability: float
    prediction: bool
    confidence: str
    
    # Información descriptiva adicional
    probability_percentage: Optional[str] = None
    compatibility_level: Optional[str] = None
    recommendation: Optional[str] = None
    decision_factors: Optional[str] = None
    
    # Análisis detallado
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    
    # Información técnica
    ranking: Optional[int] = None
    model_used: Optional[str] = None
    prediction_date: Optional[str] = None
    confidence_score: Optional[float] = None
    
    # Meta información
    summary: Optional[str] = None
    detailed_analysis: Optional[str] = None
    error: Optional[str] = None


@strawberry.type
class BatchCompatibilityResult:
    """Resultado de predicción batch"""
    predictions: List[CompatibilityPrediction]
    total_processed: int
    success_count: int
    error_count: int
    processing_time: Optional[float] = None


@strawberry.type
class TrainingMetrics:
    """Métricas de entrenamiento"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None


@strawberry.type
class DataSummary:
    """Resumen de datos de entrenamiento"""
    total_samples: Optional[int] = None
    positive_samples: Optional[int] = None
    negative_samples: Optional[int] = None
    features_count: Optional[int] = None


@strawberry.type
class ModelTrainingResult:
    """Resultado del entrenamiento de modelo"""
    success: bool
    message: str
    best_model: Optional[str] = None
    metrics: Optional[TrainingMetrics] = None
    training_time: Optional[float] = None
    data_summary: Optional[DataSummary] = None


@strawberry.type
class ModelMetrics:
    """Métricas del modelo"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


@strawberry.type
class ModelInfo:
    """Información del modelo ML"""
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    is_loaded: bool = False
    training_date: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    feature_importance_count: Optional[int] = None
    top_features: Optional[List[str]] = None


@strawberry.type
class FeatureImportance:
    """Importancia de features del modelo"""
    feature_name: str
    importance: float


@strawberry.type
class ModelFeatureImportance:
    """Lista de importancia de features"""
    features: List[FeatureImportance]
    total_features: int


@strawberry.type
class PredictionFactors:
    """Factores de predicción"""
    experience_match: Optional[float] = None
    skills_overlap: Optional[float] = None
    education_fit: Optional[float] = None
    location_match: Optional[float] = None


@strawberry.type
class PredictionExplanation:
    """Explicación detallada de una predicción"""
    prediction: CompatibilityPrediction
    key_factors: PredictionFactors
    feature_importance: List[FeatureImportance]
    recommendation: str


@strawberry.type
class TrainingDataSummary:
    """Resumen de datos de entrenamiento"""
    total_records: int
    positive_samples: int
    negative_samples: int
    features_count: int
    data_quality_score: Optional[float] = None


@strawberry.type
class ModelPerformanceMetrics:
    """Métricas de rendimiento del modelo"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None


@strawberry.input
class TrainingConfigInput:
    """Configuración para entrenamiento de modelo"""
    positive_samples_ratio: Optional[float] = 0.3
    negative_samples_multiplier: Optional[int] = 2
    enable_hyperparameter_tuning: Optional[bool] = False
    cross_validation_folds: Optional[int] = 5


# Tipos legacy mantenidos para compatibilidad
@strawberry.type
class ApplicationInput:
    """Input de datos de postulación (legacy)"""
    nombre: str
    experience_years: int
    nivel_educacion: str
    habilidades: str
    idiomas: str
    certificaciones: Optional[str] = ""
    puesto_actual: str
    industria: str
    url_cv: Optional[str] = ""
    fecha_postulacion: datetime


@strawberry.type
class JobOfferInput:
    """Input de datos de oferta laboral (legacy)"""
    titulo: str
    descripcion: str
    salario: float
    ubicacion: str
    requisitos: str
    fecha_publicacion: datetime


@strawberry.input
class PredictionInput:
    """Input completo para predicción (legacy)"""
    application: ApplicationInput
    job_offer: JobOfferInput


@strawberry.type
class MetadataInfo:
    """Información de metadatos"""
    model_version: Optional[str] = None
    features_used: Optional[int] = None
    processing_time_ms: Optional[float] = None


@strawberry.type
class PredictionResult:
    """Resultado de predicción individual (legacy)"""
    probability: float
    prediction: bool
    confidence: str
    metadata: Optional[MetadataInfo] = None


@strawberry.type
class BatchPredictionResult:
    """Resultado de predicción en lote (legacy)"""
    predictions: List[PredictionResult]
    total_processed: int
    success_count: int
    error_count: int
    processing_time: float


@strawberry.type
class TrainingStatus:
    """Estado del entrenamiento"""
    is_training: bool
    progress: float
    current_step: str
    estimated_time_remaining: Optional[int] = None
    error_message: Optional[str] = None


@strawberry.type
class TargetDistribution:
    """Distribución del target"""
    positive: int
    negative: int


@strawberry.type
class MissingValues:
    """Valores faltantes por campo"""
    total_missing: int
    fields_with_missing: List[str]


@strawberry.type
class DataTypes:
    """Tipos de datos"""
    numeric_fields: List[str]
    text_fields: List[str]
    date_fields: List[str]


@strawberry.type
class DatasetInfo:
    """Información del dataset"""
    total_records: int
    features_count: int
    target_distribution: TargetDistribution
    missing_values: MissingValues
    data_types: DataTypes
    last_updated: Optional[datetime] = None