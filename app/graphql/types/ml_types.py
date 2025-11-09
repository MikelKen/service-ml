"""
Tipos de GraphQL para Machine Learning - Sistema de Compatibilidad
"""
import strawberry
from typing import List, Optional, Any
from datetime import datetime

# Tipos auxiliares para compatibilidad con GraphQL
@strawberry.type
class KeyValuePair:
    """Par clave-valor para representar diccionarios en GraphQL"""
    key: str
    value: str

@strawberry.type
class KeyIntValuePair:
    """Par clave-valor para diccionarios con valores enteros"""
    key: str
    value: int

@strawberry.type
class KeyFloatValuePair:
    """Par clave-valor para diccionarios con valores flotantes"""
    key: str
    value: float


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


# === TIPOS PARA MODELO SEMI-SUPERVISADO ===

@strawberry.input
class SemiSupervisedTrainingInput:
    """Input para entrenamiento semi-supervisado"""
    model_types: Optional[List[str]] = None  # ['label_propagation', 'label_spreading', 'self_training']
    save_to_mongo: Optional[bool] = True
    validation_split: Optional[float] = 0.2


@strawberry.input
class PostulacionEstadoPredictionInput:
    """Input para predicción de estado de postulación"""
    postulacion_id: Optional[str] = None
    # Datos manuales si no se especifica ID
    nombre: Optional[str] = None
    anios_experiencia: Optional[int] = None
    nivel_educacion: Optional[str] = None
    habilidades: Optional[str] = None
    idiomas: Optional[str] = None
    certificaciones: Optional[str] = None
    puesto_actual: Optional[str] = None
    # Datos de la oferta
    oferta_titulo: Optional[str] = None
    oferta_salario: Optional[float] = None
    oferta_requisitos: Optional[str] = None
    empresa_rubro: Optional[str] = None


@strawberry.input
class BatchEstadoPredictionInput:
    """Input para predicción batch de estados"""
    postulaciones: List[PostulacionEstadoPredictionInput]
    model_type: Optional[str] = None  # Tipo de modelo a usar


@strawberry.type
class PostulacionEstadoPrediction:
    """Resultado de predicción de estado de postulación"""
    postulacion_id: Optional[str] = None
    predicted_estado: str
    confidence: float
    probability_distribution: Optional[List[KeyFloatValuePair]] = None
    
    # Información adicional
    confidence_level: Optional[str] = None  # 'high', 'medium', 'low'
    model_used: Optional[str] = None
    prediction_timestamp: Optional[str] = None
    
    # Factores de decisión
    key_factors: Optional[List[str]] = None
    experience_score: Optional[float] = None
    skills_score: Optional[float] = None
    education_score: Optional[float] = None
    
    # Metadatos
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


@strawberry.type
class BatchEstadoPredictionResult:
    """Resultado de predicción batch de estados"""
    predictions: List[PostulacionEstadoPrediction]
    total_processed: int
    success_count: int
    error_count: int
    model_used: str
    processing_time: float
    summary_stats: Optional[List[KeyIntValuePair]] = None


@strawberry.type
class SemiSupervisedModelInfo:
    """Información de modelo semi-supervisado"""
    model_type: str
    is_trained: bool
    training_timestamp: Optional[str] = None
    
    # Métricas de entrenamiento
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    
    # Estadísticas de datos
    labeled_samples: Optional[int] = None
    unlabeled_samples: Optional[int] = None
    total_samples: Optional[int] = None
    classes: Optional[List[str]] = None
    
    # Predicciones en datos no etiquetados
    unlabeled_predictions_count: Optional[int] = None
    prediction_confidence_mean: Optional[float] = None
    prediction_distribution: Optional[List[KeyIntValuePair]] = None
    
    # Archivos
    model_path: Optional[str] = None
    metrics_available: Optional[bool] = None


@strawberry.type
class SemiSupervisedTrainingResult:
    """Resultado del entrenamiento semi-supervisado"""
    success: bool
    message: str
    
    # Información de datos
    total_samples: int
    labeled_samples: int
    unlabeled_samples: int
    features_count: int
    classes_found: List[str]
    
    # Modelos entrenados
    models_trained: List[str]
    models_info: List[SemiSupervisedModelInfo]
    
    # Mejor modelo
    best_model_type: Optional[str] = None
    best_model_score: Optional[float] = None
    
    # Métricas generales
    training_time: Optional[float] = None
    files_generated: Optional[List[str]] = None
    
    # Resumen de predicciones
    unlabeled_predictions_generated: Optional[int] = None
    high_confidence_predictions: Optional[int] = None
    
    # Errores
    errors: Optional[List[str]] = None


@strawberry.type
class SemiSupervisedDataSummary:
    """Resumen de datos para semi-supervisado"""
    total_postulaciones: int
    labeled_postulaciones: int
    unlabeled_postulaciones: int
    labeled_percentage: float
    
    # Distribución de estados
    estado_distribution: List[KeyIntValuePair]
    
    # Estadísticas de calidad
    missing_data_percentage: Optional[float] = None
    completeness_score: Optional[float] = None
    
    # Recomendaciones
    can_train_semi_supervised: bool
    recommendations: List[str]
    
    # Estadísticas por tabla
    table_stats: Optional[List[KeyValuePair]] = None


@strawberry.type
class ModelComparisonResult:
    """Resultado de comparación de modelos"""
    model_type: str
    performance_metrics: List[KeyFloatValuePair]
    prediction_quality: List[KeyFloatValuePair]
    training_efficiency: List[KeyFloatValuePair]
    recommendation_score: float
    pros: List[str]
    cons: List[str]


@strawberry.type
class SemiSupervisedModelComparison:
    """Comparación de modelos semi-supervisados"""
    comparison_timestamp: str
    models_compared: List[ModelComparisonResult]
    recommended_model: str
    summary: str
    detailed_analysis: Optional[str] = None


@strawberry.type
class PredictionConfidenceAnalysis:
    """Análisis de confianza de predicciones"""
    total_predictions: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    
    confidence_distribution: List[KeyFloatValuePair]
    
    # Recomendaciones
    reliable_predictions: int
    review_needed_predictions: int
    manual_verification_needed: int
    
    # Estadísticas por estado predicho
    confidence_by_estado: Optional[List[KeyValuePair]] = None


@strawberry.type
class UnlabeledDataInsights:
    """Insights de datos no etiquetados"""
    total_unlabeled: int
    
    # Distribución de predicciones
    predicted_estados: List[KeyIntValuePair]
    confidence_stats: PredictionConfidenceAnalysis
    
    # Patrones identificados
    common_patterns: Optional[List[str]] = None
    outliers_detected: Optional[int] = None
    
    # Recomendaciones de etiquetado
    priority_labeling_candidates: Optional[List[str]] = None
    labeling_strategy: Optional[str] = None


@strawberry.input
class RetrainModelInput:
    """Input para re-entrenar modelo"""
    model_type: Optional[str] = None  # Si no se especifica, usa el mejor modelo anterior
    include_new_predictions: Optional[bool] = False  # Incluir predicciones como pseudo-etiquetas
    confidence_threshold: Optional[float] = 0.8  # Umbral para pseudo-etiquetas


@strawberry.type
class RetrainModelResult:
    """Resultado de re-entrenamiento"""
    success: bool
    message: str
    model_type: str
    
    # Comparación con modelo anterior
    old_performance: Optional[List[KeyFloatValuePair]] = None
    new_performance: Optional[List[KeyFloatValuePair]] = None
    improvement: Optional[List[KeyFloatValuePair]] = None
    
    # Nuevos datos utilizados
    new_labeled_data: Optional[int] = None
    pseudo_labels_used: Optional[int] = None
    
    # Métricas
    training_time: float
    model_path: str
    
    recommendations: Optional[List[str]] = None