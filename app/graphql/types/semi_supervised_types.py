#!/usr/bin/env python3
"""
🔗 TIPOS GRAPHQL PARA MODELO SEMI-SUPERVISADO
Define tipos GraphQL para entrenamiento, predicciones y consultas del modelo semi-supervisado
"""

import strawberry
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enums para el modelo semi-supervisado
@strawberry.enum
class SemiSupervisedAlgorithm(Enum):
    LABEL_PROPAGATION = "label_propagation"
    LABEL_SPREADING = "label_spreading"
    SELF_TRAINING_RF = "self_training_rf"
    SELF_TRAINING_LR = "self_training_lr"
    SELF_TRAINING_GB = "self_training_gb"

@strawberry.enum
class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@strawberry.enum
class LabelQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    PREDICTED = "predicted"
    UNLABELED = "unlabeled"

@strawberry.enum
class PredictionStatus(Enum):
    REJECTED = 0
    ACCEPTED = 1
    UNKNOWN = -1

# Tipos de datos para features y métricas
@strawberry.type
class CompatibilityFeatures:
    skill_match_score: float
    experience_match: float
    education_match: float
    location_match: bool
    salary_expectation_match: float
    overall_compatibility: float

@strawberry.type
class CandidateFeatures:
    experiencia_normalizada: float
    nivel_educacion_encoded: int
    num_habilidades: int
    num_idiomas: int
    num_certificaciones: int
    profile_completeness: float

@strawberry.type
class OfferFeatures:
    salario_normalizado: float
    dias_desde_publicacion: int
    nivel_requisitos: str
    tipo_contrato: str
    modalidad_trabajo: str

@strawberry.type
class ModelMetrics:
    train_accuracy: float
    train_precision: float
    train_recall: float
    train_f1: float
    val_accuracy: Optional[float] = None
    val_precision: Optional[float] = None
    val_recall: Optional[float] = None
    val_f1: Optional[float] = None
    val_roc_auc: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    cv_f1_std: Optional[float] = None

@strawberry.type
class PseudoLabelStats:
    total_unlabeled: int
    positive_pseudo_labels: int
    negative_pseudo_labels: int
    mean_confidence: Optional[float] = None
    median_confidence: Optional[float] = None
    high_confidence_samples: Optional[int] = None
    low_confidence_samples: Optional[int] = None

@strawberry.type
class TrainingConfig:
    algorithm: SemiSupervisedAlgorithm
    labeled_samples: int
    unlabeled_samples: int
    test_samples: Optional[int] = None
    validation_split: float
    features_used: List[str]
    hyperparameters: strawberry.scalars.JSON

# Tipos principales del modelo
@strawberry.type
class SemiSupervisedPrediction:
    application_id: str
    candidate_id: str
    offer_id: str
    prediction: PredictionStatus
    probability: float
    confidence_level: ConfidenceLevel
    compatibility_score: float
    predicted_at: datetime
    model_algorithm: SemiSupervisedAlgorithm
    model_version: str

@strawberry.type
class ApplicationWithPrediction:
    application_id: str
    candidate_id: str
    offer_id: str
    fecha_postulacion: datetime
    estado_original: str
    is_labeled: bool
    label_quality: LabelQuality
    
    # Features de compatibilidad
    compatibility_features: CompatibilityFeatures
    
    # Predicción del modelo
    ml_prediction: Optional[PredictionStatus] = None
    ml_probability: Optional[float] = None
    ml_confidence: Optional[ConfidenceLevel] = None
    
    # Información del candidato
    candidate_name: str
    candidate_email: str
    candidate_experience_years: int
    candidate_skills: List[str]
    
    # Información de la oferta
    offer_title: str
    offer_company: str
    offer_salary: Optional[float] = None
    offer_location: str

@strawberry.type
class ModelTrainingResult:
    training_id: str
    algorithm: SemiSupervisedAlgorithm
    training_started_at: datetime
    training_completed_at: datetime
    training_time_seconds: float
    
    # Configuración del entrenamiento
    training_config: TrainingConfig
    
    # Métricas del modelo
    metrics: ModelMetrics
    
    # Estadísticas de pseudo-etiquetas
    pseudo_label_stats: PseudoLabelStats
    
    # Estado del entrenamiento
    success: bool
    error_message: Optional[str] = None
    
    # Rutas de archivos guardados
    model_path: Optional[str] = None
    preprocessor_path: Optional[str] = None

@strawberry.type
class ModelInfo:
    model_id: str
    algorithm: SemiSupervisedAlgorithm
    version: str
    created_at: datetime
    is_active: bool
    
    # Métricas del modelo
    performance_metrics: ModelMetrics
    
    # Información del dataset
    total_samples: int
    labeled_samples: int
    unlabeled_samples: int
    labeled_ratio: float
    
    # Distribución de clases
    positive_samples: int
    negative_samples: int
    
    # Features utilizadas
    n_features: int
    feature_categories: strawberry.scalars.JSON

@strawberry.type
class BatchPredictionResult:
    batch_id: str
    processed_at: datetime
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    
    # Estadísticas de predicciones
    high_confidence_predictions: int
    medium_confidence_predictions: int
    low_confidence_predictions: int
    
    # Distribución de predicciones
    predicted_positive: int
    predicted_negative: int
    
    # Resultados individuales
    predictions: List[SemiSupervisedPrediction]
    
    # Errores si los hay
    errors: List[str]

@strawberry.type
class FeatureImportance:
    feature_name: str
    importance_score: float
    category: str  # 'compatibility', 'candidate', 'offer', 'text'

@strawberry.type
class ModelAnalysis:
    model_id: str
    algorithm: SemiSupervisedAlgorithm
    
    # Importancia de features
    feature_importance: List[FeatureImportance]
    
    # Análisis de predicciones
    prediction_distribution: strawberry.scalars.JSON
    confidence_distribution: strawberry.scalars.JSON
    
    # Análisis de datos no etiquetados
    unlabeled_insights: strawberry.scalars.JSON
    
    # Recomendaciones para mejorar el modelo
    recommendations: List[str]

# Tipos de entrada (Input Types)
@strawberry.input
class TrainingParameters:
    algorithm: SemiSupervisedAlgorithm
    validation_split: float = 0.2
    min_confidence_threshold: float = 0.7
    max_iterations: int = 10
    use_calibration: bool = True
    
    # Parámetros específicos por algoritmo
    gamma: Optional[float] = None
    n_neighbors: Optional[int] = None
    alpha: Optional[float] = None  # Para Label Spreading
    
    # Para Self-Training
    threshold: Optional[float] = None

@strawberry.input
class PredictionInput:
    candidate_id: str
    offer_id: str
    
    # Features opcionales para override
    override_compatibility_score: Optional[float] = None
    override_experience_match: Optional[float] = None

@strawberry.input
class BatchPredictionInput:
    application_ids: List[str]
    include_features: bool = False
    confidence_threshold: float = 0.5
    update_database: bool = True

@strawberry.input
class ModelSelectionCriteria:
    min_accuracy: float = 0.7
    min_f1_score: float = 0.6
    max_training_time_minutes: int = 30
    preferred_algorithms: Optional[List[SemiSupervisedAlgorithm]] = None

# Tipos para consultas de análisis
@strawberry.type
class DatasetStatistics:
    total_applications: int
    labeled_applications: int
    unlabeled_applications: int
    labeled_ratio: float
    
    # Distribución por estado
    accepted_applications: int
    rejected_applications: int
    pending_applications: int
    
    # Distribución temporal
    applications_last_week: int
    applications_last_month: int
    applications_last_year: int
    
    # Calidad de datos
    complete_profiles: int
    incomplete_profiles: int
    missing_features_count: int

@strawberry.type
class ModelComparison:
    model_a_id: str
    model_b_id: str
    
    # Comparación de métricas
    accuracy_difference: float
    f1_difference: float
    precision_difference: float
    recall_difference: float
    
    # Comparación de confianza
    confidence_correlation: float
    agreement_rate: float
    
    # Recomendación
    recommended_model: str
    recommendation_reason: str

# Response types para operaciones
@strawberry.type
class OperationResult:
    success: bool
    message: str
    operation_id: Optional[str] = None
    timestamp: datetime
    details: Optional[strawberry.scalars.JSON] = None

@strawberry.type
class ValidationResult:
    is_valid: bool
    validation_errors: List[str]
    warnings: List[str]
    data_quality_score: float
    recommendations: List[str]

# Tipos para filtros y paginación
@strawberry.input
class ApplicationFilter:
    candidate_ids: Optional[List[str]] = None
    offer_ids: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    estados: Optional[List[str]] = None
    is_labeled: Optional[bool] = None
    confidence_level: Optional[ConfidenceLevel] = None
    prediction_status: Optional[PredictionStatus] = None

@strawberry.input
class PaginationInput:
    page: int = 1
    page_size: int = 50
    sort_by: str = "fecha_postulacion"
    sort_order: str = "DESC"  # ASC or DESC

@strawberry.type
class PaginatedApplications:
    applications: List[ApplicationWithPrediction]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next_page: bool
    has_previous_page: bool