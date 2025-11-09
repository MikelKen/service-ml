#!/usr/bin/env python3
"""
🎯 TIPOS GRAPHQL PARA PREDICCIÓN DE POSTULACIONES
Definiciones de tipos para el modelo semi-supervisado
"""

import strawberry
from typing import Optional, List
from datetime import datetime

@strawberry.type
class PostulationPrediction:
    """Resultado de predicción de estado de postulación"""
    prediction: str
    prediction_numeric: int
    confidence: float
    prob_aceptado: float
    prob_rechazado: float
    recommendation: str
    candidate_id: Optional[str] = None
    offer_id: Optional[str] = None
    prediction_timestamp: Optional[str] = None

@strawberry.type
class CandidateRanking:
    """Candidato rankeado por compatibilidad con oferta"""
    ranking: int
    candidate_id: str
    prediction: str
    confidence: float
    prob_aceptado: float
    prob_rechazado: float
    recommendation: str

@strawberry.type
class PredictorStatus:
    """Estado del predictor de postulaciones"""
    is_loaded: bool
    features_count: int
    model_algorithm: Optional[str] = None
    base_classifier: Optional[str] = None
    loaded_at: Optional[str] = None

@strawberry.type
class PredictorModelInfo:
    """Información detallada del modelo"""
    algorithm: str
    base_classifier: str
    is_fitted: bool
    n_features: int
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    training_date: Optional[str] = None

@strawberry.input
class CandidateInput:
    """Input de datos del candidato para predicción"""
    id: str
    anios_experiencia: int
    nivel_educacion: str
    habilidades: str
    idiomas: str
    certificaciones: str
    puesto_actual: str

@strawberry.input
class OfferInput:
    """Input de datos de la oferta para predicción"""
    id: str
    titulo: str
    salario: float
    ubicacion: str
    requisitos: str
    empresa_id: str

@strawberry.input
class PostulationInput:
    """Input completo de postulación (candidato + oferta)"""
    candidate: CandidateInput
    offer: OfferInput

@strawberry.type
class BatchPredictionResult:
    """Resultado de predicción en lote"""
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[PostulationPrediction]
    errors: List[str]

@strawberry.type
class ModelTrainingStatus:
    """Estado de entrenamiento del modelo"""
    is_training: bool
    progress_percentage: Optional[float] = None
    current_step: Optional[str] = None
    estimated_time_remaining: Optional[str] = None
    last_training_date: Optional[str] = None

@strawberry.type
class FeatureImportance:
    """Importancia de características del modelo"""
    feature_name: str
    importance_score: float
    rank: int

@strawberry.type
class ModelMetrics:
    """Métricas del modelo entrenado"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float] = None
    confusion_matrix: List[List[int]]

@strawberry.type
class PredictionExplanation:
    """Explicación detallada de una predicción"""
    prediction: PostulationPrediction
    top_positive_features: List[FeatureImportance]
    top_negative_features: List[FeatureImportance]
    explanation_text: str

# Enums para mayor claridad
@strawberry.enum
class PredictionStatus:
    ACEPTADO = "ACEPTADO"
    RECHAZADO = "RECHAZADO"
    ERROR = "ERROR"

@strawberry.enum
class ModelAlgorithm:
    LABEL_PROPAGATION = "label_propagation"
    LABEL_SPREADING = "label_spreading"
    SELF_TRAINING = "self_training"

@strawberry.enum
class BaseClassifier:
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"

# Tipos de respuesta para mutations
@strawberry.type
class TrainModelResponse:
    """Respuesta de entrenamiento de modelo"""
    success: bool
    message: str
    model_path: Optional[str] = None
    training_summary: Optional[str] = None

@strawberry.type
class ReloadModelResponse:
    """Respuesta de recarga de modelo"""
    success: bool
    message: str
    model_info: Optional[PredictorModelInfo] = None

# Input para configuración de entrenamiento
@strawberry.input
class TrainingConfig:
    """Configuración para entrenamiento del modelo"""
    algorithm: ModelAlgorithm
    base_classifier: BaseClassifier
    labeled_ratio: float = 0.3
    n_samples: int = 1000
    test_size: float = 0.2