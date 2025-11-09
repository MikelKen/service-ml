"""
Tipos GraphQL para consultas del modelo semi-supervisado de postulaciones
"""
import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime


@strawberry.type
class EstadoDistribution:
    """Distribución de estados de postulaciones"""
    estado: str
    cantidad: int
    percentage: Optional[float] = None


@strawberry.type
class PostulacionPrediction:
    """Predicción de estado para una postulación"""
    postulacion_id: str
    estado_original: Optional[str]
    estado_predicho: str
    estado_predicho_encoded: int
    model_used: str
    prediction_date: str
    accuracy: Optional[float]


@strawberry.type
class PostulacionesModelMetrics:
    """Métricas del modelo semi-supervisado de postulaciones"""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


@strawberry.type
class PostulacionesTrainingDataSummary:
    """Resumen de datos de entrenamiento para postulaciones"""
    total_records: int
    labeled_records: int
    unlabeled_records: int
    features_count: int
    estado_distribution: List[EstadoDistribution]


@strawberry.type
class SemiSupervisedModelInfo:
    """Información del modelo semi-supervisado"""
    model_name: str
    model_type: str
    training_date: str
    metrics: PostulacionesModelMetrics
    data_summary: PostulacionesTrainingDataSummary
    is_trained: bool


@strawberry.type
class PostulacionFeatures:
    """Características de una postulación para predicción"""
    postulacion_id: str
    nombre: str
    anios_experiencia: int
    nivel_educacion: str
    habilidades: str
    idiomas: str
    certificaciones: str
    puesto_actual: Optional[str]
    # Información de la oferta
    oferta_titulo: str
    salario: Optional[float]
    ubicacion: str
    requisitos: str
    empresa_rubro: str


@strawberry.type
class PostulacionPredictionResult:
    """Resultado de predicción para una postulación"""
    postulacion_data: PostulacionFeatures
    prediction: PostulacionPrediction
    confidence_score: Optional[float]
    probabilities: Optional[str]  # JSON string con probabilidades


@strawberry.type
class PredictionBatch:
    """Lote de predicciones"""
    predictions: List[PostulacionPrediction]
    total_predictions: int
    model_info: SemiSupervisedModelInfo


@strawberry.type
class TrainingResult:
    """Resultado del entrenamiento del modelo"""
    success: bool
    message: str
    model_info: Optional[SemiSupervisedModelInfo]
    training_duration_seconds: Optional[float]
    files_created: List[str]


# Input Types
@strawberry.input
class PostulacionInput:
    """Input para datos de postulación"""
    nombre: str
    anios_experiencia: int
    nivel_educacion: str
    habilidades: str
    idiomas: str
    certificaciones: str
    puesto_actual: Optional[str] = None
    url_cv: Optional[str] = None
    # Información de la oferta
    oferta_titulo: str
    oferta_descripcion: Optional[str] = None
    salario: Optional[float] = None
    ubicacion: str
    requisitos: str
    empresa_rubro: str


@strawberry.input
class PredictionFilter:
    """Filtros para predicciones"""
    estado_predicho: Optional[str] = None
    model_used: Optional[str] = None
    min_accuracy: Optional[float] = None
    limit: Optional[int] = 100


@strawberry.input
class TrainingConfig:
    """Configuración para entrenamiento del modelo"""
    force_retrain: Optional[bool] = False
    test_size: Optional[float] = 0.3
    save_predictions: Optional[bool] = True
    model_types: Optional[List[str]] = None  # ['label_propagation', 'label_spreading', etc.]


@strawberry.type
class PostulacionesDatasetStats:
    """Estadísticas del dataset de postulaciones"""
    total_records: int
    labeled_records: int
    unlabeled_records: int
    state_distribution: List[EstadoDistribution]
    features_count: int
    last_update: Optional[str] = None