"""
Tipos base de GraphQL para Machine Learning
"""
import strawberry
from typing import Dict, List, Optional, Any
from datetime import datetime


@strawberry.type
class ApplicationInput:
    """Input de datos de postulación"""
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
    """Input de datos de oferta laboral"""
    titulo: str
    descripcion: str
    salario: float
    ubicacion: str
    requisitos: str
    fecha_publicacion: datetime


@strawberry.input
class PredictionInput:
    """Input completo para predicción"""
    application: ApplicationInput
    job_offer: JobOfferInput


@strawberry.type
class PredictionResult:
    """Resultado de predicción individual"""
    probability: float
    prediction: bool
    confidence: str
    metadata: Optional[Dict[str, Any]] = None


@strawberry.type
class BatchPredictionResult:
    """Resultado de predicción en lote"""
    predictions: List[PredictionResult]
    total_processed: int
    success_count: int
    error_count: int
    processing_time: float


@strawberry.type
class ModelInfo:
    """Información del modelo ML"""
    model_name: str
    model_type: str
    version: str
    training_date: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None
    features_used: Optional[List[str]] = None
    is_loaded: bool = False


@strawberry.type
class TrainingStatus:
    """Estado del entrenamiento"""
    is_training: bool
    progress: float
    current_step: str
    estimated_time_remaining: Optional[int] = None
    error_message: Optional[str] = None


@strawberry.type
class ModelMetrics:
    """Métricas de rendimiento del modelo"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    feature_importance: Optional[Dict[str, float]] = None


@strawberry.type
class DatasetInfo:
    """Información del dataset"""
    total_records: int
    features_count: int
    target_distribution: Dict[str, int]
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    last_updated: Optional[datetime] = None