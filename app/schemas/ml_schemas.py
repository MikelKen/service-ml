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
    años_experiencia: int
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
    años_experiencia: int
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