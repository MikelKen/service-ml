"""
Queries de GraphQL para Machine Learning
"""
import strawberry
from typing import List, Optional
from app.schemas.ml_schemas import (
    PredictionResult, ModelInfo, TrainingStatus, 
    ModelMetrics, DatasetInfo, PredictionInput
)
from app.services.ml_service import (
    predict_hiring, get_model_info, get_training_status,
    get_model_metrics, get_dataset_info
)


@strawberry.type
class MLQuery:
    """Queries relacionadas con Machine Learning"""
    
    @strawberry.field
    def predict_hiring_probability(self, prediction_input: PredictionInput) -> PredictionResult:
        """Predice la probabilidad de que un postulante sea contactado"""
        
        # Convertir inputs a diccionarios
        application_data = {
            'nombre': prediction_input.application.nombre,
            'años_experiencia': prediction_input.application.años_experiencia,
            'nivel_educacion': prediction_input.application.nivel_educacion,
            'habilidades': prediction_input.application.habilidades,
            'idiomas': prediction_input.application.idiomas,
            'certificaciones': prediction_input.application.certificaciones or "",
            'puesto_actual': prediction_input.application.puesto_actual,
            'industria': prediction_input.application.industria,
            'url_cv': prediction_input.application.url_cv or "",
            'fecha_postulacion': prediction_input.application.fecha_postulacion
        }
        
        job_offer_data = {
            'titulo': prediction_input.job_offer.titulo,
            'descripcion': prediction_input.job_offer.descripcion,
            'salario': prediction_input.job_offer.salario,
            'ubicacion': prediction_input.job_offer.ubicacion,
            'requisitos': prediction_input.job_offer.requisitos,
            'fecha_publicacion': prediction_input.job_offer.fecha_publicacion
        }
        
        # Realizar predicción
        result = predict_hiring(application_data, job_offer_data)
        
        # Convertir resultado a schema de GraphQL
        return PredictionResult(
            hiring_prediction=result['hiring_prediction'],
            feature_importance=result['feature_importance'],
            processing_time_ms=result['processing_time_ms']
        )
    
    @strawberry.field
    def model_info(self) -> ModelInfo:
        """Obtiene información del modelo actual"""
        info = get_model_info()
        
        return ModelInfo(
            model_name=info['model_name'],
            is_loaded=info['is_loaded'],
            last_trained=info['last_trained'],
            version=info['version']
        )
    
    @strawberry.field
    def training_status(self) -> TrainingStatus:
        """Obtiene el estado actual del entrenamiento"""
        status = get_training_status()
        
        return TrainingStatus(
            is_training=status['is_training'],
            progress=status['progress'],
            status_message=status['status_message'],
            estimated_completion=status['estimated_completion']
        )
    
    @strawberry.field
    def model_metrics(self) -> Optional[ModelMetrics]:
        """Obtiene métricas de rendimiento del modelo"""
        metrics = get_model_metrics()
        
        if metrics is None:
            return None
        
        return ModelMetrics(
            roc_auc=metrics['roc_auc'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            accuracy=metrics['accuracy']
        )
    
    @strawberry.field
    def dataset_info(self, data_path: Optional[str] = None) -> DatasetInfo:
        """Obtiene información del dataset"""
        info = get_dataset_info(data_path)
        
        return DatasetInfo(
            total_records=info['total_records'],
            positive_class_count=info['positive_class_count'],
            negative_class_count=info['negative_class_count'],
            class_balance_ratio=info['class_balance_ratio'],
            last_updated=info['last_updated']
        )
    
    @strawberry.field
    def health_check_ml(self) -> str:
        """Verifica el estado del servicio de ML"""
        try:
            model_info = get_model_info()
            if model_info['is_loaded']:
                return "ML Service: Healthy - Model loaded and ready"
            else:
                return "ML Service: Warning - No model loaded"
        except Exception as e:
            return f"ML Service: Error - {str(e)}"