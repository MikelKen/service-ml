"""
Mutaciones de GraphQL para Machine Learning
"""
import strawberry
from typing import List
from app.schemas.ml_schemas import (
    BatchPredictionResult, TrainingStatus, PredictionInput, ModelInfo
)
from app.services.ml_service import (
    predict_hiring_batch, train_model, ml_service
)


@strawberry.type
class MLMutation:
    """Mutaciones relacionadas con Machine Learning"""
    
    @strawberry.mutation
    def predict_hiring_batch(self, predictions: List[PredictionInput]) -> BatchPredictionResult:
        """Predice probabilidades de contratación para múltiples postulaciones"""
        
        # Convertir inputs a formato interno
        predictions_data = []
        
        for pred_input in predictions:
            application_data = {
                'nombre': pred_input.application.nombre,
                'años_experiencia': pred_input.application.años_experiencia,
                'nivel_educacion': pred_input.application.nivel_educacion,
                'habilidades': pred_input.application.habilidades,
                'idiomas': pred_input.application.idiomas,
                'certificaciones': pred_input.application.certificaciones or "",
                'puesto_actual': pred_input.application.puesto_actual,
                'industria': pred_input.application.industria,
                'url_cv': pred_input.application.url_cv or "",
                'fecha_postulacion': pred_input.application.fecha_postulacion
            }
            
            job_offer_data = {
                'titulo': pred_input.job_offer.titulo,
                'descripcion': pred_input.job_offer.descripcion,
                'salario': pred_input.job_offer.salario,
                'ubicacion': pred_input.job_offer.ubicacion,
                'requisitos': pred_input.job_offer.requisitos,
                'fecha_publicacion': pred_input.job_offer.fecha_publicacion
            }
            
            predictions_data.append({
                'application': application_data,
                'job_offer': job_offer_data
            })
        
        # Realizar predicciones en lote
        result = predict_hiring_batch(predictions_data)
        
        return BatchPredictionResult(
            total_applications=result['total_applications'],
            successful_predictions=result['successful_predictions'],
            failed_predictions=result['failed_predictions'],
            predictions=result['predictions']
        )
    
    @strawberry.mutation
    async def train_model(self, data_path: str = "datos_entrenamiento_realista.csv") -> TrainingStatus:
        """Inicia el entrenamiento del modelo de ML"""
        
        try:
            # Verificar si ya hay entrenamiento en progreso
            current_status = ml_service.get_training_status()
            if current_status['is_training']:
                return TrainingStatus(
                    is_training=True,
                    progress=current_status['progress'],
                    status_message="Ya hay un entrenamiento en progreso",
                    estimated_completion=None
                )
            
            # Iniciar entrenamiento asíncrono
            success = await train_model(data_path)
            
            if success:
                return TrainingStatus(
                    is_training=False,
                    progress=100.0,
                    status_message="Entrenamiento completado exitosamente",
                    estimated_completion=None
                )
            else:
                return TrainingStatus(
                    is_training=False,
                    progress=0.0,
                    status_message="Error durante el entrenamiento",
                    estimated_completion=None
                )
                
        except Exception as e:
            return TrainingStatus(
                is_training=False,
                progress=0.0,
                status_message=f"Error: {str(e)}",
                estimated_completion=None
            )
    
    @strawberry.mutation
    def reload_model(self) -> ModelInfo:
        """Recarga el modelo desde el archivo guardado"""
        
        try:
            success = ml_service.load_model()
            
            if success:
                model_info = ml_service.get_model_info()
                return ModelInfo(
                    model_name=model_info['model_name'],
                    is_loaded=model_info['is_loaded'],
                    last_trained=model_info['last_trained'],
                    version=model_info['version']
                )
            else:
                return ModelInfo(
                    model_name="Error",
                    is_loaded=False,
                    last_trained=None,
                    version="0.0.0"
                )
                
        except Exception as e:
            return ModelInfo(
                model_name=f"Error: {str(e)}",
                is_loaded=False,
                last_trained=None,
                version="0.0.0"
            )
    
    @strawberry.mutation
    def reset_training_status(self) -> TrainingStatus:
        """Resetea el estado de entrenamiento (útil si hay un estado colgado)"""
        
        try:
            ml_service.is_training = False
            ml_service.training_progress = 0.0
            
            return TrainingStatus(
                is_training=False,
                progress=0.0,
                status_message="Estado de entrenamiento reseteado",
                estimated_completion=None
            )
            
        except Exception as e:
            return TrainingStatus(
                is_training=True,  # Mantener estado si hay error
                progress=0.0,
                status_message=f"Error reseteando estado: {str(e)}",
                estimated_completion=None
            )