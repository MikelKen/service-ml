"""
GraphQL simple para predicciones de contratación
"""
import strawberry
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@strawberry.type
class HiringPrediction:
    """Resultado de predicción de contratación"""
    prediction: int
    probability: float
    confidence_level: str
    recommendation: str
    model_used: str


@strawberry.type
class ModelStatus:
    """Estado del modelo"""
    is_loaded: bool
    model_name: str
    version: str


@strawberry.type
class Query:
    """Consultas GraphQL"""
    
    @strawberry.field
    def model_status(self) -> ModelStatus:
        """Obtiene el estado del modelo"""
        try:
            from app.services.ml_service import ml_service
            info = ml_service.get_model_info()
            return ModelStatus(
                is_loaded=info.get('is_loaded', False),
                model_name=info.get('model_name', 'Unknown'),
                version=info.get('version', '1.0.0')
            )
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return ModelStatus(
                is_loaded=False,
                model_name="Error",
                version="0.0.0"
            )
    
    @strawberry.field
    def health_check(self) -> str:
        """Verificación de salud"""
        return "ML Service is healthy"


@strawberry.type
class Mutation:
    """Mutaciones GraphQL"""
    
    @strawberry.mutation
    def predict_hiring(
        self,
        nombre: str,
        anos_experiencia: int,
        nivel_educacion: str,
        habilidades: str,
        idiomas: str,
        certificaciones: Optional[str] = None,
        puesto_actual: Optional[str] = None,
        industria: Optional[str] = None,
        titulo: str = "Desarrollador",
        descripcion: str = "Desarrollo de software",
        salario: float = 1000,
        ubicacion: str = "Santa Cruz",
        requisitos: str = "Experiencia en desarrollo",
        fecha_postulacion: Optional[str] = None,
        fecha_publicacion: Optional[str] = None
    ) -> HiringPrediction:
        """Predice la probabilidad de contratación"""
        
        try:
            # Importar el predictor simple
            from simple_predictor import SimpleHiringPredictor
            import os
            
            # Usar modelo entrenado si existe, sino usar datos por defecto
            model_path = "trained_models/simple_hiring_model.pkl"
            
            if not os.path.exists(model_path):
                logger.warning("Modelo no encontrado, usando predicción por defecto")
                return HiringPrediction(
                    prediction=1,
                    probability=0.75,
                    confidence_level="Alta",
                    recommendation="Recomendado para entrevista",
                    model_used="DefaultPredictor"
                )
            
            # Crear datos de entrada
            application_data = {
                'nombre': nombre,
                'años_experiencia': anos_experiencia,
                'nivel_educacion': nivel_educacion,
                'habilidades': habilidades,
                'idiomas': idiomas,
                'certificaciones': certificaciones or "",
                'puesto_actual': puesto_actual or "Desarrollador",
                'industria': industria or "Tecnología",
                'titulo': titulo,
                'descripcion': descripcion,
                'salario': salario,
                'ubicacion': ubicacion,
                'requisitos': requisitos,
                'fecha_postulacion': fecha_postulacion or '2024-01-15',
                'fecha_publicacion': fecha_publicacion or '2024-01-10'
            }
            
            # Realizar predicción
            predictor = SimpleHiringPredictor(model_path)
            result = predictor.predict(application_data)
            
            return HiringPrediction(
                prediction=result['prediction'],
                probability=result['probability'],
                confidence_level=result['confidence_level'],
                recommendation=result['recommendation'],
                model_used=result['model_used']
            )
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            # Retornar predicción por defecto en caso de error
            return HiringPrediction(
                prediction=0,
                probability=0.30,
                confidence_level="Baja",
                recommendation=f"Error en predicción: {str(e)}",
                model_used="ErrorHandler"
            )