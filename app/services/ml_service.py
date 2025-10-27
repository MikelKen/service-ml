"""
Servicio de Machine Learning para predicción de contratación
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import time

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.models.predictor import HiringPredictor
from ml.data.preprocessing import preprocess_data
from ml.features.feature_engineering import FeatureEngineer
from ml.models.trainer import train_hiring_prediction_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLService:
    """Servicio principal de Machine Learning"""
    
    def __init__(self):
        self.predictor = None
        self.model_path = "trained_models/hiring_prediction_model.pkl"
        self.is_model_loaded = False
        self.model_info = {
            'model_name': 'Unknown',
            'last_trained': None,
            'version': '1.0.0'
        }
        self.is_training = False
        self.training_progress = 0.0
        
        # Intentar cargar modelo al inicializar
        self.load_model()
    
    def load_model(self) -> bool:
        """Carga el modelo entrenado"""
        try:
            if os.path.exists(self.model_path):
                self.predictor = HiringPredictor(self.model_path)
                self.is_model_loaded = True
                self.model_info['model_name'] = self.predictor.model_name
                
                # Obtener fecha de modificación del archivo
                mod_time = os.path.getmtime(self.model_path)
                self.model_info['last_trained'] = datetime.fromtimestamp(mod_time).isoformat()
                
                logger.info(f"Modelo cargado exitosamente: {self.model_info['model_name']}")
                return True
            else:
                logger.warning(f"Archivo de modelo no encontrado: {self.model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            self.is_model_loaded = False
            return False
    
    def predict_hiring_probability(self, application_data: Dict[str, Any], 
                                 job_offer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predice probabilidad de contratación"""
        if not self.is_model_loaded:
            raise ValueError("Modelo no cargado. Entrene o cargue un modelo primero.")
        
        start_time = time.time()
        
        try:
            # Combinar datos de postulación y oferta
            combined_data = {**application_data, **job_offer_data}
            
            # Realizar predicción
            result = self.predictor.predict_single(combined_data)
            
            # Obtener importancia de features
            feature_importance = self.predictor.get_feature_importance_for_prediction(
                combined_data, top_n=10
            )
            
            # Calcular tiempo de procesamiento
            processing_time = (time.time() - start_time) * 1000  # en ms
            
            return {
                'hiring_prediction': result,
                'feature_importance': feature_importance,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise
    
    def predict_batch(self, predictions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predice para múltiples postulaciones"""
        if not self.is_model_loaded:
            raise ValueError("Modelo no cargado. Entrene o cargue un modelo primero.")
        
        start_time = time.time()
        
        try:
            # Preparar datos combinados
            combined_data_list = []
            for pred_data in predictions_data:
                combined = {**pred_data['application'], **pred_data['job_offer']}
                combined_data_list.append(combined)
            
            # Realizar predicciones en lote
            batch_results = self.predictor.predict_batch(combined_data_list)
            
            # Procesar resultados
            successful_predictions = []
            failed_count = 0
            
            for i, result in enumerate(batch_results):
                if 'error' not in result:
                    # Obtener importancia de features para esta predicción
                    try:
                        feature_importance = self.predictor.get_feature_importance_for_prediction(
                            combined_data_list[i], top_n=5
                        )
                    except:
                        feature_importance = []
                    
                    successful_predictions.append({
                        'hiring_prediction': {
                            'prediction': result['prediction'],
                            'probability': result['probability'],
                            'confidence_level': result['confidence_level'],
                            'recommendation': result['recommendation'],
                            'model_used': result['model_used']
                        },
                        'feature_importance': feature_importance,
                        'processing_time_ms': 0  # Se calculará al final
                    })
                else:
                    failed_count += 1
            
            processing_time = (time.time() - start_time) * 1000  # en ms
            
            # Actualizar tiempo de procesamiento para cada predicción
            for pred in successful_predictions:
                pred['processing_time_ms'] = processing_time / len(successful_predictions)
            
            return {
                'total_applications': len(predictions_data),
                'successful_predictions': len(successful_predictions),
                'failed_predictions': failed_count,
                'predictions': successful_predictions
            }
            
        except Exception as e:
            logger.error(f"Error en predicción por lotes: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo"""
        return {
            'model_name': self.model_info['model_name'],
            'is_loaded': self.is_model_loaded,
            'last_trained': self.model_info['last_trained'],
            'version': self.model_info['version']
        }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Obtiene estado del entrenamiento"""
        return {
            'is_training': self.is_training,
            'progress': self.training_progress,
            'status_message': 'Entrenando modelo...' if self.is_training else 'Listo',
            'estimated_completion': None  # Se podría implementar estimación
        }
    
    async def train_model_async(self, data_path: str = "postulaciones_sinteticas_500.csv") -> bool:
        """Entrena el modelo de forma asíncrona"""
        if self.is_training:
            raise ValueError("Ya hay un entrenamiento en progreso")
        
        self.is_training = True
        self.training_progress = 0.0
        
        try:
            logger.info("Iniciando entrenamiento asíncrono del modelo...")
            
            # Simular progreso de entrenamiento
            await asyncio.sleep(1)
            self.training_progress = 20.0
            
            # 1. Preprocessar datos
            logger.info("Preprocessando datos...")
            df_processed, summary = preprocess_data(data_path)
            self.training_progress = 40.0
            await asyncio.sleep(1)
            
            # 2. Crear features
            logger.info("Creando features...")
            feature_engineer = FeatureEngineer()
            self.training_progress = 60.0
            await asyncio.sleep(1)
            
            # 3. Entrenar modelo
            logger.info("Entrenando modelo...")
            trainer = train_hiring_prediction_model(df_processed, feature_engineer)
            self.training_progress = 80.0
            await asyncio.sleep(1)
            
            # 4. Guardar modelo
            logger.info("Guardando modelo...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            trainer.save_model(self.model_path, include_feature_engineer=True)
            self.training_progress = 90.0
            await asyncio.sleep(1)
            
            # 5. Recargar modelo
            logger.info("Recargando modelo...")
            self.load_model()
            self.training_progress = 100.0
            
            logger.info("Entrenamiento completado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {str(e)}")
            return False
        
        finally:
            self.is_training = False
            self.training_progress = 0.0
    
    def get_model_metrics(self) -> Optional[Dict[str, float]]:
        """Obtiene métricas del modelo (simuladas)"""
        if not self.is_model_loaded:
            return None
        
        # En una implementación real, estas métricas deberían guardarse durante el entrenamiento
        return {
            'roc_auc': 0.85,  # Ejemplo
            'precision': 0.78,
            'recall': 0.82,
            'f1_score': 0.80,
            'accuracy': 0.83
        }
    
    def get_dataset_info(self, data_path: str = "postulaciones_sinteticas_500.csv") -> Dict[str, Any]:
        """Obtiene información del dataset"""
        try:
            df_processed, summary = preprocess_data(data_path)
            
            if 'contactado' in df_processed.columns:
                positive_count = int(df_processed['contactado'].sum())
                negative_count = int(len(df_processed) - positive_count)
                balance_ratio = positive_count / negative_count if negative_count > 0 else 0
            else:
                positive_count = 0
                negative_count = len(df_processed)
                balance_ratio = 0
            
            return {
                'total_records': len(df_processed),
                'positive_class_count': positive_count,
                'negative_class_count': negative_count,
                'class_balance_ratio': balance_ratio,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo info del dataset: {str(e)}")
            return {
                'total_records': 0,
                'positive_class_count': 0,
                'negative_class_count': 0,
                'class_balance_ratio': 0.0,
                'last_updated': None
            }


# Instancia global del servicio
ml_service = MLService()


# Funciones de utilidad para GraphQL
def predict_hiring(application_data: Dict[str, Any], job_offer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Función de utilidad para predicción de contratación"""
    return ml_service.predict_hiring_probability(application_data, job_offer_data)


def predict_hiring_batch(predictions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Función de utilidad para predicción en lote"""
    return ml_service.predict_batch(predictions_data)


async def train_model(data_path: str = None) -> bool:
    """Función de utilidad para entrenar modelo"""
    if data_path is None:
        data_path = "postulaciones_sinteticas_500.csv"
    return await ml_service.train_model_async(data_path)


def get_model_info() -> Dict[str, Any]:
    """Función de utilidad para obtener info del modelo"""
    return ml_service.get_model_info()


def get_training_status() -> Dict[str, Any]:
    """Función de utilidad para obtener estado de entrenamiento"""
    return ml_service.get_training_status()


def get_model_metrics() -> Optional[Dict[str, float]]:
    """Función de utilidad para obtener métricas del modelo"""
    return ml_service.get_model_metrics()


def get_dataset_info(data_path: str = None) -> Dict[str, Any]:
    """Función de utilidad para obtener info del dataset"""
    if data_path is None:
        data_path = "postulaciones_sinteticas_500.csv"
    return ml_service.get_dataset_info(data_path)