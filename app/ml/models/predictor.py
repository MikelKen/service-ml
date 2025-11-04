"""
Módulo de predicción para compatibilidad candidato-oferta usando datos de MongoDB
"""
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

from app.ml.data.data_extractor import data_extractor
from app.ml.preprocessing.mongo_preprocessor import mongo_preprocessor
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompatibilityPredictor:
    """Clase para realizar predicciones de compatibilidad candidato-oferta"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_name = None
        self.model_metrics = {}
        self.feature_importance = {}
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Carga modelo entrenado y preprocessor"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")
            
            # Cargar datos del modelo
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.model_metrics = model_data.get('metrics', {})
            self.feature_importance = model_data.get('feature_importance', {})
            
            # Cargar preprocessor si está incluido
            if 'preprocessor' in model_data:
                preprocessor_data = model_data['preprocessor']
                mongo_preprocessor.tfidf_skills = preprocessor_data['tfidf_skills']
                mongo_preprocessor.tfidf_requirements = preprocessor_data['tfidf_requirements']
                mongo_preprocessor.scaler = preprocessor_data['scaler']
                mongo_preprocessor.label_encoders = preprocessor_data['label_encoders']
                mongo_preprocessor.is_fitted = True
                logger.info("Preprocessor cargado desde el modelo")
            else:
                # Intentar cargar preprocessor por separado
                preprocessor_path = model_path.replace('model.pkl', 'preprocessor.pkl')
                if os.path.exists(preprocessor_path):
                    mongo_preprocessor.load_preprocessor(preprocessor_path)
                    logger.info("Preprocessor cargado desde archivo separado")
                else:
                    logger.warning("No se encontró preprocessor. Las predicciones pueden fallar.")
            
            self.is_loaded = True
            logger.info(f"Modelo cargado exitosamente: {self.model_name}")
            logger.info(f"ROC AUC del modelo: {self.model_metrics.get('roc_auc', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def predict_compatibility(self, candidate_id: str, offer_id: str) -> Dict[str, Any]:
        """Predice compatibilidad entre un candidato y una oferta específica"""
        
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Use load_model() primero.")
        
        try:
            # Obtener datos del par candidato-oferta
            pair_data = data_extractor.get_candidate_offer_pair(candidate_id, offer_id)
            
            if not pair_data:
                return {
                    'candidate_id': candidate_id,
                    'offer_id': offer_id,
                    'probability': 0.0,
                    'prediction': False,
                    'confidence': 'Error',
                    'error': 'No se pudieron obtener datos del candidato o la oferta'
                }
            
            # Convertir a DataFrame
            df = pd.DataFrame([pair_data])
            
            # Preprocessar datos
            df_processed = mongo_preprocessor.preprocess_data(df, fit_transformers=False)
            
            # Excluir columnas de ID
            exclude_columns = ['candidate_id', 'offer_id', 'created_at']
            feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
            X = df_processed[feature_columns]
            
            # Realizar predicción
            probability = self.model.predict_proba(X)[0, 1]
            prediction = self.model.predict(X)[0]
            
            # Determinar nivel de confianza
            confidence = self._calculate_confidence(probability)
            
            result = {
                'candidate_id': candidate_id,
                'offer_id': offer_id,
                'probability': float(probability),
                'prediction': bool(prediction),
                'confidence': confidence,
                'model_used': self.model_name,
                'prediction_date': datetime.now().isoformat()
            }
            
            # Agregar información adicional si está disponible
            if hasattr(X, 'shape'):
                result['features_count'] = X.shape[1]
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción de compatibilidad: {e}")
            return {
                'candidate_id': candidate_id,
                'offer_id': offer_id,
                'probability': 0.0,
                'prediction': False,
                'confidence': 'Error',
                'error': str(e)
            }
    
    def predict_candidates_for_offer(self, offer_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Predice compatibilidad de todos los candidatos para una oferta específica"""
        
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Use load_model() primero.")
        
        try:
            # Obtener todos los candidatos para la oferta
            candidates_data = data_extractor.get_all_candidates_for_offer(offer_id)
            
            if not candidates_data:
                return []
            
            # Convertir a DataFrame
            df = pd.DataFrame(candidates_data)
            
            # Preprocessar datos
            df_processed = mongo_preprocessor.preprocess_data(df, fit_transformers=False)
            
            # Preparar features
            exclude_columns = ['candidate_id', 'offer_id', 'created_at']
            feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
            X = df_processed[feature_columns]
            
            # Realizar predicciones
            probabilities = self.model.predict_proba(X)[:, 1]
            predictions = self.model.predict(X)
            
            # Crear resultados
            results = []
            for i, (_, row) in enumerate(df_processed.iterrows()):
                probability = float(probabilities[i])
                prediction = bool(predictions[i])
                confidence = self._calculate_confidence(probability)
                
                result = {
                    'candidate_id': row.get('candidate_id', ''),
                    'offer_id': offer_id,
                    'probability': probability,
                    'prediction': prediction,
                    'confidence': confidence,
                    'ranking': i + 1  # Se actualizará después de ordenar
                }
                results.append(result)
            
            # Ordenar por probabilidad descendente
            results.sort(key=lambda x: x['probability'], reverse=True)
            
            # Actualizar ranking
            for i, result in enumerate(results):
                result['ranking'] = i + 1
            
            # Retornar top N candidatos
            top_results = results[:top_n]
            
            logger.info(f"Predicciones completadas para {len(results)} candidatos, retornando top {len(top_results)}")
            
            return top_results
            
        except Exception as e:
            logger.error(f"Error en predicción batch para oferta {offer_id}: {e}")
            return []
    
    def predict_batch_custom(self, pairs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Realiza predicciones para una lista de pares candidato-oferta personalizados"""
        
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Use load_model() primero.")
        
        results = []
        
        try:
            # Convertir a DataFrame
            df = pd.DataFrame(pairs_data)
            
            if df.empty:
                return results
            
            # Preprocessar datos
            df_processed = mongo_preprocessor.preprocess_data(df, fit_transformers=False)
            
            # Preparar features
            exclude_columns = ['candidate_id', 'offer_id', 'created_at', 'target']
            feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
            X = df_processed[feature_columns]
            
            # Realizar predicciones
            probabilities = self.model.predict_proba(X)[:, 1]
            predictions = self.model.predict(X)
            
            # Crear resultados
            for i, (_, row) in enumerate(df_processed.iterrows()):
                probability = float(probabilities[i])
                prediction = bool(predictions[i])
                confidence = self._calculate_confidence(probability)
                
                result = {
                    'candidate_id': row.get('candidate_id', f'candidate_{i}'),
                    'offer_id': row.get('offer_id', f'offer_{i}'),
                    'probability': probability,
                    'prediction': prediction,
                    'confidence': confidence,
                    'model_used': self.model_name,
                    'prediction_date': datetime.now().isoformat()
                }
                results.append(result)
            
            logger.info(f"Predicciones batch completadas para {len(results)} pares")
            
        except Exception as e:
            logger.error(f"Error en predicción batch personalizada: {e}")
            # Retornar resultados con error para cada entrada
            for i, pair in enumerate(pairs_data):
                results.append({
                    'candidate_id': pair.get('candidate_id', f'candidate_{i}'),
                    'offer_id': pair.get('offer_id', f'offer_{i}'),
                    'probability': 0.0,
                    'prediction': False,
                    'confidence': 'Error',
                    'error': str(e)
                })
        
        return results
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calcula el nivel de confianza basado en la probabilidad"""
        distance_from_center = abs(probability - 0.5)
        
        if distance_from_center >= 0.4:
            return 'Muy Alta'
        elif distance_from_center >= 0.3:
            return 'Alta'
        elif distance_from_center >= 0.15:
            return 'Media'
        elif distance_from_center >= 0.05:
            return 'Baja'
        else:
            return 'Muy Baja'
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Obtiene la importancia de las features del modelo"""
        if not self.is_loaded:
            return {}
        
        # Si ya está cargada, retornarla
        if self.feature_importance:
            # Retornar top N features
            sorted_features = dict(sorted(self.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
            return dict(list(sorted_features.items())[:top_n])
        
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información detallada del modelo cargado"""
        if not self.is_loaded:
            return {'status': 'No model loaded'}
        
        info = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'is_loaded': self.is_loaded,
            'preprocessor_fitted': mongo_preprocessor.is_fitted,
            'metrics': self.model_metrics
        }
        
        # Añadir información específica del modelo
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
        if hasattr(self.model, 'C'):
            info['regularization_C'] = self.model.C
        
        # Estadísticas de feature importance
        if self.feature_importance:
            info['feature_importance_count'] = len(self.feature_importance)
            info['top_features'] = list(self.feature_importance.keys())[:5]
        
        return info
    
    def explain_prediction(self, candidate_id: str, offer_id: str) -> Dict[str, Any]:
        """Proporciona explicación detallada de una predicción"""
        
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Use load_model() primero.")
        
        try:
            # Realizar predicción básica
            prediction_result = self.predict_compatibility(candidate_id, offer_id)
            
            if 'error' in prediction_result:
                return prediction_result
            
            # Obtener datos del par para análisis
            pair_data = data_extractor.get_candidate_offer_pair(candidate_id, offer_id)
            
            if not pair_data:
                return prediction_result
            
            # Análisis de factores clave
            factors = {
                'skills_overlap': pair_data.get('skills_overlap', 0),
                'years_experience': pair_data.get('years_experience', 0),
                'education_score': pair_data.get('education_score', 0),
                'salary_per_experience': pair_data.get('salary_per_experience', 0),
                'has_certifications': pair_data.get('has_certifications', 0)
            }
            
            # Agregar explicación
            explanation = {
                'prediction': prediction_result,
                'key_factors': factors,
                'feature_importance': self.get_feature_importance(10),
                'recommendation': self._generate_recommendation(prediction_result, factors)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generando explicación: {e}")
            return {
                'error': str(e),
                'prediction': {'probability': 0.0, 'prediction': False}
            }
    
    def _generate_recommendation(self, prediction: Dict, factors: Dict) -> str:
        """Genera recomendación basada en la predicción y factores"""
        
        probability = prediction.get('probability', 0)
        
        if probability >= 0.8:
            return "Candidato altamente recomendado. Excelente compatibilidad con la oferta."
        elif probability >= 0.6:
            return "Buen candidato. Se recomienda continuar con el proceso de entrevista."
        elif probability >= 0.4:
            return "Candidato con potencial moderado. Revisar requisitos específicos."
        elif probability >= 0.2:
            return "Compatibilidad baja. Considerar solo si hay pocos candidatos disponibles."
        else:
            return "Compatibilidad muy baja. No se recomienda para esta posición."


# Instancia global del predictor
compatibility_predictor = CompatibilityPredictor()


def load_default_model():
    """Carga el modelo por defecto si existe"""
    default_model_path = os.path.join(settings.ml_models_path, "compatibility_model.pkl")
    if os.path.exists(default_model_path):
        compatibility_predictor.load_model(default_model_path)
        logger.info("Modelo por defecto cargado")
    else:
        logger.warning("Modelo por defecto no encontrado")


def predict_single_compatibility(candidate_id: str, offer_id: str) -> Dict[str, Any]:
    """Función conveniente para predicción individual"""
    return compatibility_predictor.predict_compatibility(candidate_id, offer_id)


def predict_top_candidates(offer_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """Función conveniente para obtener top candidatos para una oferta"""
    return compatibility_predictor.predict_candidates_for_offer(offer_id, top_n)


# Intentar cargar modelo por defecto al importar
try:
    load_default_model()
except Exception as e:
    logger.warning(f"No se pudo cargar modelo por defecto: {e}")


if __name__ == "__main__":
    # Test del predictor
    try:
        model_path = os.path.join(settings.ml_models_path, "compatibility_model.pkl")
        if os.path.exists(model_path):
            predictor = CompatibilityPredictor(model_path)
            print("Predictor inicializado correctamente")
            print(f"Modelo cargado: {predictor.model_name}")
        else:
            print("Archivo de modelo no encontrado para prueba")
    except Exception as e:
        print(f"Error en test del predictor: {e}")