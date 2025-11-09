#!/usr/bin/env python3
"""
🎯 PREDICTOR INTEGRADO DE ESTADO DE POSTULACIONES
Interfaz principal para hacer predicciones en el sistema
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
import json
from datetime import datetime

# Configurar path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from app.ml.preprocessing.postulation_preprocessor import PostulationPreprocessor
from app.ml.models.semi_supervised_predictor import SemiSupervisedPredictor

logger = logging.getLogger(__name__)

class PostulationStatusPredictor:
    """
    Predictor principal del estado de postulaciones
    Interfaz unificada para predicciones desde el sistema GraphQL
    """
    
    def __init__(self):
        self.model: Optional[SemiSupervisedPredictor] = None
        self.preprocessor: Optional[PostulationPreprocessor] = None
        self.is_loaded = False
        self.model_info = {}
        
        # Cargar modelo automáticamente si existe
        self._auto_load_latest_model()
    
    def _auto_load_latest_model(self):
        """Carga automáticamente el modelo más reciente disponible"""
        try:
            model_dir = os.path.join(
                os.path.dirname(__file__), 
                'trained_models', 
                'semi_supervised'
            )
            
            if not os.path.exists(model_dir):
                logger.warning("📁 Directorio de modelos no encontrado")
                return
            
            # Buscar archivos de modelo
            model_files = [f for f in os.listdir(model_dir) 
                          if f.startswith('semi_supervised_model_') and f.endswith('.pkl')]
            preprocessor_files = [f for f in os.listdir(model_dir) 
                                if f.startswith('preprocessor_') and f.endswith('.pkl')]
            
            if not model_files or not preprocessor_files:
                logger.warning("📁 No se encontraron archivos de modelo entrenado")
                return
            
            # Usar el más reciente
            latest_model = sorted(model_files)[-1]
            latest_preprocessor = sorted(preprocessor_files)[-1]
            
            model_path = os.path.join(model_dir, latest_model)
            preprocessor_path = os.path.join(model_dir, latest_preprocessor)
            
            self.load_model(model_path, preprocessor_path)
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo automáticamente: {e}")
    
    def load_model(self, model_path: str, preprocessor_path: str):
        """Carga el modelo y preprocessor entrenados"""
        try:
            logger.info(f"📂 Cargando modelo desde: {model_path}")
            self.model = SemiSupervisedPredictor.load_model(model_path)
            
            logger.info(f"📂 Cargando preprocessor desde: {preprocessor_path}")
            self.preprocessor = PostulationPreprocessor.load_preprocessor(preprocessor_path)
            
            self.is_loaded = True
            
            # Guardar información del modelo
            self.model_info = {
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'loaded_at': datetime.now().isoformat(),
                'model_summary': self.model.get_model_summary() if self.model else {}
            }
            
            logger.info("✅ Modelo y preprocessor cargados exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            self.is_loaded = False
            raise
    
    def predict_postulation_status(self, 
                                 candidate_data: Dict[str, Any], 
                                 offer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predice el estado de una postulación específica
        
        Args:
            candidate_data: Datos del candidato
            offer_data: Datos de la oferta de trabajo
        
        Returns:
            Predicción con probabilidades, confianza y recomendaciones
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Usar load_model() primero.")
        
        try:
            # Combinar datos del candidato y oferta
            combined_data = self._combine_candidate_offer_data(candidate_data, offer_data)
            
            # Convertir a DataFrame
            df = pd.DataFrame([combined_data])
            
            # Preprocessar
            X = self.preprocessor.transform(df)
            
            # Predecir
            prediction = self.model.predict_single(X[0])
            
            # Enriquecer predicción con información adicional
            prediction['candidate_id'] = candidate_data.get('id', 'unknown')
            prediction['offer_id'] = offer_data.get('id', 'unknown')
            prediction['prediction_timestamp'] = datetime.now().isoformat()
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return {
                'error': str(e),
                'prediction': 'ERROR',
                'confidence': 0.0,
                'probabilities': {'ACEPTADO': 0.0, 'RECHAZADO': 1.0}
            }
    
    def predict_candidates_for_offer(self, 
                                   candidates_list: List[Dict[str, Any]], 
                                   offer_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predice el estado para múltiples candidatos aplicando a una oferta
        
        Args:
            candidates_list: Lista de candidatos
            offer_data: Datos de la oferta
        
        Returns:
            Lista de predicciones rankeadas por probabilidad de aceptación
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado.")
        
        predictions = []
        
        for candidate in candidates_list:
            try:
                prediction = self.predict_postulation_status(candidate, offer_data)
                prediction['candidate_data'] = candidate
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"❌ Error prediciendo candidato {candidate.get('id', 'unknown')}: {e}")
                continue
        
        # Ordenar por probabilidad de aceptación (descendente)
        predictions.sort(
            key=lambda x: x.get('probabilities', {}).get('ACEPTADO', 0),
            reverse=True
        )
        
        # Agregar ranking
        for i, pred in enumerate(predictions, 1):
            pred['ranking'] = i
        
        return predictions
    
    def get_model_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del modelo"""
        return {
            'is_loaded': self.is_loaded,
            'model_info': self.model_info,
            'features_count': len(self.preprocessor.feature_names) if self.preprocessor else 0,
            'model_algorithm': self.model.algorithm if self.model else None,
            'base_classifier': self.model.base_classifier_type if self.model else None
        }
    
    def _combine_candidate_offer_data(self, 
                                    candidate_data: Dict[str, Any], 
                                    offer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combina datos del candidato y oferta en el formato esperado"""
        
        # Mapear campos del candidato
        candidate_mapped = {
            'candidate_id': candidate_data.get('id', ''),
            'years_experience': candidate_data.get('aniosExperiencia', candidate_data.get('years_experience', 0)),
            'education_level': candidate_data.get('nivelEducacion', candidate_data.get('education_level', '')),
            'skills': candidate_data.get('habilidades', candidate_data.get('skills', '')),
            'languages': candidate_data.get('idiomas', candidate_data.get('languages', '')),
            'certifications': candidate_data.get('certificaciones', candidate_data.get('certifications', '')),
            'current_position': candidate_data.get('puestoActual', candidate_data.get('current_position', ''))
        }
        
        # Mapear campos de la oferta
        offer_mapped = {
            'offer_id': offer_data.get('id', ''),
            'job_title': offer_data.get('titulo', offer_data.get('job_title', '')),
            'salary': offer_data.get('salario', offer_data.get('salary', 0)),
            'location': offer_data.get('ubicacion', offer_data.get('location', '')),
            'requirements': offer_data.get('requisitos', offer_data.get('requirements', '')),
            'company_id': offer_data.get('empresaId', offer_data.get('company_id', ''))
        }
        
        # Combinar datos
        combined = {**candidate_mapped, **offer_mapped}
        
        return combined
    
    def batch_predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones en lote desde un DataFrame
        
        Args:
            df: DataFrame con columnas de candidatos y ofertas
        
        Returns:
            DataFrame original con columnas de predicción agregadas
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado.")
        
        # Preprocessar todo el DataFrame
        X = self.preprocessor.transform(df)
        
        # Predecir en lote
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Agregar resultados al DataFrame
        df_result = df.copy()
        df_result['predicted_status'] = [self.model.label_mapping[pred] for pred in predictions]
        df_result['prob_aceptado'] = probabilities[:, 1]
        df_result['prob_rechazado'] = probabilities[:, 0]
        df_result['confidence'] = np.max(probabilities, axis=1)
        
        return df_result

# Instancia global para usar en el sistema
postulation_predictor = PostulationStatusPredictor()

# Funciones de conveniencia para usar desde GraphQL
def predict_postulation_status(candidate_data: Dict[str, Any], 
                             offer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Función de conveniencia para predicción individual"""
    return postulation_predictor.predict_postulation_status(candidate_data, offer_data)

def predict_candidates_for_offer(candidates_list: List[Dict[str, Any]], 
                               offer_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Función de conveniencia para predicción múltiple"""
    return postulation_predictor.predict_candidates_for_offer(candidates_list, offer_data)

def get_predictor_status() -> Dict[str, Any]:
    """Función de conveniencia para obtener estado del predictor"""
    return postulation_predictor.get_model_status()

def reload_model(model_path: str = None, preprocessor_path: str = None):
    """Función para recargar el modelo"""
    if model_path and preprocessor_path:
        postulation_predictor.load_model(model_path, preprocessor_path)
    else:
        postulation_predictor._auto_load_latest_model()

if __name__ == "__main__":
    # Ejemplo de uso
    print("🎯 PostulationStatusPredictor")
    print(f"✅ Estado del modelo: {postulation_predictor.get_model_status()}")
    
    if postulation_predictor.is_loaded:
        print("🚀 Modelo listo para predicciones")
    else:
        print("⚠️ Modelo no cargado - ejecutar entrenamiento primero")