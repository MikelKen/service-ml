"""
Módulo de predicción para evaluar probabilidad de contratación
"""
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Any
from ml.features.feature_engineering import FeatureEngineer
from ml.data.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HiringPredictor:
    """Clase para realizar predicciones de contratación"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_engineer = None
        self.model_name = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Carga modelo y artefactos entrenados"""
        try:
            artifacts = joblib.load(model_path)
            
            self.model = artifacts['model']
            self.model_name = artifacts['model_name']
            
            if 'feature_engineer' in artifacts:
                self.feature_engineer = artifacts['feature_engineer']
            else:
                # Si no está incluido, crear uno nuevo (necesitará ser ajustado)
                self.feature_engineer = FeatureEngineer()
                logger.warning("Feature engineer no encontrado en artefactos. Usando uno nuevo.")
            
            self.is_loaded = True
            logger.info(f"Modelo cargado exitosamente: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def preprocess_single_application(self, application_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocessa datos de una sola postulación"""
        # Convertir a DataFrame
        df = pd.DataFrame([application_data])
        
        # Usar preprocessador
        preprocessor = DataPreprocessor()
        
        # Aplicar normalización básica
        for col in df.columns:
            if col in ['habilidades', 'idiomas', 'certificaciones', 'requisitos', 'descripcion', 'titulo']:
                df[col] = df[col].apply(preprocessor.normalize_text)
        
        # Convertir tipos de datos
        if 'años_experiencia' in df.columns:
            df['años_experiencia'] = pd.to_numeric(df['años_experiencia'], errors='coerce').fillna(0)
        
        if 'salario' in df.columns:
            df['salario'] = pd.to_numeric(df['salario'], errors='coerce').fillna(0)
        
        # Procesar fechas si están presentes
        date_columns = ['fecha_postulacion', 'fecha_publicacion']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Crear features temporales básicas
        if 'fecha_postulacion' in df.columns and 'fecha_publicacion' in df.columns:
            df['dias_desde_publicacion'] = (df['fecha_postulacion'] - df['fecha_publicacion']).dt.days.fillna(0)
        
        if 'fecha_postulacion' in df.columns:
            df['mes_postulacion'] = df['fecha_postulacion'].dt.month.fillna(1)
            df['dia_semana_postulacion'] = df['fecha_postulacion'].dt.dayofweek.fillna(0)
        
        # Calcular features adicionales
        df = self._add_computed_features(df)
        
        return df
    
    def _add_computed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega features computadas necesarias para la predicción"""
        
        # Coincidencia de habilidades
        if 'habilidades' in df.columns and 'requisitos' in df.columns:
            df['coincidencia_habilidades'] = df.apply(
                lambda row: self._calculate_skill_overlap(row['habilidades'], row['requisitos']), 
                axis=1
            )
        else:
            df['coincidencia_habilidades'] = 0.0
        
        # Número de habilidades
        if 'habilidades' in df.columns:
            df['num_habilidades'] = df['habilidades'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0
            )
        else:
            df['num_habilidades'] = 0
        
        # Número de idiomas
        if 'idiomas' in df.columns:
            df['num_idiomas'] = df['idiomas'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0
            )
        else:
            df['num_idiomas'] = 0
        
        # Tiene certificaciones
        if 'certificaciones' in df.columns:
            df['tiene_certificaciones'] = df['certificaciones'].apply(
                lambda x: 1 if (pd.notna(x) and str(x).lower() not in ['', 'sin certificacion', 'ninguna']) else 0
            )
        else:
            df['tiene_certificaciones'] = 0
        
        # Salario por año de experiencia
        if 'años_experiencia' in df.columns and 'salario' in df.columns:
            df['salario_por_año_exp'] = df.apply(
                lambda row: row['salario'] / max(row['años_experiencia'], 1), axis=1
            )
        else:
            df['salario_por_año_exp'] = 0
        
        # Crear variable objetivo dummy (no se usa para predicción)
        df['contactado'] = 0
        
        return df
    
    def _calculate_skill_overlap(self, applicant_skills: str, job_requirements: str) -> float:
        """Calcula overlap entre habilidades del postulante y requisitos del trabajo"""
        if pd.isna(applicant_skills) or pd.isna(job_requirements) or applicant_skills == '' or job_requirements == '':
            return 0.0
        
        # Normalizar y dividir en listas
        skills_list = set([skill.strip().lower() for skill in str(applicant_skills).split(',') if skill.strip()])
        requirements_list = set([req.strip().lower() for req in str(job_requirements).split(',') if req.strip()])
        
        if not skills_list or not requirements_list:
            return 0.0
        
        # Calcular intersección
        intersection = skills_list.intersection(requirements_list)
        return len(intersection) / len(requirements_list) if requirements_list else 0.0
    
    def predict_single(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predice probabilidad de contratación para una sola postulación"""
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Use load_model() primero.")
        
        try:
            # Preprocessar datos
            df = self.preprocess_single_application(application_data)
            
            # Crear features
            X = self.feature_engineer.transform(df)
            
            # Realizar predicción
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0, 1]
            
            # Interpretar resultado
            confidence_level = self._get_confidence_level(probability)
            recommendation = self._get_recommendation(probability)
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence_level': confidence_level,
                'recommendation': recommendation,
                'model_used': self.model_name
            }
            
            logger.info(f"Predicción realizada: {probability:.3f} probabilidad de contacto")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise
    
    def predict_batch(self, applications_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predice para múltiples postulaciones"""
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Use load_model() primero.")
        
        results = []
        
        for i, app_data in enumerate(applications_data):
            try:
                result = self.predict_single(app_data)
                result['application_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error en aplicación {i}: {str(e)}")
                results.append({
                    'application_index': i,
                    'error': str(e),
                    'prediction': None,
                    'probability': None
                })
        
        return results
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determina nivel de confianza basado en probabilidad"""
        if probability >= 0.8:
            return "Muy Alta"
        elif probability >= 0.6:
            return "Alta"
        elif probability >= 0.4:
            return "Media"
        elif probability >= 0.2:
            return "Baja"
        else:
            return "Muy Baja"
    
    def _get_recommendation(self, probability: float) -> str:
        """Genera recomendación basada en probabilidad"""
        if probability >= 0.7:
            return "Fuertemente recomendado para entrevista"
        elif probability >= 0.5:
            return "Recomendado para entrevista"
        elif probability >= 0.3:
            return "Considerar para entrevista"
        else:
            return "No recomendado en esta ronda"
    
    def get_feature_importance_for_prediction(self, application_data: Dict[str, Any], 
                                            top_n: int = 10) -> List[Dict[str, Any]]:
        """Obtiene importancia de features para una predicción específica"""
        if not self.is_loaded or not hasattr(self.model, 'feature_importances_'):
            return []
        
        try:
            # Obtener nombres de features del feature engineer
            feature_names = self.feature_engineer.get_feature_importance_names()
            
            if not feature_names:
                return []
            
            # Obtener importancia del modelo
            importance = self.model.feature_importances_
            
            # Crear lista de importancia ordenada
            feature_importance = [
                {
                    'feature_name': name,
                    'importance': float(imp)
                }
                for name, imp in zip(feature_names, importance)
            ]
            
            # Ordenar por importancia y tomar top N
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return feature_importance[:top_n]
            
        except Exception as e:
            logger.error(f"Error obteniendo importancia de features: {str(e)}")
            return []


# Funciones de utilidad para uso directo
def predict_hiring_probability(application_data: Dict[str, Any], 
                             model_path: str) -> Dict[str, Any]:
    """Función de utilidad para predicción directa"""
    predictor = HiringPredictor(model_path)
    return predictor.predict_single(application_data)


def predict_batch_hiring_probability(applications_data: List[Dict[str, Any]], 
                                   model_path: str) -> List[Dict[str, Any]]:
    """Función de utilidad para predicción en lote"""
    predictor = HiringPredictor(model_path)
    return predictor.predict_batch(applications_data)


if __name__ == "__main__":
    # Test del predictor
    
    # Datos de ejemplo para prueba
    test_application = {
        'nombre': 'Juan Pérez',
        'años_experiencia': 5,
        'nivel_educacion': 'licenciatura',
        'habilidades': 'python, machine learning, sql',
        'idiomas': 'español, inglés',
        'certificaciones': 'aws cloud practitioner',
        'puesto_actual': 'data scientist',
        'industria': 'tecnología',
        'titulo': 'Senior Data Scientist',
        'descripcion': 'Buscamos un data scientist con experiencia en ML',
        'salario': 8000,
        'ubicacion': 'santa cruz',
        'requisitos': 'python, machine learning, sql, aws',
        'fecha_postulacion': '2024-01-15',
        'fecha_publicacion': '2024-01-10'
    }
    
    # Nota: Requiere modelo entrenado
    try:
        result = predict_hiring_probability(test_application, "../../trained_models/hiring_model.pkl")
        print(f"Probabilidad de contacto: {result['probability']:.3f}")
        print(f"Recomendación: {result['recommendation']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Nota: Entrene el modelo primero usando el script de entrenamiento")