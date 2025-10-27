"""
Predictor simple para el modelo de contratación
"""
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHiringPredictor:
    """Predictor simple para probabilidad de contratación"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_name = ""
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Carga modelo simple"""
        try:
            artifacts = joblib.load(model_path)
            
            self.model = artifacts['model']
            self.scaler = artifacts['scaler']
            self.feature_names = artifacts['feature_names']
            self.model_name = artifacts['model_name']
            self.is_loaded = True
            
            logger.info(f"Modelo cargado: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def preprocess_application(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocesa una aplicación para predicción"""
        
        # Convertir a DataFrame
        df = pd.DataFrame([data])
        
        # Limpiar datos básicos
        df['años_experiencia'] = pd.to_numeric(df.get('años_experiencia', 0), errors='coerce').fillna(0)
        df['salario'] = pd.to_numeric(df.get('salario', 0), errors='coerce').fillna(0)
        
        # Fechas
        df['fecha_postulacion'] = pd.to_datetime(df.get('fecha_postulacion', '2024-01-01'), errors='coerce')
        df['fecha_publicacion'] = pd.to_datetime(df.get('fecha_publicacion', '2024-01-01'), errors='coerce')
        df['dias_desde_publicacion'] = (df['fecha_postulacion'] - df['fecha_publicacion']).dt.days.fillna(0)
        
        # Crear features simples
        features = pd.DataFrame()
        
        features['años_experiencia'] = df['años_experiencia']
        features['salario'] = df['salario']
        features['dias_desde_publicacion'] = df['dias_desde_publicacion']
        
        # Skill match simple
        skills = str(data.get('habilidades', '')).lower()
        reqs = str(data.get('requisitos', '')).lower()
        
        if skills and reqs:
            skills_set = set([s.strip() for s in skills.split(',') if s.strip()])
            reqs_set = set([r.strip() for r in reqs.split(',') if r.strip()])
            
            if skills_set and reqs_set:
                intersection = skills_set.intersection(reqs_set)
                skill_match = len(intersection) / len(reqs_set)
            else:
                skill_match = 0
        else:
            skill_match = 0
        
        features['skill_match'] = skill_match
        
        # Nivel educativo
        education_map = {'técnico': 1, 'licenciatura': 2, 'maestría': 3, 'doctorado': 4}
        nivel_edu = str(data.get('nivel_educacion', 'licenciatura')).lower()
        features['nivel_educacion_num'] = education_map.get(nivel_edu, 2)
        
        # Salario por experiencia
        exp = max(features['años_experiencia'].iloc[0], 1)
        features['salario_por_exp'] = features['salario'].iloc[0] / exp
        
        # Certificaciones
        cert = str(data.get('certificaciones', '')).lower()
        features['tiene_certificaciones'] = 1 if cert and cert not in ['', 'sin certificacion', 'ninguna'] else 0
        
        # Número de habilidades
        features['num_habilidades'] = len([s.strip() for s in skills.split(',') if s.strip()]) if skills else 0
        
        # Mes
        features['mes'] = df['fecha_postulacion'].dt.month.fillna(6).iloc[0]
        
        return features
    
    def predict(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza predicción"""
        if not self.is_loaded:
            raise ValueError("Modelo no cargado")
        
        try:
            # Preprocesar
            features = self.preprocess_application(application_data)
            
            # Asegurar orden correcto de features
            feature_array = features[self.feature_names].values.reshape(1, -1)
            
            # Escalar
            feature_scaled = self.scaler.transform(feature_array)
            
            # Predecir
            prediction = self.model.predict(feature_scaled)[0]
            probability = self.model.predict_proba(feature_scaled)[0, 1]
            
            # Interpretar resultado
            confidence = self._get_confidence_level(probability)
            recommendation = self._get_recommendation(probability)
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence_level': confidence,
                'recommendation': recommendation,
                'model_used': self.model_name
            }
            
            logger.info(f"Predicción: {probability:.3f} probabilidad de contacto")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determina nivel de confianza"""
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
        """Genera recomendación"""
        if probability >= 0.7:
            return "Fuertemente recomendado para entrevista"
        elif probability >= 0.5:
            return "Recomendado para entrevista"
        elif probability >= 0.3:
            return "Considerar para entrevista"
        else:
            return "No recomendado en esta ronda"

def test_simple_predictor():
    """Prueba el predictor simple"""
    print("=== Prueba del Predictor Simple ===")
    
    # Datos de ejemplo
    test_data = {
        'nombre': 'Test User',
        'años_experiencia': 5,
        'nivel_educacion': 'maestría',
        'habilidades': 'python, machine learning, sql',
        'idiomas': 'español, inglés',
        'certificaciones': 'aws cloud practitioner',
        'puesto_actual': 'data scientist',
        'industria': 'tecnología',
        'titulo': 'Senior Data Scientist',
        'descripcion': 'Posición senior en data science',
        'salario': 12000,
        'ubicacion': 'santa cruz',
        'requisitos': 'python, machine learning, sql, aws',
        'fecha_postulacion': '2024-01-15',
        'fecha_publicacion': '2024-01-10'
    }
    
    try:
        # Crear predictor
        predictor = SimpleHiringPredictor("trained_models/simple_hiring_model.pkl")
        
        # Realizar predicción
        result = predictor.predict(test_data)
        
        print("✅ Predicción exitosa!")
        print(f"Probabilidad: {result['probability']:.1%}")
        print(f"Confianza: {result['confidence_level']}")
        print(f"Recomendación: {result['recommendation']}")
        print(f"Modelo usado: {result['model_used']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_simple_predictor()