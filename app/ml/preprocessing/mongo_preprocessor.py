"""
Módulo de preprocessado de datos para predicción de contratación
Versión optimizada para datos de MongoDB
"""
import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata
import re
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDataPreprocessor:
    """Clase para preprocessar datos de candidatos y ofertas desde MongoDB"""
    
    def __init__(self):
        # Inicializar transformadores
        self.tfidf_skills = TfidfVectorizer(max_features=100, ngram_range=(1, 2), 
                                          stop_words=['y', 'e', 'o', 'de', 'la', 'el'])
        self.tfidf_requirements = TfidfVectorizer(max_features=100, ngram_range=(1, 2), 
                                                stop_words=['y', 'e', 'o', 'de', 'la', 'el'])
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
    
    def normalize_text(self, text: str) -> str:
        """Normaliza texto removiendo acentos y convirtiendo a minúsculas"""
        if pd.isna(text) or text == '':
            return ""
        
        # Convertir a string y minúsculas
        text = str(text).lower().strip()
        
        # Remover acentos
        text = unicodedata.normalize('NFKD', text)
        text = "".join([c for c in text if not unicodedata.combining(c)])
        
        # Limpiar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def parse_skills_list(self, skills_str: str, separator: str = ',') -> List[str]:
        """Convierte string de habilidades en lista limpia"""
        if pd.isna(skills_str) or skills_str == '':
            return []
        
        skills = [skill.strip() for skill in str(skills_str).split(separator)]
        skills = [self.normalize_text(skill) for skill in skills if skill.strip()]
        
        return skills
    
    def preprocess_data(self, df: pd.DataFrame, fit_transformers: bool = True) -> pd.DataFrame:
        """Preprocessa datos específicos de MongoDB para el modelo de compatibilidad"""
        logger.info(f"Iniciando preprocessamiento de datos MongoDB: {df.shape}")
        
        if df.empty:
            logger.warning("DataFrame vacío recibido")
            return df
        
        df_processed = df.copy()
        
        # 1. Limpiar y normalizar campos de texto
        text_fields = ['skills', 'languages', 'certifications', 'education_level', 
                      'current_position', 'job_title', 'location', 'requirements']
        
        for field in text_fields:
            if field in df_processed.columns:
                df_processed[field] = df_processed[field].fillna('').apply(self.normalize_text)
        
        # 2. Procesar campos numéricos
        numeric_fields = ['years_experience', 'salary']
        for field in numeric_fields:
            if field in df_processed.columns:
                df_processed[field] = pd.to_numeric(df_processed[field], errors='coerce').fillna(0)
        
        # 3. Crear features de compatibilidad
        df_processed = self._create_compatibility_features(df_processed)
        
        # 3.5. Crear features mejoradas para candidatos junior
        df_processed = self._create_improved_features(df_processed)
        
        # 4. Vectorización de texto
        if fit_transformers:
            df_processed = self._vectorize_text_features(df_processed, fit=True)
        else:
            df_processed = self._vectorize_text_features(df_processed, fit=False)
        
        # 5. Encoding de variables categóricas
        df_processed = self._encode_categorical_features(df_processed, fit_transformers)
        
        # 6. Normalización de features numéricas
        df_processed = self._normalize_numeric_features(df_processed, fit_transformers)
        
        # 7. Seleccionar features finales
        df_processed = self._select_final_features(df_processed)
        
        if fit_transformers:
            self.is_fitted = True
        
        logger.info(f"Preprocessamiento completado: {df_processed.shape}")
        return df_processed
    
    def _create_compatibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de compatibilidad entre candidato y oferta"""
        
        # Skills overlap
        if 'skills' in df.columns and 'requirements' in df.columns:
            df['skills_overlap'] = df.apply(
                lambda row: self._calculate_text_similarity(row['skills'], row['requirements']), 
                axis=1
            )
        
        # Experience vs salary ratio
        if 'years_experience' in df.columns and 'salary' in df.columns:
            df['salary_per_experience'] = df.apply(
                lambda row: row['salary'] / max(row['years_experience'], 1), axis=1
            )
        
        # Skills count
        if 'skills' in df.columns:
            df['skills_count'] = df['skills'].apply(
                lambda x: len(self.parse_skills_list(str(x)))
            )
        
        # Languages count
        if 'languages' in df.columns:
            df['languages_count'] = df['languages'].apply(
                lambda x: len(self.parse_skills_list(str(x)))
            )
        
        # Has certifications
        if 'certifications' in df.columns:
            df['has_certifications'] = df['certifications'].apply(
                lambda x: 1 if x and str(x).lower() not in ['', 'sin certificacion', 'ninguna'] else 0
            )
        
        # Education level encoding (simplificado)
        if 'education_level' in df.columns:
            education_mapping = {
                'bachillerato': 1, 'tecnico': 2, 'universidad': 3, 'ingenier': 4, 
                'licenciatura': 4, 'maestr': 5, 'doctor': 6, 'phd': 6
            }
            df['education_score'] = df['education_level'].apply(
                lambda x: self._map_education_level(str(x).lower(), education_mapping)
            )
        
        # Job title vs current position similarity
        if 'job_title' in df.columns and 'current_position' in df.columns:
            df['position_similarity'] = df.apply(
                lambda row: self._calculate_text_similarity(row['job_title'], row['current_position']), 
                axis=1
            )
        
        return df
    
    def _create_improved_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features adicionales para mejorar evaluación de candidatos junior"""
        
        # Peso especial para candidatos junior (0-2 años)
        if 'years_experience' in df.columns:
            df['is_junior'] = (df['years_experience'] <= 2).astype(int)
        else:
            df['is_junior'] = 0
        
        # Boost para educación técnica en candidatos junior
        if 'education_score' in df.columns and 'is_junior' in df.columns:
            df['junior_education_boost'] = df['education_score'] * df['is_junior'] * 1.5
        else:
            df['junior_education_boost'] = 0
        
        # Skills relevantes vs experience ratio
        if 'skills_overlap' in df.columns and 'years_experience' in df.columns:
            df['skills_to_experience_ratio'] = (
                df['skills_overlap'] / (df['years_experience'] + 1)
            )
        else:
            df['skills_to_experience_ratio'] = 0
        
        # Peso para certificaciones en candidatos junior
        if 'has_certifications' in df.columns and 'is_junior' in df.columns:
            df['junior_cert_boost'] = df['has_certifications'] * df['is_junior'] * 2
        else:
            df['junior_cert_boost'] = 0
        
        # Ajuste salarial para junior positions
        if 'is_junior' in df.columns and 'salary_per_experience' in df.columns:
            df['salary_expectation_realistic'] = np.where(
                (df['is_junior'] == 1) & (df['salary_per_experience'] < 50000),
                1.2, 1.0
            )
        else:
            df['salary_expectation_realistic'] = 1.0
        
        # Simular detección de tecnologías modernas
        if 'skills_overlap' in df.columns:
            df['modern_tech_score'] = np.random.random(len(df)) * df['skills_overlap']
        else:
            df['modern_tech_score'] = 0
            
        # Boost para stack tecnológico moderno en juniors
        if 'modern_tech_score' in df.columns and 'is_junior' in df.columns:
            df['junior_modern_boost'] = df['modern_tech_score'] * df['is_junior'] * 1.3
        else:
            df['junior_modern_boost'] = 0
        
        return df
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcula similitud simple entre dos textos"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _map_education_level(self, education: str, mapping: Dict) -> int:
        """Mapea nivel de educación a score numérico"""
        for key, value in mapping.items():
            if key in education:
                return value
        return 1  # Valor por defecto
    
    def _vectorize_text_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Vectoriza features de texto usando TF-IDF"""
        
        # Vectorizar skills
        if 'skills' in df.columns:
            skills_text = df['skills'].fillna('').astype(str)
            if fit:
                skills_tfidf = self.tfidf_skills.fit_transform(skills_text)
            else:
                skills_tfidf = self.tfidf_skills.transform(skills_text)
            
            # Añadir features TF-IDF de skills
            skills_feature_names = [f'skills_tfidf_{i}' for i in range(skills_tfidf.shape[1])]
            skills_df = pd.DataFrame(skills_tfidf.toarray(), columns=skills_feature_names, index=df.index)
            df = pd.concat([df, skills_df], axis=1)
        
        # Vectorizar requirements
        if 'requirements' in df.columns:
            req_text = df['requirements'].fillna('').astype(str)
            if fit:
                req_tfidf = self.tfidf_requirements.fit_transform(req_text)
            else:
                req_tfidf = self.tfidf_requirements.transform(req_text)
            
            # Añadir features TF-IDF de requirements
            req_feature_names = [f'requirements_tfidf_{i}' for i in range(req_tfidf.shape[1])]
            req_df = pd.DataFrame(req_tfidf.toarray(), columns=req_feature_names, index=df.index)
            df = pd.concat([df, req_df], axis=1)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Codifica variables categóricas"""
        categorical_features = ['location', 'current_position']
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                    
                    # Asegurar que 'unknown' esté en los datos de entrenamiento
                    values = df[feature].fillna('unknown').astype(str)
                    if 'unknown' not in values.unique():
                        # Agregar 'unknown' al conjunto de entrenamiento
                        values = pd.concat([values, pd.Series(['unknown'])])
                    
                    df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(values)[:len(df)]
                else:
                    if feature in self.label_encoders:
                        # Para datos nuevos, manejar categorías no vistas
                        df[feature] = df[feature].fillna('unknown').astype(str)
                        
                        # Mapear valores no conocidos a 'unknown'
                        known_values = set(self.label_encoders[feature].classes_)
                        df[feature] = df[feature].apply(
                            lambda x: x if x in known_values else 'unknown'
                        )
                        
                        try:
                            df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature])
                        except ValueError as e:
                            logger.warning(f"Error encoding {feature}: {e}. Usando valor por defecto.")
                            # Si aún hay problemas, asignar el valor de 'unknown'
                            unknown_encoded = 0
                            if 'unknown' in self.label_encoders[feature].classes_:
                                unknown_encoded = self.label_encoders[feature].transform(['unknown'])[0]
                            df[f'{feature}_encoded'] = unknown_encoded
                    else:
                        # Si no hay encoder, asignar 0
                        df[f'{feature}_encoded'] = 0
        
        return df
    
    def _normalize_numeric_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normaliza features numéricas"""
        numeric_features = [
            'years_experience', 'salary', 'skills_count', 'languages_count',
            'education_score', 'skills_overlap', 'salary_per_experience', 'position_similarity'
        ]
        
        # Filtrar features que existen en el DataFrame
        existing_numeric_features = [f for f in numeric_features if f in df.columns]
        
        if existing_numeric_features:
            if fit:
                df[existing_numeric_features] = self.scaler.fit_transform(
                    df[existing_numeric_features].fillna(0)
                )
            else:
                df[existing_numeric_features] = self.scaler.transform(
                    df[existing_numeric_features].fillna(0)
                )
        
        return df
    
    def _select_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selecciona las features finales para el modelo"""
        # Features base
        base_features = [
            'years_experience', 'salary', 'skills_count', 'languages_count',
            'has_certifications', 'education_score', 'skills_overlap', 
            'salary_per_experience', 'position_similarity'
        ]
        
        # Features de encoding
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        
        # Features TF-IDF
        tfidf_features = [col for col in df.columns if 'tfidf' in col]
        
        # Combinar todas las features
        all_features = base_features + encoded_features + tfidf_features
        
        # Filtrar features que existen
        existing_features = [f for f in all_features if f in df.columns]
        
        # Incluir target si existe
        if 'target' in df.columns:
            existing_features.append('target')
        
        # Incluir IDs si existen
        id_features = ['candidate_id', 'offer_id']
        for id_feat in id_features:
            if id_feat in df.columns:
                existing_features.append(id_feat)
        
        return df[existing_features]
    
    def save_preprocessor(self, filepath: str):
        """Guarda el preprocessor entrenado"""
        preprocessor_data = {
            'tfidf_skills': self.tfidf_skills,
            'tfidf_requirements': self.tfidf_requirements,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'is_fitted': self.is_fitted
        }
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor guardado en: {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Carga un preprocessor entrenado"""
        if os.path.exists(filepath):
            preprocessor_data = joblib.load(filepath)
            
            self.tfidf_skills = preprocessor_data['tfidf_skills']
            self.tfidf_requirements = preprocessor_data['tfidf_requirements']
            self.scaler = preprocessor_data['scaler']
            self.label_encoders = preprocessor_data['label_encoders']
            self.is_fitted = preprocessor_data['is_fitted']
            
            logger.info(f"Preprocessor cargado desde: {filepath}")
        else:
            logger.warning(f"Archivo de preprocessor no encontrado: {filepath}")
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Retorna resumen de features del dataset"""
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else {},
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'feature_names': df.columns.tolist()
        }
        
        return summary


# Instancia global del preprocessor
mongo_preprocessor = MongoDataPreprocessor()


def preprocess_mongodb_data(df: pd.DataFrame, fit_transformers: bool = True) -> pd.DataFrame:
    """Función conveniente para preprocessar datos de MongoDB"""
    return mongo_preprocessor.preprocess_data(df, fit_transformers)


def get_feature_summary(df: pd.DataFrame) -> Dict:
    """Función conveniente para obtener resumen de features"""
    return mongo_preprocessor.get_feature_summary(df)


if __name__ == "__main__":
    # Test del preprocessador
    print("MongoDataPreprocessor inicializado correctamente")