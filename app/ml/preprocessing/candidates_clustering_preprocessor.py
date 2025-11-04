#!/usr/bin/env python3
"""
🔧 PREPROCESSOR PARA CLUSTERING DE CANDIDATOS
Prepara los datos de candidates_features para clustering no supervisado
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re
from typing import Dict, List, Tuple, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandidatesClusteringPreprocessor:
    """Preprocessor especializado para clustering de candidatos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Configuración TF-IDF
        self.tfidf_config = {
            'max_features': 100,  # Reducido para clustering
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95
        }
        
        # Mapeos de categorías
        self.education_levels = {
            'técnico': 1, 'licenciatura': 2, 'ingeniería': 3, 
            'maestría': 4, 'doctorado': 5
        }
        
        self.seniority_levels = {
            'intern': 1, 'junior': 2, 'developer': 3, 
            'senior': 4, 'lead': 5, 'manager': 6, 'director': 7
        }
    
    def _extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae características numéricas"""
        logger.info("🔢 Extrayendo características numéricas...")
        
        numeric_features = pd.DataFrame()
        
        # Años de experiencia (ya numérico)
        numeric_features['anios_experiencia'] = df['anios_experiencia'].fillna(0)
        
        # Nivel de educación (convertir a ordinal)
        numeric_features['nivel_educacion_score'] = df['nivel_educacion'].apply(
            self._education_to_score
        )
        
        # Nivel de seniority del puesto actual
        numeric_features['seniority_score'] = df['puesto_actual'].apply(
            self._position_to_seniority
        )
        
        # Cantidad de idiomas
        numeric_features['num_idiomas'] = df['idiomas'].apply(
            lambda x: len(re.findall(r'\w+\s*\([^)]+\)', str(x))) if pd.notna(x) else 0
        )
        
        # Nivel de inglés (importante para clustering)
        numeric_features['nivel_ingles'] = df['idiomas'].apply(
            self._extract_english_level
        )
        
        logger.info(f"✅ Características numéricas extraídas: {numeric_features.shape[1]} features")
        return numeric_features
    
    def _extract_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae y codifica características categóricas"""
        logger.info("🏷️ Extrayendo características categóricas...")
        
        categorical_features = pd.DataFrame()
        
        # Área de educación
        categorical_features['area_educacion'] = df['nivel_educacion'].apply(
            self._extract_education_area
        )
        
        # Área de trabajo actual
        categorical_features['area_trabajo'] = df['puesto_actual'].apply(
            self._extract_work_area
        )
        
        # Codificar categóricas con LabelEncoder
        for col in categorical_features.columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            # Manejar valores faltantes
            categorical_features[col] = categorical_features[col].fillna('unknown')
            categorical_features[col] = self.label_encoders[col].fit_transform(categorical_features[col])
        
        logger.info(f"✅ Características categóricas extraídas: {categorical_features.shape[1]} features")
        return categorical_features
    
    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae características de texto usando TF-IDF"""
        logger.info("📝 Extrayendo características de texto...")
        
        text_features = pd.DataFrame()
        
        # Preparar campos de texto
        text_fields = {
            'habilidades': 'skills',
            'certificaciones': 'certs'
        }
        
        for field, prefix in text_fields.items():
            logger.info(f"  🔍 Procesando {field}...")
            
            # Limpiar y preparar texto
            text_data = df[field].fillna('').apply(self._clean_text)
            
            # Vectorizador TF-IDF
            if f'{prefix}_tfidf' not in self.tfidf_vectorizers:
                self.tfidf_vectorizers[f'{prefix}_tfidf'] = TfidfVectorizer(**self.tfidf_config)
            
            # Ajustar y transformar
            try:
                tfidf_matrix = self.tfidf_vectorizers[f'{prefix}_tfidf'].fit_transform(text_data)
                
                # Convertir a DataFrame
                feature_names = [f'{prefix}_{name}' for name in 
                               self.tfidf_vectorizers[f'{prefix}_tfidf'].get_feature_names_out()]
                
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(), 
                    columns=feature_names,
                    index=df.index
                )
                
                # Concatenar características
                text_features = pd.concat([text_features, tfidf_df], axis=1)
                
                logger.info(f"  ✅ {field}: {tfidf_matrix.shape[1]} features extraídas")
                
            except Exception as e:
                logger.warning(f"  ⚠️ Error procesando {field}: {e}")
        
        logger.info(f"✅ Total características de texto: {text_features.shape[1]} features")
        return text_features
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Ajusta el preprocessor y transforma los datos"""
        logger.info("🚀 INICIANDO PREPROCESSING PARA CLUSTERING")
        logger.info(f"📊 Dataset: {df.shape[0]} candidatos, {df.shape[1]} campos")
        
        # Extraer diferentes tipos de características
        numeric_features = self._extract_numeric_features(df)
        categorical_features = self._extract_categorical_features(df)
        text_features = self._extract_text_features(df)
        
        # Combinar todas las características
        all_features = pd.concat([
            numeric_features,
            categorical_features,
            text_features
        ], axis=1)
        
        logger.info(f"🔗 Características combinadas: {all_features.shape[1]} features totales")
        
        # Normalizar características
        logger.info("⚖️ Normalizando características...")
        scaled_features = self.scaler.fit_transform(all_features.fillna(0))
        
        # Guardar nombres de características
        self.feature_names = all_features.columns.tolist()
        self.is_fitted = True
        
        logger.info("✅ PREPROCESSING COMPLETADO")
        logger.info(f"📈 Matriz final: {scaled_features.shape}")
        
        return scaled_features
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforma nuevos datos usando el preprocessor ajustado"""
        if not self.is_fitted:
            raise ValueError("Preprocessor no ha sido ajustado. Usar fit_transform primero.")
        
        # Aplicar mismas transformaciones
        numeric_features = self._extract_numeric_features(df)
        categorical_features = self._extract_categorical_features(df)
        text_features = self._extract_text_features(df)
        
        all_features = pd.concat([
            numeric_features,
            categorical_features,
            text_features
        ], axis=1)
        
        # Asegurar mismas columnas
        for col in self.feature_names:
            if col not in all_features.columns:
                all_features[col] = 0
        
        all_features = all_features[self.feature_names]
        
        return self.scaler.transform(all_features.fillna(0))
    
    # === MÉTODOS AUXILIARES ===
    
    def _education_to_score(self, education: str) -> int:
        """Convierte nivel educativo a score"""
        if pd.isna(education):
            return 0
        
        education_lower = education.lower()
        for level, score in self.education_levels.items():
            if level in education_lower:
                return score
        return 2  # Default: licenciatura
    
    def _position_to_seniority(self, position: str) -> int:
        """Extrae nivel de seniority del puesto"""
        if pd.isna(position):
            return 0
        
        position_lower = position.lower()
        for level, score in self.seniority_levels.items():
            if level in position_lower:
                return score
        return 3  # Default: developer
    
    def _extract_english_level(self, languages: str) -> int:
        """Extrae nivel de inglés"""
        if pd.isna(languages):
            return 0
        
        languages_lower = languages.lower()
        if 'inglés' in languages_lower or 'english' in languages_lower:
            if 'avanzado' in languages_lower or 'advanced' in languages_lower:
                return 3
            elif 'intermedio' in languages_lower or 'intermediate' in languages_lower:
                return 2
            elif 'básico' in languages_lower or 'basic' in languages_lower:
                return 1
        return 0
    
    def _extract_education_area(self, education: str) -> str:
        """Extrae área de educación"""
        if pd.isna(education):
            return 'unknown'
        
        education_lower = education.lower()
        
        if any(x in education_lower for x in ['sistemas', 'computación', 'software', 'informática']):
            return 'sistemas'
        elif any(x in education_lower for x in ['industrial', 'administración', 'comercial']):
            return 'industrial'
        elif any(x in education_lower for x in ['telecomunicaciones', 'electrónica', 'eléctrica']):
            return 'telecomunicaciones'
        elif any(x in education_lower for x in ['matemáticas', 'estadística', 'inteligencia']):
            return 'matematicas'
        else:
            return 'otros'
    
    def _extract_work_area(self, position: str) -> str:
        """Extrae área de trabajo"""
        if pd.isna(position):
            return 'unknown'
        
        position_lower = position.lower()
        
        if any(x in position_lower for x in ['developer', 'desarrollador', 'programmer']):
            return 'desarrollo'
        elif any(x in position_lower for x in ['manager', 'gerente', 'director']):
            return 'management'
        elif any(x in position_lower for x in ['analyst', 'analista', 'data']):
            return 'analisis'
        elif any(x in position_lower for x in ['designer', 'diseñador', 'ui', 'ux']):
            return 'diseno'
        else:
            return 'otros'
    
    def _clean_text(self, text: str) -> str:
        """Limpia texto para TF-IDF"""
        if pd.isna(text):
            return ''
        
        # Convertir a lowercase
        text = str(text).lower()
        
        # Remover caracteres especiales
        text = re.sub(r'[^a-zA-Z0-9\s\+\#]', ' ', text)
        
        # Remover espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save_preprocessor(self, filepath: str):
        """Guarda el preprocessor entrenado"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'tfidf_vectorizers': self.tfidf_vectorizers,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'tfidf_config': self.tfidf_config,
            'education_levels': self.education_levels,
            'seniority_levels': self.seniority_levels
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"💾 Preprocessor guardado en: {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str):
        """Carga un preprocessor entrenado"""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.label_encoders = preprocessor_data['label_encoders']
        preprocessor.tfidf_vectorizers = preprocessor_data['tfidf_vectorizers']
        preprocessor.feature_names = preprocessor_data['feature_names']
        preprocessor.is_fitted = preprocessor_data['is_fitted']
        preprocessor.tfidf_config = preprocessor_data['tfidf_config']
        preprocessor.education_levels = preprocessor_data['education_levels']
        preprocessor.seniority_levels = preprocessor_data['seniority_levels']
        
        logger.info(f"📂 Preprocessor cargado desde: {filepath}")
        return preprocessor

if __name__ == "__main__":
    # Ejemplo de uso
    print("🔧 CandidatesClusteringPreprocessor creado")
    print("✅ Listo para procesar datos de candidatos")
    print("🚀 Siguiente paso: Entrenar modelo de clustering")