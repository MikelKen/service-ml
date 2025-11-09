#!/usr/bin/env python3
"""
🔧 PREPROCESSOR PARA DATOS DE POSTULACIONES (MODELO SEMI-SUPERVISADO)
Prepara datos combinados de candidatos y ofertas para predicción de estado
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
import re
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostulationPreprocessor:
    """Preprocessor especializado para datos de postulaciones candidato-oferta"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Configuración TF-IDF
        self.tfidf_config = {
            'max_features': 50,  # Reducido para eficiencia
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 1,
            'max_df': 0.95,
            'lowercase': True
        }
        
        # Mapeos de categorías
        self.education_levels = {
            'técnico': 1, 'tecnico': 1,
            'licenciatura': 2, 'bachelor': 2,
            'ingeniería': 3, 'ingenieria': 3, 'engineering': 3,
            'maestría': 4, 'maestria': 4, 'master': 4,
            'doctorado': 5, 'phd': 5, 'doctorate': 5
        }
        
        self.seniority_levels = {
            'intern': 1, 'practicante': 1,
            'junior': 2, 'jr': 2,
            'developer': 3, 'desarrollador': 3,
            'senior': 4, 'sr': 4,
            'lead': 5, 'lider': 5,
            'manager': 6, 'gerente': 6,
            'director': 7
        }
        
        # Tecnologías relevantes (para matching)
        self.tech_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'sql', 'mongodb', 'postgresql', 'mysql', 'docker', 'kubernetes',
            'aws', 'azure', 'git', 'linux', 'windows', 'agile', 'scrum'
        ]
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Ajusta el preprocessor y transforma los datos"""
        logger.info("🚀 INICIANDO PREPROCESSING PARA POSTULACIONES")
        logger.info(f"📊 Dataset: {df.shape[0]} registros, {df.shape[1]} campos")
        
        # Extraer diferentes tipos de características
        candidate_features = self._extract_candidate_features(df)
        offer_features = self._extract_offer_features(df)
        compatibility_features = self._extract_compatibility_features(df)
        
        # Combinar todas las características
        all_features = pd.concat([
            candidate_features,
            offer_features,
            compatibility_features
        ], axis=1)
        
        logger.info(f"🔗 Características extraídas: {all_features.shape[1]} features totales")
        
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
        candidate_features = self._extract_candidate_features(df)
        offer_features = self._extract_offer_features(df)
        compatibility_features = self._extract_compatibility_features(df)
        
        all_features = pd.concat([
            candidate_features,
            offer_features,
            compatibility_features
        ], axis=1)
        
        # Asegurar mismas columnas
        for col in self.feature_names:
            if col not in all_features.columns:
                all_features[col] = 0
        
        all_features = all_features[self.feature_names]
        
        return self.scaler.transform(all_features.fillna(0))
    
    def _extract_candidate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae características del candidato"""
        logger.info("👤 Extrayendo características del candidato...")
        
        features = pd.DataFrame()
        
        # === CARACTERÍSTICAS NUMÉRICAS DEL CANDIDATO ===
        
        # Años de experiencia
        features['candidate_years_experience'] = df['years_experience'].fillna(0)
        
        # Nivel de educación (convertir a score)
        features['candidate_education_score'] = df['education_level'].apply(
            self._education_to_score
        )
        
        # Nivel de seniority del puesto actual
        features['candidate_seniority_score'] = df['current_position'].apply(
            self._position_to_seniority
        )
        
        # Cantidad de idiomas
        features['candidate_num_languages'] = df['languages'].apply(
            self._count_languages
        )
        
        # Nivel de inglés
        features['candidate_english_level'] = df['languages'].apply(
            self._extract_english_level
        )
        
        # Cantidad de certificaciones
        features['candidate_num_certifications'] = df['certifications'].apply(
            self._count_certifications
        )
        
        # === CARACTERÍSTICAS DE TEXTO DEL CANDIDATO ===
        
        # Skills TF-IDF
        self._process_text_field(df, 'skills', 'candidate_skills', features)
        
        # Certifications TF-IDF
        self._process_text_field(df, 'certifications', 'candidate_certs', features)
        
        logger.info(f"✅ Características del candidato: {features.shape[1]} features")
        return features
    
    def _extract_offer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae características de la oferta"""
        logger.info("💼 Extrayendo características de la oferta...")
        
        features = pd.DataFrame()
        
        # === CARACTERÍSTICAS NUMÉRICAS DE LA OFERTA ===
        
        # Salario (normalizado)
        features['offer_salary'] = df['salary'].fillna(0)
        
        # Seniority requerido en el título del trabajo
        features['offer_seniority_required'] = df['job_title'].apply(
            self._position_to_seniority
        )
        
        # Longitud de requisitos (proxy de complejidad)
        features['offer_requirements_length'] = df['requirements'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        
        # === CARACTERÍSTICAS CATEGÓRICAS DE LA OFERTA ===
        
        # Área del trabajo
        offer_area = df['job_title'].apply(self._extract_job_area)
        features['offer_area'] = self._encode_categorical(offer_area, 'offer_area')
        
        # Ubicación
        offer_location = df['location'].fillna('unknown')
        features['offer_location'] = self._encode_categorical(offer_location, 'offer_location')
        
        # === CARACTERÍSTICAS DE TEXTO DE LA OFERTA ===
        
        # Requirements TF-IDF
        self._process_text_field(df, 'requirements', 'offer_reqs', features)
        
        # Job title TF-IDF
        self._process_text_field(df, 'job_title', 'offer_title', features)
        
        logger.info(f"✅ Características de la oferta: {features.shape[1]} features")
        return features
    
    def _extract_compatibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae características de compatibilidad candidato-oferta"""
        logger.info("🔗 Extrayendo características de compatibilidad...")
        
        features = pd.DataFrame()
        
        # === COMPATIBILIDAD DE EXPERIENCIA ===
        
        # Diferencia entre experiencia del candidato y seniority requerido
        candidate_exp = df['years_experience'].fillna(0)
        offer_seniority = df['job_title'].apply(self._position_to_seniority)
        
        # Mapear seniority a años esperados (estimación)
        seniority_to_years = {1: 0, 2: 1, 3: 3, 4: 5, 5: 7, 6: 10, 7: 15}
        expected_years = offer_seniority.map(seniority_to_years).fillna(3)
        
        features['experience_match'] = (candidate_exp - expected_years).apply(
            lambda x: max(0, min(1, (x + 2) / 4))  # Normalizar a [0,1]
        )
        
        # === COMPATIBILIDAD DE SKILLS ===
        
        # Matching de tecnologías
        features['tech_skill_match'] = df.apply(
            lambda row: self._calculate_tech_match(
                str(row['skills']), str(row['requirements'])
            ), axis=1
        )
        
        # === COMPATIBILIDAD DE EDUCACIÓN ===
        
        # Score de educación vs requerimientos del puesto
        candidate_edu_score = df['education_level'].apply(self._education_to_score)
        offer_edu_required = df['requirements'].apply(self._extract_education_requirement)
        
        features['education_match'] = (
            candidate_edu_score >= offer_edu_required
        ).astype(int)
        
        # === COMPATIBILIDAD DE UBICACIÓN ===
        
        # Simplificado: mismo ciudad/región
        features['location_match'] = 1  # Por defecto, asumir compatibilidad
        
        # === COMPATIBILIDAD DE SALARIO ===
        
        # Proxy: salarios más altos pueden ser más atractivos
        salary_normalized = df['salary'].fillna(0) / (df['salary'].max() or 1)
        features['salary_attractiveness'] = salary_normalized
        
        logger.info(f"✅ Características de compatibilidad: {features.shape[1]} features")
        return features
    
    def _process_text_field(self, df: pd.DataFrame, field: str, prefix: str, 
                           features: pd.DataFrame):
        """Procesa un campo de texto con TF-IDF"""
        try:
            # Limpiar y preparar texto
            text_data = df[field].fillna('').apply(self._clean_text)
            
            # Vectorizador TF-IDF
            if f'{prefix}_tfidf' not in self.tfidf_vectorizers:
                self.tfidf_vectorizers[f'{prefix}_tfidf'] = TfidfVectorizer(**self.tfidf_config)
            
            # Ajustar y transformar solo si hay texto
            if text_data.str.len().sum() > 0:
                tfidf_matrix = self.tfidf_vectorizers[f'{prefix}_tfidf'].fit_transform(text_data)
                
                # Convertir a DataFrame
                feature_names = [f'{prefix}_{name}' for name in 
                               self.tfidf_vectorizers[f'{prefix}_tfidf'].get_feature_names_out()]
                
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(), 
                    columns=feature_names,
                    index=df.index
                )
                
                # Agregar al DataFrame de características
                for col in tfidf_df.columns:
                    features[col] = tfidf_df[col]
                
                logger.info(f"  📝 {field}: {tfidf_matrix.shape[1]} features TF-IDF")
            else:
                logger.warning(f"  ⚠️ {field}: No hay texto para procesar")
                
        except Exception as e:
            logger.warning(f"  ⚠️ Error procesando {field}: {e}")
    
    # === MÉTODOS AUXILIARES ===
    
    def _education_to_score(self, education: str) -> int:
        """Convierte nivel educativo a score"""
        if pd.isna(education):
            return 0
        
        education_lower = str(education).lower()
        for level, score in self.education_levels.items():
            if level in education_lower:
                return score
        return 2  # Default: licenciatura
    
    def _position_to_seniority(self, position: str) -> int:
        """Extrae nivel de seniority del puesto"""
        if pd.isna(position):
            return 0
        
        position_lower = str(position).lower()
        for level, score in self.seniority_levels.items():
            if level in position_lower:
                return score
        return 3  # Default: developer
    
    def _count_languages(self, languages: str) -> int:
        """Cuenta cantidad de idiomas"""
        if pd.isna(languages):
            return 0
        
        # Buscar patrones como "Español (nativo)", "Inglés (intermedio)"
        pattern = r'\w+\s*\([^)]+\)'
        matches = re.findall(pattern, str(languages))
        return len(matches) if matches else 1  # Al menos 1 si hay texto
    
    def _extract_english_level(self, languages: str) -> int:
        """Extrae nivel de inglés"""
        if pd.isna(languages):
            return 0
        
        languages_lower = str(languages).lower()
        if 'inglés' in languages_lower or 'english' in languages_lower:
            if 'avanzado' in languages_lower or 'advanced' in languages_lower:
                return 3
            elif 'intermedio' in languages_lower or 'intermediate' in languages_lower:
                return 2
            elif 'básico' in languages_lower or 'basic' in languages_lower:
                return 1
            else:
                return 2  # Default intermedio si se menciona inglés
        return 0
    
    def _count_certifications(self, certifications: str) -> int:
        """Cuenta cantidad de certificaciones"""
        if pd.isna(certifications):
            return 0
        
        text = str(certifications)
        # Contar por separadores comunes
        separators = [',', ';', '\n', '|']
        max_count = 1
        
        for sep in separators:
            count = len([x for x in text.split(sep) if x.strip()])
            max_count = max(max_count, count)
        
        return max_count
    
    def _extract_job_area(self, job_title: str) -> str:
        """Extrae área del trabajo"""
        if pd.isna(job_title):
            return 'unknown'
        
        title_lower = str(job_title).lower()
        
        if any(x in title_lower for x in ['developer', 'desarrollador', 'programmer', 'dev']):
            return 'desarrollo'
        elif any(x in title_lower for x in ['manager', 'gerente', 'director', 'lead']):
            return 'management'
        elif any(x in title_lower for x in ['analyst', 'analista', 'data']):
            return 'analisis'
        elif any(x in title_lower for x in ['designer', 'diseñador', 'ui', 'ux']):
            return 'diseno'
        elif any(x in title_lower for x in ['qa', 'tester', 'quality']):
            return 'testing'
        else:
            return 'otros'
    
    def _calculate_tech_match(self, candidate_skills: str, offer_requirements: str) -> float:
        """Calcula matching de tecnologías entre candidato y oferta"""
        if pd.isna(candidate_skills) or pd.isna(offer_requirements):
            return 0.0
        
        skills_lower = str(candidate_skills).lower()
        reqs_lower = str(offer_requirements).lower()
        
        matches = 0
        total_required = 0
        
        for tech in self.tech_keywords:
            if tech in reqs_lower:
                total_required += 1
                if tech in skills_lower:
                    matches += 1
        
        return matches / total_required if total_required > 0 else 0.5
    
    def _extract_education_requirement(self, requirements: str) -> int:
        """Extrae requerimiento de educación de la oferta"""
        if pd.isna(requirements):
            return 2  # Default: licenciatura
        
        reqs_lower = str(requirements).lower()
        
        if any(x in reqs_lower for x in ['doctorado', 'phd']):
            return 5
        elif any(x in reqs_lower for x in ['maestría', 'maestria', 'master']):
            return 4
        elif any(x in reqs_lower for x in ['ingeniería', 'ingenieria', 'engineering']):
            return 3
        elif any(x in reqs_lower for x in ['licenciatura', 'bachelor', 'universitario']):
            return 2
        elif any(x in reqs_lower for x in ['técnico', 'tecnico']):
            return 1
        else:
            return 2  # Default
    
    def _encode_categorical(self, series: pd.Series, column_name: str) -> pd.Series:
        """Codifica variable categórica"""
        if column_name not in self.label_encoders:
            self.label_encoders[column_name] = LabelEncoder()
        
        # Manejar valores faltantes
        series_filled = series.fillna('unknown')
        return pd.Series(
            self.label_encoders[column_name].fit_transform(series_filled),
            index=series.index
        )
    
    def _clean_text(self, text: str) -> str:
        """Limpia texto para TF-IDF"""
        if pd.isna(text):
            return ''
        
        # Convertir a lowercase
        text = str(text).lower()
        
        # Remover caracteres especiales, mantener + y #
        text = re.sub(r'[^a-zA-Z0-9\s\+\#]', ' ', text)
        
        # Remover espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_feature_names(self) -> List[str]:
        """Retorna nombres de las características"""
        return self.feature_names.copy()
    
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
            'seniority_levels': self.seniority_levels,
            'tech_keywords': self.tech_keywords,
            'training_date': datetime.now().isoformat()
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
        preprocessor.tech_keywords = preprocessor_data['tech_keywords']
        
        logger.info(f"📂 Preprocessor cargado desde: {filepath}")
        logger.info(f"🗓️ Entrenado el: {preprocessor_data.get('training_date', 'fecha desconocida')}")
        
        return preprocessor

if __name__ == "__main__":
    # Ejemplo de uso
    print("🔧 PostulationPreprocessor creado")
    print("✅ Listo para procesar datos de postulaciones")
    print("🚀 Siguiente paso: Entrenar modelo semi-supervisado")