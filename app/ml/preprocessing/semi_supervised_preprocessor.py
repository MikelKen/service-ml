#!/usr/bin/env python3
"""
🔧 PREPROCESSOR PARA APRENDIZAJE SEMI-SUPERVISADO
Maneja datos etiquetados y no etiquetados para modelos semi-supervisados
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
import pickle
import json

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from app.config.mongodb_connection import get_mongodb_sync, mongodb_connection

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiSupervisedPreprocessor:
    """Preprocessor especializado para aprendizaje semi-supervisado"""
    
    def __init__(self):
        # Transformadores para features
        self.tfidf_skills = TfidfVectorizer(
            max_features=200, 
            ngram_range=(1, 2), 
            stop_words=['y', 'e', 'o', 'de', 'la', 'el', 'en', 'con', 'para'],
            min_df=2,
            max_df=0.95
        )
        self.tfidf_requirements = TfidfVectorizer(
            max_features=200, 
            ngram_range=(1, 2), 
            stop_words=['y', 'e', 'o', 'de', 'la', 'el', 'en', 'con', 'para'],
            min_df=2,
            max_df=0.95
        )
        
        # Escaladores
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Imputadores
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Encoder para variables categóricas
        self.label_encoders = {}
        
        # PCA para reducción de dimensionalidad
        self.pca = PCA(n_components=0.95)  # Mantener 95% de la varianza
        
        # Estado del preprocessor
        self.is_fitted = False
        self.feature_names = []
        self.feature_importance = {}
        
        # Configuración semi-supervisada
        self.labeled_threshold = 0.7  # Confianza mínima para pseudo-labels
        self.min_labeled_samples = 50  # Mínimo de muestras etiquetadas requeridas
        
        # Metadatos
        self.preprocessing_config = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'algorithm_type': 'semi_supervised',
            'feature_engineering': {
                'text_vectorization': True,
                'compatibility_features': True,
                'temporal_features': True,
                'profile_completeness': True
            }
        }
    
    def connect_mongodb(self):
        """Conectar a MongoDB"""
        try:
            mongodb_connection.connect_sync()
            self.db = get_mongodb_sync()
            logger.info("✅ Conectado a MongoDB")
        except Exception as e:
            logger.error(f"❌ Error conectando a MongoDB: {e}")
            raise
    
    def load_data_from_mongodb(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cargar datos desde MongoDB"""
        logger.info("📊 Cargando datos desde MongoDB...")
        
        # Cargar aplicaciones (dataset principal)
        applications_collection = self.db['ml_applications']
        applications_cursor = applications_collection.find({})
        applications_data = list(applications_cursor)
        
        if not applications_data:
            raise ValueError("No se encontraron datos de aplicaciones en MongoDB")
        
        applications_df = pd.DataFrame(applications_data)
        
        # Cargar candidatos
        candidates_collection = self.db['ml_candidates']
        candidates_cursor = candidates_collection.find({})
        candidates_data = list(candidates_cursor)
        candidates_df = pd.DataFrame(candidates_data)
        
        # Cargar ofertas
        offers_collection = self.db['ml_job_offers']
        offers_cursor = offers_collection.find({})
        offers_data = list(offers_cursor)
        offers_df = pd.DataFrame(offers_data)
        
        logger.info(f"📈 Datos cargados:")
        logger.info(f"  📝 Aplicaciones: {len(applications_df)}")
        logger.info(f"  👥 Candidatos: {len(candidates_df)}")
        logger.info(f"  💼 Ofertas: {len(offers_df)}")
        
        return applications_df, candidates_df, offers_df
    
    def extract_features_from_nested_data(self, df: pd.DataFrame, nested_column: str, prefix: str) -> pd.DataFrame:
        """Extrae features de columnas con datos anidados (dict)"""
        features_df = pd.DataFrame()
        
        for idx, row in df.iterrows():
            nested_data = row.get(nested_column, {})
            if isinstance(nested_data, dict):
                for key, value in nested_data.items():
                    column_name = f"{prefix}_{key}"
                    features_df.loc[idx, column_name] = value
            else:
                # Si no es dict, crear columnas vacías
                features_df.loc[idx, f"{prefix}_empty"] = 0
        
        return features_df
    
    def create_comprehensive_features(self, applications_df: pd.DataFrame, 
                                    candidates_df: pd.DataFrame, 
                                    offers_df: pd.DataFrame) -> pd.DataFrame:
        """Crear conjunto completo de features para el modelo"""
        logger.info("🔧 Creando features comprehensivas...")
        
        # Crear mapas para joins
        candidates_map = candidates_df.set_index('candidate_id').to_dict('index')
        offers_map = offers_df.set_index('offer_id').to_dict('index')
        
        # DataFrame resultado
        features_list = []
        
        for _, app_row in applications_df.iterrows():
            try:
                candidate_id = app_row['candidate_id']
                offer_id = app_row['offer_id']
                
                candidate_data = candidates_map.get(candidate_id, {})
                offer_data = offers_map.get(offer_id, {})
                
                # Features base de la aplicación
                features = {
                    'application_id': app_row['application_id'],
                    'candidate_id': candidate_id,
                    'offer_id': offer_id,
                    
                    # Target variable
                    'target': app_row.get('ml_target', -1),
                    'is_labeled': app_row.get('is_labeled', False),
                    'label_quality': app_row.get('label_quality', 'unknown'),
                    
                    # Features temporales
                    'days_since_application': (datetime.now() - pd.to_datetime(app_row.get('fecha_postulacion', datetime.now()))).days,
                }
                
                # Features de compatibilidad (ya calculadas)
                compatibility_features = app_row.get('compatibility_features', {})
                for key, value in compatibility_features.items():
                    features[f'compat_{key}'] = value
                
                # Features del candidato
                candidate_features = candidate_data.get('features', {})
                for key, value in candidate_features.items():
                    if key != 'skills_vector':  # Vectores se manejan aparte
                        features[f'candidate_{key}'] = value
                
                # Features básicas del candidato
                features.update({
                    'candidate_anos_experiencia': candidate_data.get('anos_experiencia', 0),
                    'candidate_num_habilidades': len(candidate_data.get('habilidades', [])),
                    'candidate_num_idiomas': len(candidate_data.get('idiomas', [])),
                    'candidate_num_certificaciones': len(candidate_data.get('certificaciones', [])),
                })
                
                # Features de la oferta
                offer_features = offer_data.get('features', {})
                for key, value in offer_features.items():
                    if key != 'requisitos_vector':  # Vectores se manejan aparte
                        features[f'offer_{key}'] = value
                
                # Features básicas de la oferta
                features.update({
                    'offer_salario': offer_data.get('salario', 0),
                    'offer_activa': offer_data.get('activa', False),
                    'offer_empresa_rubro_encoded': 0,  # Se codificará después
                })
                
                # Features derivadas
                features.update({
                    'experience_salary_ratio': features.get('candidate_anos_experiencia', 0) / max(features.get('offer_salario', 1), 1) * 1000,
                    'profile_offer_match': features.get('compat_overall_compatibility', 0) * features.get('candidate_profile_completeness', 0),
                    'experience_level_match': self._calculate_experience_level_match(
                        features.get('candidate_anos_experiencia', 0),
                        offer_data.get('features', {}).get('nivel_requisitos', 'mid')
                    ),
                })
                
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"❌ Error procesando aplicación {app_row.get('application_id', 'unknown')}: {e}")
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Features creadas: {features_df.shape}")
        
        return features_df
    
    def _calculate_experience_level_match(self, years_experience: int, job_level: str) -> float:
        """Calcula compatibilidad entre experiencia del candidato y nivel del trabajo"""
        if job_level == 'junior':
            if years_experience <= 2:
                return 1.0
            elif years_experience <= 5:
                return 0.7
            else:
                return 0.3
        elif job_level == 'mid':
            if 2 <= years_experience <= 7:
                return 1.0
            elif years_experience <= 10:
                return 0.8
            else:
                return 0.5
        elif job_level == 'senior':
            if years_experience >= 5:
                return 1.0
            elif years_experience >= 3:
                return 0.6
            else:
                return 0.2
        else:
            return 0.5  # Neutral para casos desconocidos
    
    def process_text_features(self, applications_df: pd.DataFrame, 
                            candidates_df: pd.DataFrame, 
                            offers_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Procesar features de texto (habilidades y requisitos)"""
        logger.info("📝 Procesando features de texto...")
        
        # Crear mapas
        candidates_map = candidates_df.set_index('candidate_id').to_dict('index')
        offers_map = offers_df.set_index('offer_id').to_dict('index')
        
        # Recopilar textos
        skills_texts = []
        requirements_texts = []
        
        for _, row in applications_df.iterrows():
            candidate_data = candidates_map.get(row['candidate_id'], {})
            offer_data = offers_map.get(row['offer_id'], {})
            
            # Texto de habilidades del candidato
            skills_text = candidate_data.get('habilidades_raw', '')
            skills_texts.append(str(skills_text) if skills_text else '')
            
            # Texto de requisitos de la oferta
            req_text = offer_data.get('requisitos', '')
            requirements_texts.append(str(req_text) if req_text else '')
        
        # Vectorizar textos
        if any(text.strip() for text in skills_texts):
            skills_vectors = self.tfidf_skills.fit_transform(skills_texts)
        else:
            skills_vectors = np.zeros((len(skills_texts), 200))
        
        if any(text.strip() for text in requirements_texts):
            requirements_vectors = self.tfidf_requirements.fit_transform(requirements_texts)
        else:
            requirements_vectors = np.zeros((len(requirements_texts), 200))
        
        logger.info(f"📝 Vectores de habilidades: {skills_vectors.shape}")
        logger.info(f"📝 Vectores de requisitos: {requirements_vectors.shape}")
        
        return skills_vectors, requirements_vectors
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codificar variables categóricas"""
        logger.info("🏷️ Codificando variables categóricas...")
        
        categorical_columns = [
            'candidate_nivel_educacion', 'offer_empresa_rubro_encoded',
            'offer_nivel_requisitos', 'offer_modalidad_trabajo'
        ]
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Manejar valores nulos y convertir todo a string
                df_encoded[col] = df_encoded[col].fillna('unknown').astype(str)
                
                # Ajustar y transformar
                if not hasattr(self.label_encoders[col], 'classes_'):
                    self.label_encoders[col].fit(df_encoded[col])
                
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        # Convertir columnas booleanas a numéricas
        boolean_columns = df_encoded.select_dtypes(include=['bool']).columns
        for col in boolean_columns:
            df_encoded[col] = df_encoded[col].astype(int)
        
        # Convertir columnas object restantes que deberían ser numéricas
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object' and col not in ['application_id', 'candidate_id', 'offer_id', 'label_quality']:
                try:
                    # Intentar convertir valores como 'indefinido' a NaN y luego a 0
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)
                except:
                    logger.warning(f"⚠️ No se pudo convertir columna {col} a numérica")
        
        return df_encoded
    
    def split_labeled_unlabeled_data(self, X: np.ndarray, y: np.ndarray, 
                                   is_labeled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separar datos etiquetados y no etiquetados"""
        labeled_mask = (is_labeled == True) & (y != -1)
        unlabeled_mask = ~labeled_mask
        
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]
        X_unlabeled = X[unlabeled_mask]
        y_unlabeled = y[unlabeled_mask]  # Será -1 para datos no etiquetados
        
        logger.info(f"📊 Datos etiquetados: {len(X_labeled)}")
        logger.info(f"📊 Datos no etiquetados: {len(X_unlabeled)}")
        logger.info(f"📊 Ratio etiquetado: {len(X_labeled)/(len(X_labeled)+len(X_unlabeled))*100:.1f}%")
        
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled
    
    def fit_transform(self, save_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Ajustar preprocessor y transformar datos"""
        logger.info("🔧 Iniciando ajuste y transformación del preprocessor...")
        
        # Conectar a MongoDB
        self.connect_mongodb()
        
        # Cargar datos
        applications_df, candidates_df, offers_df = self.load_data_from_mongodb()
        
        # Crear features comprehensivas
        features_df = self.create_comprehensive_features(applications_df, candidates_df, offers_df)
        
        # Procesar features de texto
        skills_vectors, req_vectors = self.process_text_features(applications_df, candidates_df, offers_df)
        
        # Codificar variables categóricas
        features_df = self.encode_categorical_features(features_df)
        
        # Separar features numéricas de metadatos
        metadata_columns = ['application_id', 'candidate_id', 'offer_id', 'target', 'is_labeled', 'label_quality']
        numeric_columns = [col for col in features_df.columns if col not in metadata_columns]
        
        # Extraer metadatos
        y = features_df['target'].values
        is_labeled = features_df['is_labeled'].values
        
        # Preparar features numéricas
        X_numeric = features_df[numeric_columns].copy()
        
        # Asegurar que todas las columnas numéricas son realmente numéricas
        for col in X_numeric.columns:
            X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce').fillna(0)
        
        logger.info(f"📊 Features numéricas a procesar: {X_numeric.shape}")
        logger.info(f"📊 Tipos de datos: {X_numeric.dtypes.value_counts().to_dict()}")
        
        # Imputar valores faltantes
        X_numeric_imputed = self.numeric_imputer.fit_transform(X_numeric)
        
        # Escalar features numéricas
        X_numeric_scaled = self.standard_scaler.fit_transform(X_numeric_imputed)
        
        # Combinar features numéricas con vectores de texto
        if hasattr(skills_vectors, 'toarray'):
            skills_array = skills_vectors.toarray()
        else:
            skills_array = skills_vectors
        
        if hasattr(req_vectors, 'toarray'):
            req_array = req_vectors.toarray()
        else:
            req_array = req_vectors
        
        # Concatenar todas las features
        X_combined = np.hstack([X_numeric_scaled, skills_array, req_array])
        
        # Aplicar PCA para reducir dimensionalidad
        X_final = self.pca.fit_transform(X_combined)
        
        # Separar datos etiquetados y no etiquetados
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = self.split_labeled_unlabeled_data(
            X_final, y, is_labeled
        )
        
        # Marcar como ajustado
        self.is_fitted = True
        
        # Guardar feature names
        self.feature_names = (numeric_columns + 
                            [f'skill_tfidf_{i}' for i in range(skills_array.shape[1])] +
                            [f'req_tfidf_{i}' for i in range(req_array.shape[1])])
        
        # Crear resumen de procesamiento
        processing_summary = {
            'total_samples': len(X_final),
            'labeled_samples': len(X_labeled),
            'unlabeled_samples': len(X_unlabeled),
            'n_features_original': X_combined.shape[1],
            'n_features_pca': X_final.shape[1],
            'pca_explained_variance_ratio': self.pca.explained_variance_ratio_.sum(),
            'class_distribution': {
                'positive': int(np.sum(y_labeled == 1)),
                'negative': int(np.sum(y_labeled == 0))
            },
            'feature_categories': {
                'numeric': len(numeric_columns),
                'text_skills': skills_array.shape[1],
                'text_requirements': req_array.shape[1]
            }
        }
        
        # Guardar preprocessor si se especifica ruta
        if save_path:
            self.save_preprocessor(save_path)
        
        logger.info("✅ Preprocessor ajustado y datos transformados")
        logger.info(f"📊 Resumen:")
        logger.info(f"  📈 Total muestras: {processing_summary['total_samples']}")
        logger.info(f"  🏷️ Etiquetadas: {processing_summary['labeled_samples']}")
        logger.info(f"  ❓ No etiquetadas: {processing_summary['unlabeled_samples']}")
        logger.info(f"  🔧 Features finales: {processing_summary['n_features_pca']}")
        logger.info(f"  📊 Varianza explicada: {processing_summary['pca_explained_variance_ratio']:.3f}")
        
        return X_labeled, y_labeled, X_unlabeled, processing_summary
    
    def transform(self, applications_df: pd.DataFrame, 
                 candidates_df: pd.DataFrame, 
                 offers_df: pd.DataFrame) -> np.ndarray:
        """Transformar nuevos datos usando preprocessor ajustado"""
        if not self.is_fitted:
            raise ValueError("Preprocessor no ha sido ajustado. Ejecutar fit_transform primero.")
        
        logger.info("🔄 Transformando nuevos datos...")
        
        # Crear features
        features_df = self.create_comprehensive_features(applications_df, candidates_df, offers_df)
        
        # Procesar texto
        skills_vectors, req_vectors = self.process_text_features(applications_df, candidates_df, offers_df)
        
        # Codificar categóricas
        features_df = self.encode_categorical_features(features_df)
        
        # Preparar features
        metadata_columns = ['application_id', 'candidate_id', 'offer_id', 'target', 'is_labeled', 'label_quality']
        numeric_columns = [col for col in features_df.columns if col not in metadata_columns]
        
        X_numeric = features_df[numeric_columns]
        X_numeric_imputed = self.numeric_imputer.transform(X_numeric)
        X_numeric_scaled = self.standard_scaler.transform(X_numeric_imputed)
        
        # Vectores de texto
        skills_array = skills_vectors.toarray() if hasattr(skills_vectors, 'toarray') else skills_vectors
        req_array = req_vectors.toarray() if hasattr(req_vectors, 'toarray') else req_vectors
        
        # Combinar y aplicar PCA
        X_combined = np.hstack([X_numeric_scaled, skills_array, req_array])
        X_final = self.pca.transform(X_combined)
        
        return X_final
    
    def save_preprocessor(self, filepath: str):
        """Guardar preprocessor completo"""
        logger.info(f"💾 Guardando preprocessor en {filepath}")
        
        preprocessor_data = {
            'tfidf_skills': self.tfidf_skills,
            'tfidf_requirements': self.tfidf_requirements,
            'standard_scaler': self.standard_scaler,
            'minmax_scaler': self.minmax_scaler,
            'numeric_imputer': self.numeric_imputer,
            'categorical_imputer': self.categorical_imputer,
            'label_encoders': self.label_encoders,
            'pca': self.pca,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'preprocessing_config': self.preprocessing_config,
            'labeled_threshold': self.labeled_threshold,
            'min_labeled_samples': self.min_labeled_samples
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"✅ Preprocessor guardado exitosamente")
    
    def load_preprocessor(self, filepath: str):
        """Cargar preprocessor desde archivo"""
        logger.info(f"📁 Cargando preprocessor desde {filepath}")
        
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        # Restaurar atributos
        for key, value in preprocessor_data.items():
            setattr(self, key, value)
        
        logger.info(f"✅ Preprocessor cargado exitosamente")
    
    def get_feature_importance_from_model(self, model, top_n: int = 20) -> Dict[str, float]:
        """Obtener importancia de features desde un modelo entrenado"""
        if not self.is_fitted:
            return {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return {}
            
            # Crear diccionario con nombres de features
            feature_importance = {}
            n_features = min(len(importances), len(self.feature_names))
            
            for i in range(n_features):
                if i < len(self.feature_names):
                    feature_importance[self.feature_names[i]] = float(importances[i])
                else:
                    feature_importance[f'pca_component_{i}'] = float(importances[i])
            
            # Ordenar por importancia
            sorted_features = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:top_n])
            
            self.feature_importance = sorted_features
            return sorted_features
            
        except Exception as e:
            logger.error(f"❌ Error calculando importancia de features: {e}")
            return {}


# Instancia global del preprocessor
semi_supervised_preprocessor = SemiSupervisedPreprocessor()


def preprocess_for_semi_supervised_learning(save_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Función conveniente para preprocessar datos para aprendizaje semi-supervisado"""
    return semi_supervised_preprocessor.fit_transform(save_path)


if __name__ == "__main__":
    # Test del preprocessor
    try:
        X_labeled, y_labeled, X_unlabeled, summary = preprocess_for_semi_supervised_learning(
            save_path="trained_models/semi_supervised_preprocessor.pkl"
        )
        
        print("🎉 Preprocessor semi-supervisado completado")
        print(f"📊 Datos etiquetados: {X_labeled.shape}")
        print(f"📊 Datos no etiquetados: {X_unlabeled.shape}")
        
    except Exception as e:
        logger.error(f"❌ Error en test del preprocessor: {e}")