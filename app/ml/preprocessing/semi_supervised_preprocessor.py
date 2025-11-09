import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import logging
import re
import pickle
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SemiSupervisedPreprocessor:
    """Preprocesador para modelo semi-supervisado de predicción de estados de postulaciones"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.standard_scaler = StandardScaler()
        self.tfidf_vectorizers = {}
        self.one_hot_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        self.estado_mapping = {}
        
        # Configuración de características de texto
        self.text_features = [
            'habilidades', 'idiomas', 'certificaciones', 'puesto_actual',
            'oferta_descripcion', 'oferta_requisitos', 'empresa_rubro'
        ]
        
        # Configuración de características categóricas
        self.categorical_features = [
            'nivel_educacion', 'oferta_ubicacion', 'empresa_nombre'
        ]
        
        # Configuración de características numéricas
        self.numerical_features = [
            'anios_experiencia', 'oferta_salario', 'total_entrevistas',
            'promedio_duracion_entrevistas', 'promedio_calificacion_tecnica',
            'promedio_calificacion_actitud', 'promedio_calificacion_general'
        ]
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características adicionales para el modelo"""
        df = df.copy()
        
        try:
            # Características de fecha
            if 'fecha_postulacion' in df.columns:
                df['fecha_postulacion'] = pd.to_datetime(df['fecha_postulacion'], errors='coerce')
                df['dias_desde_postulacion'] = (datetime.now() - df['fecha_postulacion']).dt.days
                df['mes_postulacion'] = df['fecha_postulacion'].dt.month
                df['dia_semana_postulacion'] = df['fecha_postulacion'].dt.dayofweek
            
            # Características de texto - longitud y conteos
            for feature in self.text_features:
                if feature in df.columns:
                    df[f'{feature}_length'] = df[feature].astype(str).str.len()
                    df[f'{feature}_word_count'] = df[feature].astype(str).str.split().str.len()
                    
                    # Palabras clave específicas
                    if feature == 'habilidades':
                        df['habilidades_tech_keywords'] = df[feature].astype(str).str.lower().str.count(
                            '|'.join(['python', 'java', 'javascript', 'sql', 'react', 'angular', 'node'])
                        )
                    elif feature == 'idiomas':
                        df['idiomas_count'] = df[feature].astype(str).str.count(',') + 1
                        df['tiene_ingles'] = df[feature].astype(str).str.lower().str.contains('ingles|english').astype(int)
            
            # Características de salario
            if 'oferta_salario' in df.columns:
                df['salario_por_experiencia'] = df['oferta_salario'] / (df['anios_experiencia'] + 1)
                df['salario_alto'] = (df['oferta_salario'] > df['oferta_salario'].quantile(0.75)).astype(int)
                df['salario_bajo'] = (df['oferta_salario'] < df['oferta_salario'].quantile(0.25)).astype(int)
            
            # Características de experiencia
            if 'anios_experiencia' in df.columns:
                df['experiencia_categoria'] = pd.cut(
                    df['anios_experiencia'], 
                    bins=[0, 2, 5, 10, float('inf')],
                    labels=['junior', 'semi_senior', 'senior', 'expert']
                ).astype(str)
            
            # Características de entrevistas
            if 'total_entrevistas' in df.columns:
                df['tiene_entrevistas'] = (df['total_entrevistas'] > 0).astype(int)
                df['multiples_entrevistas'] = (df['total_entrevistas'] > 1).astype(int)
            
            # Características de evaluaciones
            eval_cols = ['promedio_calificacion_tecnica', 'promedio_calificacion_actitud', 'promedio_calificacion_general']
            for col in eval_cols:
                if col in df.columns:
                    df[f'{col}_alta'] = (df[col] > 3.5).astype(int)
                    df[f'{col}_baja'] = (df[col] < 2.5).astype(int)
            
            # Características combinadas
            if all(col in df.columns for col in eval_cols):
                df['promedio_evaluaciones_total'] = df[eval_cols].mean(axis=1)
                df['evaluacion_excelente'] = (df['promedio_evaluaciones_total'] > 4.0).astype(int)
                df['evaluacion_deficiente'] = (df['promedio_evaluaciones_total'] < 2.0).astype(int)
            
            logger.info(f"Características creadas. Shape final: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error creando características: {str(e)}")
            return df
    
    def preprocess_text_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Procesa las características de texto usando TF-IDF"""
        try:
            text_features_processed = []
            
            for feature in self.text_features:
                if feature in df.columns:
                    # Limpiar texto
                    text_data = df[feature].astype(str).fillna('')
                    text_data = text_data.apply(self._clean_text)
                    
                    if fit:
                        # Crear y ajustar vectorizador
                        vectorizer = TfidfVectorizer(
                            max_features=100,  # Reducido para semi-supervisado
                            stop_words='english',
                            ngram_range=(1, 2),
                            min_df=2,
                            max_df=0.8
                        )
                        tfidf_matrix = vectorizer.fit_transform(text_data)
                        self.tfidf_vectorizers[feature] = vectorizer
                    else:
                        if feature in self.tfidf_vectorizers:
                            tfidf_matrix = self.tfidf_vectorizers[feature].transform(text_data)
                        else:
                            continue
                    
                    text_features_processed.append(tfidf_matrix.toarray())
            
            if text_features_processed:
                return np.hstack(text_features_processed)
            else:
                return np.array([]).reshape(len(df), 0)
                
        except Exception as e:
            logger.error(f"Error procesando características de texto: {str(e)}")
            return np.array([]).reshape(len(df), 0)
    
    def preprocess_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Procesa las características categóricas usando One-Hot Encoding"""
        try:
            categorical_features_processed = []
            
            for feature in self.categorical_features:
                if feature in df.columns:
                    # Rellenar valores nulos
                    data = df[feature].fillna('unknown').astype(str)
                    
                    if fit:
                        # Crear y ajustar encoder
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded_data = encoder.fit_transform(data.values.reshape(-1, 1))
                        self.one_hot_encoders[feature] = encoder
                    else:
                        if feature in self.one_hot_encoders:
                            encoded_data = self.one_hot_encoders[feature].transform(data.values.reshape(-1, 1))
                        else:
                            continue
                    
                    categorical_features_processed.append(encoded_data)
            
            if categorical_features_processed:
                return np.hstack(categorical_features_processed)
            else:
                return np.array([]).reshape(len(df), 0)
                
        except Exception as e:
            logger.error(f"Error procesando características categóricas: {str(e)}")
            return np.array([]).reshape(len(df), 0)
    
    def preprocess_numerical_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Procesa las características numéricas"""
        try:
            # Seleccionar características numéricas disponibles
            available_numerical = [col for col in self.numerical_features if col in df.columns]
            
            if not available_numerical:
                return np.array([]).reshape(len(df), 0)
            
            # Crear características adicionales numéricas
            additional_numerical = []
            for col in df.columns:
                if col.endswith('_length') or col.endswith('_word_count') or col.endswith('_count'):
                    additional_numerical.append(col)
                elif col in ['dias_desde_postulacion', 'mes_postulacion', 'dia_semana_postulacion']:
                    additional_numerical.append(col)
                elif col.endswith('_alta') or col.endswith('_baja') or col.endswith('_categoria'):
                    if df[col].dtype in ['int64', 'float64']:
                        additional_numerical.append(col)
            
            all_numerical = available_numerical + additional_numerical
            numerical_data = df[all_numerical].fillna(0)
            
            if fit:
                scaled_data = self.standard_scaler.fit_transform(numerical_data)
            else:
                scaled_data = self.standard_scaler.transform(numerical_data)
            
            return scaled_data
            
        except Exception as e:
            logger.error(f"Error procesando características numéricas: {str(e)}")
            return np.array([]).reshape(len(df), 0)
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = 'estado') -> Tuple[np.ndarray, np.ndarray]:
        """Ajusta el preprocesador y transforma los datos"""
        try:
            logger.info("Iniciando ajuste y transformación de datos")
            
            # Crear características adicionales
            df_processed = self.create_features(df)
            
            # Separar datos etiquetados y no etiquetados
            labeled_mask = df_processed[target_column].notna() & (df_processed[target_column] != '')
            labeled_df = df_processed[labeled_mask]
            unlabeled_df = df_processed[~labeled_mask]
            
            logger.info(f"Datos etiquetados: {len(labeled_df)}, No etiquetados: {len(unlabeled_df)}")
            
            # Procesar etiquetas
            if len(labeled_df) > 0:
                y_labeled = self.label_encoder.fit_transform(labeled_df[target_column])
                self.estado_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
                logger.info(f"Estados encontrados: {self.estado_mapping}")
            else:
                y_labeled = np.array([])
            
            # Procesar características
            text_features = self.preprocess_text_features(df_processed, fit=True)
            categorical_features = self.preprocess_categorical_features(df_processed, fit=True)
            numerical_features = self.preprocess_numerical_features(df_processed, fit=True)
            
            # Combinar todas las características
            features_list = []
            if text_features.shape[1] > 0:
                features_list.append(text_features)
            if categorical_features.shape[1] > 0:
                features_list.append(categorical_features)
            if numerical_features.shape[1] > 0:
                features_list.append(numerical_features)
            
            if features_list:
                X = np.hstack(features_list)
            else:
                X = np.array([]).reshape(len(df_processed), 0)
            
            # Separar características etiquetadas y no etiquetadas
            X_labeled = X[labeled_mask] if len(labeled_df) > 0 else np.array([]).reshape(0, X.shape[1])
            X_unlabeled = X[~labeled_mask] if len(unlabeled_df) > 0 else np.array([]).reshape(0, X.shape[1])
            
            self.is_fitted = True
            
            logger.info(f"Preprocesamiento completado. X_labeled: {X_labeled.shape}, X_unlabeled: {X_unlabeled.shape}")
            
            return X_labeled, y_labeled, X_unlabeled
            
        except Exception as e:
            logger.error(f"Error en fit_transform: {str(e)}")
            raise e
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforma nuevos datos usando el preprocesador ajustado"""
        try:
            if not self.is_fitted:
                raise ValueError("El preprocesador no ha sido ajustado. Llama a fit_transform primero.")
            
            # Crear características adicionales
            df_processed = self.create_features(df)
            
            # Procesar características
            text_features = self.preprocess_text_features(df_processed, fit=False)
            categorical_features = self.preprocess_categorical_features(df_processed, fit=False)
            numerical_features = self.preprocess_numerical_features(df_processed, fit=False)
            
            # Combinar todas las características
            features_list = []
            if text_features.shape[1] > 0:
                features_list.append(text_features)
            if categorical_features.shape[1] > 0:
                features_list.append(categorical_features)
            if numerical_features.shape[1] > 0:
                features_list.append(numerical_features)
            
            if features_list:
                X = np.hstack(features_list)
            else:
                X = np.array([]).reshape(len(df_processed), 0)
            
            return X
            
        except Exception as e:
            logger.error(f"Error en transform: {str(e)}")
            raise e
    
    def _clean_text(self, text: str) -> str:
        """Limpia el texto para el procesamiento"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convertir a minúsculas
        text = str(text).lower()
        
        # Remover caracteres especiales
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remover espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save(self, filepath: str):
        """Guarda el preprocesador entrenado"""
        try:
            preprocessor_data = {
                'label_encoder': self.label_encoder,
                'standard_scaler': self.standard_scaler,
                'tfidf_vectorizers': self.tfidf_vectorizers,
                'one_hot_encoders': self.one_hot_encoders,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
                'estado_mapping': self.estado_mapping,
                'text_features': self.text_features,
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(preprocessor_data, f)
            
            logger.info(f"Preprocesador guardado en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando preprocesador: {str(e)}")
            raise e
    
    def load(self, filepath: str):
        """Carga un preprocesador guardado"""
        try:
            with open(filepath, 'rb') as f:
                preprocessor_data = pickle.load(f)
            
            self.label_encoder = preprocessor_data['label_encoder']
            self.standard_scaler = preprocessor_data['standard_scaler']
            self.tfidf_vectorizers = preprocessor_data['tfidf_vectorizers']
            self.one_hot_encoders = preprocessor_data['one_hot_encoders']
            self.feature_names = preprocessor_data['feature_names']
            self.is_fitted = preprocessor_data['is_fitted']
            self.estado_mapping = preprocessor_data['estado_mapping']
            self.text_features = preprocessor_data['text_features']
            self.categorical_features = preprocessor_data['categorical_features']
            self.numerical_features = preprocessor_data['numerical_features']
            
            logger.info(f"Preprocesador cargado desde: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando preprocesador: {str(e)}")
            raise e