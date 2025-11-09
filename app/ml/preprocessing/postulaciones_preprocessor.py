"""
Preprocesador específico para datos de postulaciones extraídos de PostgreSQL
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import re
from datetime import datetime
import joblib
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class PostulacionesPreprocessor:
    """Preprocesador específico para datos de postulaciones"""
    
    def __init__(self):
        # Inicializar transformadores
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        self.column_transformer = None
        self.feature_names = []
        self.is_fitted = False
        
        # Configuración de preprocesamiento
        self.text_columns = ['habilidades', 'idiomas', 'certificaciones', 'requisitos', 'oferta_descripcion']
        self.categorical_columns = ['nivel_educacion', 'estado', 'empresa_rubro', 'ubicacion']
        self.numerical_columns = ['anios_experiencia', 'salario', 'total_visualizaciones', 
                                 'num_evaluaciones', 'promedio_tecnica', 'promedio_actitud', 
                                 'promedio_general']
        
        # Mapeo de estados para semi-supervisado (basado en datos reales)
        self.estado_mapping = {
            'Enviada': 0,                    # Inicial
            'En Revisión': 1,               # En proceso
            'En Proceso': 2,                # En proceso avanzado
            'Preseleccionada': 3,           # Preseleccionada
            'Evaluación Técnica': 4,        # Evaluación técnica
            'Entrevista Programada': 5,     # Entrevista programada
            'Evaluación Final': 6,          # Evaluación final
            'Oferta Enviada': 7,            # Oferta enviada
            'Aceptada': 8,                  # Exitosa
            'ACEPTADO': 8,                  # Exitosa (variante)
            'Rechazada': -1,                # Rechazada
            'RECHAZADO': -1                 # Rechazada (variante)
        }
        
        # Estados intermedios que serán predichos (sin etiquetar para semi-supervisado)
        self.unlabeled_states = ['En Revisión', 'En Proceso', 'Preseleccionada', 'Evaluación Técnica', 'Entrevista Programada']
        # Estados con resultado final conocido (etiquetados)
        self.labeled_states = ['Enviada', 'Evaluación Final', 'Oferta Enviada', 'Aceptada', 'ACEPTADO', 'Rechazada', 'RECHAZADO']
    
    def clean_text(self, text: str) -> str:
        """Limpia y normaliza texto"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convertir a string
        text = str(text).lower()
        
        # Remover caracteres especiales pero mantener espacios y comas
        text = re.sub(r'[^\w\s,.-]', ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae características de fechas"""
        df_copy = df.copy()
        
        # Procesar fecha_postulacion
        if 'fecha_postulacion' in df_copy.columns:
            df_copy['fecha_postulacion'] = pd.to_datetime(df_copy['fecha_postulacion'], errors='coerce')
            df_copy['postulacion_year'] = df_copy['fecha_postulacion'].dt.year
            df_copy['postulacion_month'] = df_copy['fecha_postulacion'].dt.month
            df_copy['postulacion_day_of_week'] = df_copy['fecha_postulacion'].dt.dayofweek
        
        # Procesar fecha_publicacion
        if 'fecha_publicacion' in df_copy.columns:
            df_copy['fecha_publicacion'] = pd.to_datetime(df_copy['fecha_publicacion'], errors='coerce')
            df_copy['publicacion_year'] = df_copy['fecha_publicacion'].dt.year
            df_copy['publicacion_month'] = df_copy['fecha_publicacion'].dt.month
            
            # Días entre publicación y postulación
            if 'fecha_postulacion' in df_copy.columns:
                df_copy['dias_desde_publicacion'] = (
                    df_copy['fecha_postulacion'] - df_copy['fecha_publicacion']
                ).dt.days
        
        return df_copy
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características adicionales de texto"""
        df_copy = df.copy()
        
        # Características de habilidades
        if 'habilidades' in df_copy.columns:
            df_copy['num_habilidades'] = df_copy['habilidades'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
        
        # Características de idiomas
        if 'idiomas' in df_copy.columns:
            df_copy['num_idiomas'] = df_copy['idiomas'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
        
        # Características de certificaciones
        if 'certificaciones' in df_copy.columns:
            df_copy['num_certificaciones'] = df_copy['certificaciones'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
        
        # Match entre habilidades y requisitos (características de compatibilidad)
        if 'habilidades' in df_copy.columns and 'requisitos' in df_copy.columns:
            df_copy['skill_match_score'] = df_copy.apply(
                lambda row: self._calculate_skill_match(
                    str(row['habilidades']), str(row['requisitos'])
                ), axis=1
            )
        
        return df_copy
    
    def _calculate_skill_match(self, skills: str, requirements: str) -> float:
        """Calcula score de match entre habilidades y requisitos"""
        if pd.isna(skills) or pd.isna(requirements):
            return 0.0
        
        skills_clean = self.clean_text(skills)
        requirements_clean = self.clean_text(requirements)
        
        skills_list = [s.strip() for s in skills_clean.split(',') if s.strip()]
        requirements_list = [r.strip() for r in requirements_clean.split(',') if r.strip()]
        
        if not skills_list or not requirements_list:
            return 0.0
        
        # Buscar coincidencias exactas y parciales
        matches = 0
        for skill in skills_list:
            for req in requirements_list:
                if skill in req or req in skill:
                    matches += 1
                    break
        
        return matches / max(len(skills_list), len(requirements_list))
    
    def prepare_for_semi_supervised(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepara datos para aprendizaje semi-supervisado
        Retorna: (datos_etiquetados, datos_sin_etiquetar, datos_completos)
        """
        df_copy = df.copy()
        
        # Separar datos etiquetados (estados conocidos) y sin etiquetar
        labeled_mask = df_copy['estado'].isin(self.labeled_states)
        unlabeled_mask = df_copy['estado'].isin(self.unlabeled_states) | df_copy['estado'].isna()
        
        labeled_data = df_copy[labeled_mask].copy()
        unlabeled_data = df_copy[unlabeled_mask].copy()
        
        logger.info(f"Datos etiquetados: {len(labeled_data)}")
        logger.info(f"Datos sin etiquetar: {len(unlabeled_data)}")
        logger.info(f"Distribución etiquetados: {labeled_data['estado'].value_counts().to_dict()}")
        
        return labeled_data, unlabeled_data, df_copy
    
    def encode_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica la variable objetivo (estado) usando el mapeo definido"""
        df_copy = df.copy()
        
        # Mapear estados a valores numéricos
        df_copy['estado_encoded'] = df_copy['estado'].map(self.estado_mapping)
        
        # Para valores sin mapear, asignar -999 (será usado para identificar sin etiquetar)
        df_copy['estado_encoded'] = df_copy['estado_encoded'].fillna(-999)
        
        # Crear variable binaria de éxito (contratado = 1, resto = 0)
        df_copy['success_target'] = (df_copy['estado'] == 'contratado').astype(int)
        
        return df_copy
    
    def preprocess_features(self, df: pd.DataFrame, fit_transformers: bool = False) -> pd.DataFrame:
        """Preprocesa todas las características"""
        logger.info("Iniciando preprocesamiento de características...")
        
        # 1. Extraer características de fechas
        df_processed = self.extract_date_features(df)
        
        # 2. Crear características de texto
        df_processed = self.create_text_features(df_processed)
        
        # 3. Limpiar texto
        for col in self.text_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].apply(self.clean_text)
        
        # 4. Codificar variable objetivo
        df_processed = self.encode_target_variable(df_processed)
        
        # 5. Manejar valores faltantes en columnas numéricas
        for col in self.numerical_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = df_processed[col].fillna(0)
        
        # 6. Crear características categóricas adicionales
        df_processed = self._create_categorical_features(df_processed)
        
        logger.info(f"Preprocesamiento completado: {df_processed.shape}")
        
        # 7. Crear matriz de características
        if fit_transformers:
            feature_matrix = self._fit_transform_features(df_processed)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Los transformadores no han sido entrenados. Use fit_transformers=True")
            feature_matrix = self._transform_features(df_processed)
        
        return feature_matrix
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características categóricas adicionales"""
        df_copy = df.copy()
        
        # Categorías de experiencia
        df_copy['experiencia_categoria'] = pd.cut(
            df_copy['anios_experiencia'], 
            bins=[-1, 0, 2, 5, 10, float('inf')],
            labels=['sin_experiencia', 'junior', 'intermedio', 'senior', 'experto']
        )
        
        # Categorías de salario
        if 'salario' in df_copy.columns:
            salario_quantiles = df_copy['salario'].quantile([0.25, 0.5, 0.75]).values
            df_copy['salario_categoria'] = pd.cut(
                df_copy['salario'],
                bins=[-1] + list(salario_quantiles) + [float('inf')],
                labels=['bajo', 'medio_bajo', 'medio_alto', 'alto']
            )
        
        # Indicadores binarios
        df_copy['tiene_cv'] = (~df_copy['url_cv'].isna()).astype(int)
        df_copy['tiene_puesto_actual'] = (~df_copy['puesto_actual'].isna()).astype(int)
        df_copy['tiene_entrevistas'] = (df_copy['num_evaluaciones'] > 0).astype(int)
        
        return df_copy
    
    def _fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta y transforma características (para entrenamiento)"""
        
        # Preparar columnas para transformación
        numerical_features = []
        categorical_features = []
        text_features = []
        
        # Identificar columnas disponibles
        for col in df.columns:
            if col in self.numerical_columns or col.endswith('_categoria') or 'num_' in col or 'promedio_' in col:
                if df[col].dtype in ['int64', 'float64']:
                    numerical_features.append(col)
            elif col in self.categorical_columns or col.endswith('_categoria'):
                categorical_features.append(col)
            elif col in self.text_columns:
                text_features.append(col)
        
        # Agregar características derivadas numéricas
        derived_numerical = ['postulacion_year', 'postulacion_month', 'postulacion_day_of_week',
                           'publicacion_year', 'publicacion_month', 'dias_desde_publicacion',
                           'num_habilidades', 'num_idiomas', 'num_certificaciones', 'skill_match_score',
                           'tiene_cv', 'tiene_puesto_actual', 'tiene_entrevistas']
        
        for col in derived_numerical:
            if col in df.columns and col not in numerical_features:
                numerical_features.append(col)
        
        logger.info(f"Características numéricas: {len(numerical_features)}")
        logger.info(f"Características categóricas: {len(categorical_features)}")
        logger.info(f"Características de texto: {len(text_features)}")
        
        # Crear transformadores
        transformers = []
        
        # Transformador para características numéricas
        if numerical_features:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_features))
        
        # Transformador para características categóricas
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Crear ColumnTransformer solo si hay transformadores
        if transformers:
            self.column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder='drop'
            )
            
            # Ajustar y transformar características numéricas/categóricas
            feature_array = self.column_transformer.fit_transform(df)
            
            # Obtener nombres de características
            feature_names = []
            for name, transformer, columns in transformers:
                if name == 'num':
                    feature_names.extend(columns)
                elif name == 'cat':
                    try:
                        if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                            cat_names = transformer.named_steps['onehot'].get_feature_names_out(columns)
                            feature_names.extend(cat_names)
                        else:
                            # Fallback para versiones anteriores
                            categories = transformer.named_steps['onehot'].categories_
                            for i, col in enumerate(columns):
                                for cat in categories[i]:
                                    feature_names.append(f"{col}_{cat}")
                    except Exception as e:
                        logger.warning(f"Error obteniendo nombres de características categóricas: {e}")
                        # Usar nombres genéricos
                        for i, col in enumerate(columns):
                            feature_names.extend([f"{col}_cat_{j}" for j in range(10)])  # Estimación
        else:
            feature_array = np.array([]).reshape(len(df), 0)
            feature_names = []
            self.column_transformer = None
        
        # Procesar características de texto
        text_features_arrays = []
        for col in text_features:
            if col in df.columns:
                # Configurar TF-IDF
                tfidf = TfidfVectorizer(
                    max_features=100,  # Limitado para eficiencia
                    stop_words=None,
                    ngram_range=(1, 2),
                    min_df=2
                )
                
                # Ajustar y transformar
                text_array = tfidf.fit_transform(df[col].fillna(''))
                text_features_arrays.append(text_array.toarray())
                
                # Guardar transformador
                self.tfidf_vectorizers[col] = tfidf
                
                # Agregar nombres de características
                if hasattr(tfidf, 'get_feature_names_out'):
                    text_feature_names = [f"{col}_{name}" for name in tfidf.get_feature_names_out()]
                else:
                    text_feature_names = [f"{col}_tfidf_{i}" for i in range(text_array.shape[1])]
                
                feature_names.extend(text_feature_names)
        
        # Combinar todas las características
        if text_features_arrays:
            text_combined = np.hstack(text_features_arrays)
            if feature_array.size > 0:
                feature_array = np.hstack([feature_array, text_combined])
            else:
                feature_array = text_combined
        
        # Asegurar que feature_array tenga la forma correcta
        if feature_array.size == 0:
            feature_array = np.zeros((len(df), 1))  # Al menos una característica dummy
            feature_names = ['dummy_feature']
        
        # Guardar nombres de características
        self.feature_names = feature_names
        
        # Crear DataFrame de características
        try:
            feature_df = pd.DataFrame(feature_array, columns=feature_names, index=df.index)
        except ValueError as e:
            logger.warning(f"Error creando DataFrame con nombres: {e}")
            # Crear nombres genéricos si hay mismatch
            generic_names = [f"feature_{i}" for i in range(feature_array.shape[1])]
            feature_df = pd.DataFrame(feature_array, columns=generic_names, index=df.index)
            self.feature_names = generic_names
        
        # Agregar columnas objetivo
        target_columns = ['estado', 'estado_encoded', 'success_target', 'postulacion_id']
        for col in target_columns:
            if col in df.columns:
                feature_df[col] = df[col].values
        
        logger.info(f"Matriz de características creada: {feature_df.shape}")
        
        return feature_df
    
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma características usando transformadores ajustados"""
        if not self.is_fitted:
            raise ValueError("Los transformadores no han sido ajustados")
        
        # Transformar características numéricas/categóricas
        if self.column_transformer is not None:
            feature_array = self.column_transformer.transform(df)
        else:
            feature_array = np.array([]).reshape(len(df), 0)
        
        # Transformar características de texto
        text_features_arrays = []
        for col, tfidf in self.tfidf_vectorizers.items():
            if col in df.columns:
                text_array = tfidf.transform(df[col].fillna(''))
                text_features_arrays.append(text_array.toarray())
        
        # Combinar características
        if text_features_arrays:
            text_combined = np.hstack(text_features_arrays)
            if feature_array.size > 0:
                feature_array = np.hstack([feature_array, text_combined])
            else:
                feature_array = text_combined
        
        # Crear DataFrame
        if len(self.feature_names) > 0:
            feature_df = pd.DataFrame(feature_array, columns=self.feature_names, index=df.index)
        else:
            # Si no hay nombres de características, crear nombres genéricos
            feature_df = pd.DataFrame(feature_array, index=df.index)
            feature_df.columns = [f"feature_{i}" for i in range(feature_array.shape[1])]
        
        # Agregar columnas objetivo
        target_columns = ['estado', 'estado_encoded', 'success_target', 'postulacion_id']
        for col in target_columns:
            if col in df.columns:
                feature_df[col] = df[col].values
        
        return feature_df
    
    def save_preprocessor(self, filepath: str):
        """Guarda el preprocesador entrenado"""
        if not self.is_fitted:
            raise ValueError("El preprocesador no ha sido entrenado")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'column_transformer': self.column_transformer,
            'tfidf_vectorizers': self.tfidf_vectorizers,
            'feature_names': self.feature_names,
            'estado_mapping': self.estado_mapping,
            'labeled_states': self.labeled_states,
            'unlabeled_states': self.unlabeled_states,
            'is_fitted': self.is_fitted,
            'version': '1.0.0',
            'created_at': datetime.now().isoformat()
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocesador guardado en: {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Carga un preprocesador guardado"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        preprocessor_data = joblib.load(filepath)
        
        self.column_transformer = preprocessor_data['column_transformer']
        self.tfidf_vectorizers = preprocessor_data['tfidf_vectorizers']
        self.feature_names = preprocessor_data['feature_names']
        self.estado_mapping = preprocessor_data.get('estado_mapping', self.estado_mapping)
        self.labeled_states = preprocessor_data.get('labeled_states', self.labeled_states)
        self.unlabeled_states = preprocessor_data.get('unlabeled_states', self.unlabeled_states)
        self.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocesador cargado desde: {filepath}")


# Instancia global
postulaciones_preprocessor = PostulacionesPreprocessor()