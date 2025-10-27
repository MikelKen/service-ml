"""
Módulo de Feature Engineering para predicción de contratación
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Any
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Clase para crear y transformar features para el modelo de ML"""
    
    def __init__(self):
        self.text_vectorizer = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.is_fitted = False
    
    def create_text_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Crea features de texto usando TF-IDF"""
        logger.info("Creando features de texto...")
        
        # Combinar campos de texto relevantes
        combined_texts = []
        
        # Para cada fila, combinar todos los textos relevantes
        for index, row in df.iterrows():
            text_parts = []
            
            # Descripción del trabajo + requisitos
            if 'descripcion' in df.columns:
                text_parts.append(str(row.get('descripcion', '')))
            if 'requisitos' in df.columns:
                text_parts.append(str(row.get('requisitos', '')))
            
            # Habilidades + certificaciones del postulante
            if 'habilidades' in df.columns:
                text_parts.append(str(row.get('habilidades', '')))
            if 'certificaciones' in df.columns:
                text_parts.append(str(row.get('certificaciones', '')))
            
            # Título del puesto
            if 'titulo' in df.columns:
                text_parts.append(str(row.get('titulo', '')))
            
            # Combinar todos los textos de esta fila
            combined_text = ' '.join([text for text in text_parts if text and text != 'nan'])
            combined_texts.append(combined_text if combined_text.strip() else 'empty')
        
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=200,  # Limitar features para evitar overfitting
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        # Transformar texto
        text_features = vectorizer.fit_transform(combined_texts)
        
        logger.info(f"Features de texto creadas: {text_features.shape[1]} features")
        
        return text_features.toarray(), vectorizer
    
    def create_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features numéricas"""
        logger.info("Creando features numéricas...")
        
        features_df = pd.DataFrame()
        
        # Features básicas numéricas
        numeric_cols = ['años_experiencia', 'salario', 'dias_desde_publicacion', 
                       'coincidencia_habilidades', 'num_habilidades', 'num_idiomas',
                       'salario_por_año_exp']
        
        for col in numeric_cols:
            if col in df.columns:
                features_df[col] = df[col].fillna(0)
        
        # Features temporales
        if 'mes_postulacion' in df.columns:
            features_df['mes_postulacion'] = df['mes_postulacion'].fillna(1)
        
        if 'dia_semana_postulacion' in df.columns:
            features_df['dia_semana_postulacion'] = df['dia_semana_postulacion'].fillna(0)
        
        # Features binarias
        binary_cols = ['tiene_certificaciones']
        for col in binary_cols:
            if col in df.columns:
                features_df[col] = df[col].fillna(0)
        
        # Features de ratio/interacciones
        if 'años_experiencia' in df.columns and 'num_habilidades' in df.columns:
            features_df['habilidades_por_año_exp'] = df.apply(
                lambda row: row['num_habilidades'] / max(row['años_experiencia'], 1) 
                if not pd.isna(row['num_habilidades']) and not pd.isna(row['años_experiencia']) else 0, 
                axis=1
            )
        
        # Feature de seniority level basado en experiencia
        if 'años_experiencia' in df.columns:
            features_df['nivel_seniority'] = df['años_experiencia'].apply(
                lambda x: 0 if x < 2 else (1 if x < 5 else (2 if x < 10 else 3))
            )
        
        # Feature de rango salarial
        if 'salario' in df.columns:
            features_df['rango_salarial'] = pd.cut(
                df['salario'].fillna(0), 
                bins=[0, 5000, 10000, 15000, np.inf], 
                labels=[0, 1, 2, 3]
            ).astype(int)
        
        logger.info(f"Features numéricas creadas: {features_df.shape[1]} features")
        
        return features_df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features categóricas usando encoding"""
        logger.info("Creando features categóricas...")
        
        features_df = pd.DataFrame()
        
        # Columnas categóricas para label encoding
        categorical_cols = ['nivel_educacion', 'puesto_actual', 'industria', 'ubicacion']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit encoder
                    values = df[col].fillna('unknown').astype(str)
                    self.label_encoders[col].fit(values)
                
                # Transform
                encoded_values = self.label_encoders[col].transform(
                    df[col].fillna('unknown').astype(str)
                )
                features_df[f'{col}_encoded'] = encoded_values
        
        # One-hot encoding para variables con pocas categorías
        if 'nivel_educacion' in df.columns:
            education_dummies = pd.get_dummies(
                df['nivel_educacion'].fillna('unknown'), 
                prefix='edu',
                dummy_na=False
            )
            features_df = pd.concat([features_df, education_dummies], axis=1)
        
        logger.info(f"Features categóricas creadas: {features_df.shape[1]} features")
        
        return features_df
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Ajusta transformadores y transforma datos de entrenamiento"""
        logger.info("Ajustando transformadores y creando features...")
        
        # Crear diferentes tipos de features
        numerical_features = self.create_numerical_features(df)
        categorical_features = self.create_categorical_features(df)
        text_features_array, self.text_vectorizer = self.create_text_features(df)
        
        # Combinar features numéricas y categóricas
        structured_features = pd.concat([numerical_features, categorical_features], axis=1)
        
        # Escalar features estructuradas
        self.scaler = StandardScaler()
        structured_features_scaled = self.scaler.fit_transform(structured_features)
        
        # Combinar con features de texto
        all_features = np.hstack([structured_features_scaled, text_features_array])
        
        # Guardar nombres de columnas para referencia
        self.feature_columns = (
            numerical_features.columns.tolist() + 
            categorical_features.columns.tolist() + 
            [f'text_feature_{i}' for i in range(text_features_array.shape[1])]
        )
        
        self.is_fitted = True
        
        logger.info(f"Features finales: {all_features.shape[1]} features para {all_features.shape[0]} muestras")
        
        return all_features
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforma nuevos datos usando transformadores ya ajustados"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer debe ser ajustado primero usando fit_transform()")
        
        logger.info("Transformando nuevos datos...")
        
        # Crear features usando transformadores ya ajustados
        numerical_features = self.create_numerical_features(df)
        categorical_features = self.create_categorical_features(df)
        
        # Para texto, usar el vectorizador ya ajustado con la misma lógica
        combined_texts = []
        
        # Para cada fila, combinar todos los textos relevantes
        for index, row in df.iterrows():
            text_parts = []
            
            # Descripción del trabajo + requisitos
            if 'descripcion' in df.columns:
                text_parts.append(str(row.get('descripcion', '')))
            if 'requisitos' in df.columns:
                text_parts.append(str(row.get('requisitos', '')))
            
            # Habilidades + certificaciones del postulante
            if 'habilidades' in df.columns:
                text_parts.append(str(row.get('habilidades', '')))
            if 'certificaciones' in df.columns:
                text_parts.append(str(row.get('certificaciones', '')))
            
            # Título del puesto
            if 'titulo' in df.columns:
                text_parts.append(str(row.get('titulo', '')))
            
            # Combinar todos los textos de esta fila
            combined_text = ' '.join([text for text in text_parts if text and text != 'nan'])
            combined_texts.append(combined_text if combined_text.strip() else 'empty')
        
        text_features_array = self.text_vectorizer.transform(combined_texts).toarray()
        
        # Combinar y escalar features estructuradas
        structured_features = pd.concat([numerical_features, categorical_features], axis=1)
        structured_features_scaled = self.scaler.transform(structured_features)
        
        # Combinar todas las features
        all_features = np.hstack([structured_features_scaled, text_features_array])
        
        logger.info(f"Transformación completada: {all_features.shape}")
        
        return all_features
    
    def save_transformers(self, filepath: str):
        """Guarda transformadores para uso futuro"""
        transformers = {
            'text_vectorizer': self.text_vectorizer,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(transformers, filepath)
        logger.info(f"Transformadores guardados en: {filepath}")
    
    def load_transformers(self, filepath: str):
        """Carga transformadores guardados"""
        transformers = joblib.load(filepath)
        
        self.text_vectorizer = transformers['text_vectorizer']
        self.scaler = transformers['scaler']
        self.label_encoders = transformers['label_encoders']
        self.feature_columns = transformers['feature_columns']
        self.is_fitted = transformers['is_fitted']
        
        logger.info(f"Transformadores cargados desde: {filepath}")
    
    def get_feature_importance_names(self) -> List[str]:
        """Retorna nombres de features para interpretabilidad"""
        return self.feature_columns if self.feature_columns else []


def create_features(df: pd.DataFrame, fit_transformers: bool = True) -> Tuple[np.ndarray, FeatureEngineer]:
    """Función principal para crear features"""
    engineer = FeatureEngineer()
    
    if fit_transformers:
        features = engineer.fit_transform(df)
    else:
        raise ValueError("Para nuevos datos, use el método transform del FeatureEngineer ya ajustado")
    
    return features, engineer


if __name__ == "__main__":
    # Test del feature engineer
    from ml.data.preprocessing import preprocess_data
    
    csv_path = "../../postulaciones_sinteticas_500.csv"
    df, _ = preprocess_data(csv_path)
    
    features, engineer = create_features(df)
    print(f"Features creadas: {features.shape}")
    print(f"Nombres de features: {len(engineer.get_feature_importance_names())}")