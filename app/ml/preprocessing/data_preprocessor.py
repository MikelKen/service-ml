"""
Módulo de preprocessado de datos para predicción de contratación
"""
import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata
import re
from typing import Dict, List, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clase para preprocessar datos de postulaciones y ofertas de trabajo"""
    
    def __init__(self):
        self.target_mapping = {
            'aceptado': 1,
            'entrevista': 1,
            'contratado': 1,
            'rechazado': 0,
            'en revisión': 0,
            'pendiente': 0,
            'descartado': 0
        }
    
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
    
    def calculate_days_difference(self, date1: pd.Timestamp, date2: pd.Timestamp) -> int:
        """Calcula diferencia en días entre dos fechas"""
        if pd.isna(date1) or pd.isna(date2):
            return 0
        return (date1 - date2).days
    
    def load_and_preprocess(self, csv_path: str) -> pd.DataFrame:
        """Carga y preprocessa el dataset principal"""
        logger.info(f"Cargando datos desde: {csv_path}")
        
        # Cargar datos
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Crear copia para procesamiento
        df_processed = df.copy()
        
        # 1. Limpiar nombres de columnas
        df_processed.columns = df_processed.columns.str.lower().str.strip()
        
        # 2. Convertir columnas numéricas
        if 'años_experiencia' in df_processed.columns:
            df_processed['años_experiencia'] = pd.to_numeric(
                df_processed['años_experiencia'], errors='coerce'
            ).fillna(0)
        
        if 'salario' in df_processed.columns:
            df_processed['salario'] = pd.to_numeric(
                df_processed['salario'], errors='coerce'
            ).fillna(0)
        
        # 3. Normalizar campos de texto
        text_columns = [
            'nombre', 'nivel_educacion', 'habilidades', 'idiomas', 
            'certificaciones', 'puesto_actual', 'industria', 'titulo',
            'descripcion', 'ubicacion', 'requisitos'
        ]
        
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].apply(self.normalize_text)
        
        # 4. Procesar fechas
        date_columns = ['fecha_postulacion', 'fecha_publicacion']
        for col in date_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(
                    df_processed[col], errors='coerce'
                )
        
        # 5. Crear features temporales
        if 'fecha_postulacion' in df_processed.columns and 'fecha_publicacion' in df_processed.columns:
            df_processed['dias_desde_publicacion'] = df_processed.apply(
                lambda row: self.calculate_days_difference(
                    row['fecha_postulacion'], row['fecha_publicacion']
                ), axis=1
            )
        
        # Agregar feature de mes y día de la semana
        if 'fecha_postulacion' in df_processed.columns:
            df_processed['mes_postulacion'] = df_processed['fecha_postulacion'].dt.month
            df_processed['dia_semana_postulacion'] = df_processed['fecha_postulacion'].dt.dayofweek
        
        # 6. Crear variable objetivo
        df_processed['contactado'] = self.create_target_variable(df_processed)
        
        # 7. Crear features adicionales
        df_processed = self.create_additional_features(df_processed)
        
        logger.info(f"Preprocessamiento completado: {df_processed.shape[0]} filas, {df_processed.shape[1]} columnas")
        
        return df_processed
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """Crea la variable objetivo 'contactado' basada en el estado"""
        if 'estado' not in df.columns:
            logger.warning("Columna 'estado' no encontrada. Creando target aleatorio para demo.")
            return np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        
        target = df['estado'].apply(
            lambda x: self.target_mapping.get(self.normalize_text(str(x)), 0)
        )
        
        logger.info(f"Distribución del target: {target.value_counts().to_dict()}")
        return target
    
    def create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features adicionales para mejorar el modelo"""
        
        # Feature: coincidencia de habilidades
        if 'habilidades' in df.columns and 'requisitos' in df.columns:
            df['coincidencia_habilidades'] = df.apply(
                lambda row: self.calculate_skill_overlap(
                    row['habilidades'], row['requisitos']
                ), axis=1
            )
        
        # Feature: experiencia vs salario (ratio)
        if 'años_experiencia' in df.columns and 'salario' in df.columns:
            df['salario_por_año_exp'] = df.apply(
                lambda row: row['salario'] / max(row['años_experiencia'], 1), axis=1
            )
        
        # Feature: número de habilidades
        if 'habilidades' in df.columns:
            df['num_habilidades'] = df['habilidades'].apply(
                lambda x: len(self.parse_skills_list(str(x)))
            )
        
        # Feature: número de idiomas
        if 'idiomas' in df.columns:
            df['num_idiomas'] = df['idiomas'].apply(
                lambda x: len(self.parse_skills_list(str(x)))
            )
        
        # Feature: tiene certificaciones
        if 'certificaciones' in df.columns:
            df['tiene_certificaciones'] = df['certificaciones'].apply(
                lambda x: 1 if (not pd.isna(x) and str(x).lower() not in ['', 'sin certificacion', 'ninguna']) else 0
            )
        
        # Feature: coincidencia de industria
        if 'industria' in df.columns:
            # Esto se puede expandir con lógica más sofisticada
            df['industria_encoded'] = pd.Categorical(df['industria']).codes
        
        # Feature: coincidencia de ubicación (simplificado)
        if 'ubicacion' in df.columns:
            df['ubicacion_encoded'] = pd.Categorical(df['ubicacion']).codes
        
        return df
    
    def calculate_skill_overlap(self, applicant_skills: str, job_requirements: str) -> float:
        """Calcula el overlap entre habilidades del postulante y requisitos del trabajo"""
        if pd.isna(applicant_skills) or pd.isna(job_requirements):
            return 0.0
        
        skills_list = set(self.parse_skills_list(str(applicant_skills)))
        requirements_list = set(self.parse_skills_list(str(job_requirements)))
        
        if not skills_list or not requirements_list:
            return 0.0
        
        intersection = skills_list.intersection(requirements_list)
        return len(intersection) / len(requirements_list) if requirements_list else 0.0
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Retorna resumen de features del dataset"""
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'target_distribution': df['contactado'].value_counts().to_dict() if 'contactado' in df.columns else {},
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'date_features': df.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        return summary


def preprocess_data(csv_path: str) -> Tuple[pd.DataFrame, Dict]:
    """Función principal para preprocessar datos"""
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.load_and_preprocess(csv_path)
    summary = preprocessor.get_feature_summary(df_processed)
    
    return df_processed, summary


if __name__ == "__main__":
    # Test del preprocessador
    csv_path = "../../postulaciones_sinteticas_500.csv"
    df, summary = preprocess_data(csv_path)
    
    print("Resumen del preprocessamiento:")
    print(f"Filas: {summary['total_rows']}")
    print(f"Columnas: {summary['total_columns']}")
    print(f"Distribución del target: {summary['target_distribution']}")