"""
Extractor de datos desde MongoDB para entrenamiento de modelos ML
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import asyncio

from app.config.mongodb_connection import get_collection_sync
from app.config.settings import settings

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extrae y combina datos desde MongoDB para el entrenamiento"""
    
    def __init__(self):
        self.candidates_collection = None
        self.companies_collection = None
        self.job_offers_collection = None
        
    def _connect_collections(self):
        """Conecta a las colecciones de MongoDB"""
        try:
            self.candidates_collection = get_collection_sync("candidates_features")
            self.companies_collection = get_collection_sync("companies_features")
            self.job_offers_collection = get_collection_sync("job_offers_features")
            logger.info("Conectado a las colecciones de MongoDB")
        except Exception as e:
            logger.error(f"Error conectando a MongoDB: {e}")
            raise
    
    def extract_candidates(self) -> pd.DataFrame:
        """Extrae datos de candidatos desde MongoDB"""
        if self.candidates_collection is None:
            self._connect_collections()
            
        try:
            candidates = list(self.candidates_collection.find({}))
            logger.info(f"Extraídos {len(candidates)} candidatos")
            
            if not candidates:
                logger.warning("No se encontraron candidatos en la base de datos")
                return pd.DataFrame()
            
            df = pd.DataFrame(candidates)
            
            # Limpiar columnas de MongoDB
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Usar postulante_id como candidate_id si está disponible
            if 'postulante_id' in df.columns:
                df['candidate_id'] = df['postulante_id']
            
            # Renombrar columnas para consistencia
            column_mapping = {
                'anios_experiencia': 'years_experience',
                'nivel_educacion': 'education_level',
                'puesto_actual': 'current_position'
            }
            df = df.rename(columns=column_mapping)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extrayendo candidatos: {e}")
            return pd.DataFrame()
    
    def extract_companies(self) -> pd.DataFrame:
        """Extrae datos de empresas desde MongoDB"""
        if self.companies_collection is None:
            self._connect_collections()
            
        try:
            companies = list(self.companies_collection.find({}))
            logger.info(f"Extraídas {len(companies)} empresas")
            
            if not companies:
                logger.warning("No se encontraron empresas en la base de datos")
                return pd.DataFrame()
            
            df = pd.DataFrame(companies)
            
            # Limpiar columnas de MongoDB
            if '_id' in df.columns:
                df['company_id'] = df['_id'].astype(str)
                df = df.drop('_id', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extrayendo empresas: {e}")
            return pd.DataFrame()
    
    def extract_job_offers(self) -> pd.DataFrame:
        """Extrae datos de ofertas de trabajo desde MongoDB"""
        if self.job_offers_collection is None:
            self._connect_collections()
            
        try:
            offers = list(self.job_offers_collection.find({}))
            logger.info(f"Extraídas {len(offers)} ofertas de trabajo")
            
            if not offers:
                logger.warning("No se encontraron ofertas de trabajo en la base de datos")
                return pd.DataFrame()
            
            df = pd.DataFrame(offers)
            
            # Limpiar columnas de MongoDB
            if '_id' in df.columns:
                df['offer_id'] = df['_id'].astype(str)
                df = df.drop('_id', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extrayendo ofertas: {e}")
            return pd.DataFrame()
    
    def create_training_dataset(self, 
                              positive_samples_ratio: float = 0.3,
                              negative_samples_multiplier: int = 2) -> pd.DataFrame:
        """
        Crea un dataset de entrenamiento combinando candidatos y ofertas
        
        Args:
            positive_samples_ratio: Proporción de muestras positivas (contratados)
            negative_samples_multiplier: Multiplicador para muestras negativas
        """
        # Extraer datos
        candidates_df = self.extract_candidates()
        companies_df = self.extract_companies()
        offers_df = self.extract_job_offers()
        
        if candidates_df.empty or offers_df.empty:
            logger.error("No hay datos suficientes para crear dataset de entrenamiento")
            return pd.DataFrame()
        
        # Crear combinaciones candidato-oferta
        training_data = []
        
        # Samples positivos (contratados/llamados)
        num_positive = int(len(candidates_df) * positive_samples_ratio)
        
        for i in range(num_positive):
            candidate = candidates_df.iloc[i % len(candidates_df)]
            offer = offers_df.iloc[i % len(offers_df)]
            
            # Combinar datos del candidato y la oferta
            combined_record = self._combine_candidate_offer(candidate, offer, target=1)
            training_data.append(combined_record)
        
        # Samples negativos (no contratados/no llamados)
        num_negative = num_positive * negative_samples_multiplier
        
        for i in range(num_negative):
            # Mezclar aleatoriamente candidatos y ofertas para crear pares negativos
            candidate_idx = np.random.randint(0, len(candidates_df))
            offer_idx = np.random.randint(0, len(offers_df))
            
            candidate = candidates_df.iloc[candidate_idx]
            offer = offers_df.iloc[offer_idx]
            
            combined_record = self._combine_candidate_offer(candidate, offer, target=0)
            training_data.append(combined_record)
        
        # Crear DataFrame final
        training_df = pd.DataFrame(training_data)
        
        # Mezclar datos
        training_df = training_df.sample(frac=1).reset_index(drop=True)
        
        logger.info(f"Dataset de entrenamiento creado: {len(training_df)} registros")
        logger.info(f"Distribución del target: {training_df['target'].value_counts().to_dict()}")
        
        return training_df
    
    def _combine_candidate_offer(self, candidate: pd.Series, offer: pd.Series, target: int) -> Dict:
        """Combina datos de candidato y oferta en un registro único"""
        
        # Obtener candidate_id desde postulante_id o _id
        candidate_id = candidate.get('postulante_id', candidate.get('candidate_id', candidate.get('_id', '')))
        
        record = {
            # Información del candidato
            'candidate_id': candidate_id,
            'years_experience': candidate.get('years_experience', candidate.get('anios_experiencia', 0)),
            'education_level': candidate.get('education_level', candidate.get('nivel_educacion', '')),
            'skills': candidate.get('habilidades', ''),
            'languages': candidate.get('idiomas', ''),
            'certifications': candidate.get('certificaciones', ''),
            'current_position': candidate.get('current_position', candidate.get('puesto_actual', '')),
            
            # Información de la oferta
            'offer_id': offer.get('offer_id', offer.get('oferta_id', offer.get('_id', ''))),
            'job_title': offer.get('titulo', ''),
            'salary': offer.get('salario', 0),
            'location': offer.get('ubicacion', ''),
            'requirements': offer.get('requisitos', ''),
            'company_id': offer.get('empresa_id', ''),
            
            # Target variable
            'target': target,
            
            # Timestamps
            'created_at': datetime.now().isoformat()
        }
        
        return record
    
    def get_candidate_offer_pair(self, candidate_id: str, offer_id: str) -> Optional[Dict]:
        """Obtiene un par específico candidato-oferta para predicción"""
        if self.candidates_collection is None:
            self._connect_collections()
        
        try:
            # Buscar candidato por postulante_id primero, luego por _id
            candidate = self.candidates_collection.find_one({"postulante_id": candidate_id})
            if not candidate:
                candidate = self.candidates_collection.find_one({"_id": candidate_id})
            
            # Buscar oferta
            offer = self.job_offers_collection.find_one({"_id": offer_id})
            if not offer:
                offer = self.job_offers_collection.find_one({"oferta_id": offer_id})
            
            if not candidate or not offer:
                logger.warning(f"No se encontró candidato {candidate_id} u oferta {offer_id}")
                return None
            
            # Convertir a Series para usar el método existente
            candidate_series = pd.Series(candidate)
            offer_series = pd.Series(offer)
            
            # Combinar sin target (para predicción)
            record = self._combine_candidate_offer(candidate_series, offer_series, target=-1)
            del record['target']  # Remover target para predicción
            
            return record
            
        except Exception as e:
            logger.error(f"Error obteniendo par candidato-oferta: {e}")
            return None
    
    def get_all_candidates_for_offer(self, offer_id: str) -> List[Dict]:
        """Obtiene todos los candidatos para una oferta específica"""
        if self.candidates_collection is None:
            self._connect_collections()
        
        try:
            # Buscar oferta
            offer = self.job_offers_collection.find_one({"_id": offer_id})
            if not offer:
                offer = self.job_offers_collection.find_one({"oferta_id": offer_id})
            
            if not offer:
                logger.warning(f"No se encontró oferta {offer_id}")
                return []
            
            # Obtener todos los candidatos
            candidates = list(self.candidates_collection.find({}))
            
            results = []
            offer_series = pd.Series(offer)
            
            for candidate in candidates:
                candidate_series = pd.Series(candidate)
                record = self._combine_candidate_offer(candidate_series, offer_series, target=-1)
                del record['target']  # Remover target para predicción
                results.append(record)
            
            logger.info(f"Obtenidos {len(results)} candidatos para oferta {offer_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error obteniendo candidatos para oferta: {e}")
            return []


# Instancia global
data_extractor = DataExtractor()


def get_training_data() -> pd.DataFrame:
    """Función conveniente para obtener datos de entrenamiento"""
    return data_extractor.create_training_dataset()


def get_prediction_data(candidate_id: str, offer_id: str) -> Optional[Dict]:
    """Función conveniente para obtener datos de predicción"""
    return data_extractor.get_candidate_offer_pair(candidate_id, offer_id)


def get_candidates_for_offer(offer_id: str) -> List[Dict]:
    """Función conveniente para obtener candidatos para una oferta"""
    return data_extractor.get_all_candidates_for_offer(offer_id)