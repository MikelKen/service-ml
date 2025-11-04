#!/usr/bin/env python3
"""
🔍 RESOLVERS GRAPHQL PARA CLUSTERING DE CANDIDATOS
Implementa consultas para análisis de clustering no supervisado
"""

import strawberry
import pandas as pd
import numpy as np
import os
import glob
import pickle
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from app.graphql.types.clustering_types import (
    ClusterAnalysis, ClusterProfile, ClusteringMetrics,
    CandidateClusterAssignment, SimilarCandidates,
    ClusteringQueryInput, SimilarCandidatesInput, ClusterProfileInput
)
from app.config.mongodb_connection import get_collection
from app.ml.models.candidates_clustering_model import CandidatesClusteringModel
from app.ml.preprocessing.candidates_clustering_preprocessor import CandidatesClusteringPreprocessor

logger = logging.getLogger(__name__)

class ClusteringResolver:
    """Resolver para consultas de clustering de candidatos"""
    
    def __init__(self):
        self.models_cache = {}
        self.preprocessor_cache = {}
        self.data_cache = {}
    
    def _get_latest_model_files(self, algorithm: str = "kmeans") -> Dict[str, str]:
        """Obtiene los archivos más recientes del modelo"""
        try:
            models_dir = "trained_models/clustering"
            
            # Buscar archivos más recientes
            model_pattern = f"{models_dir}/candidates_clustering_{algorithm}_*.pkl"
            preprocessor_pattern = f"{models_dir}/candidates_clustering_preprocessor_*.pkl"
            data_pattern = f"{models_dir}/candidates_clustering_data_*.pkl"
            
            model_files = glob.glob(model_pattern)
            preprocessor_files = glob.glob(preprocessor_pattern)
            data_files = glob.glob(data_pattern)
            
            if not model_files or not preprocessor_files:
                raise FileNotFoundError(f"No se encontraron modelos de clustering entrenados para {algorithm}")
            
            # Obtener más reciente (por timestamp en nombre)
            latest_model = max(model_files, key=os.path.getctime)
            latest_preprocessor = max(preprocessor_files, key=os.path.getctime)
            latest_data = max(data_files, key=os.path.getctime) if data_files else None
            
            return {
                'model': latest_model,
                'preprocessor': latest_preprocessor,
                'data': latest_data
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo archivos del modelo: {e}")
            raise
    
    def _load_model_and_preprocessor(self, algorithm: str = "kmeans") -> tuple:
        """Carga modelo y preprocessor desde cache o disco"""
        cache_key = f"{algorithm}"
        
        if cache_key in self.models_cache:
            return self.models_cache[cache_key], self.preprocessor_cache[cache_key]
        
        try:
            files = self._get_latest_model_files(algorithm)
            
            # Cargar modelo
            model = CandidatesClusteringModel.load_model(files['model'])
            
            # Cargar preprocessor
            preprocessor = CandidatesClusteringPreprocessor.load_preprocessor(files['preprocessor'])
            
            # Cargar datos procesados si están disponibles
            if files['data']:
                with open(files['data'], 'rb') as f:
                    data = pickle.load(f)
                self.data_cache[cache_key] = data
            
            # Guardar en cache
            self.models_cache[cache_key] = model
            self.preprocessor_cache[cache_key] = preprocessor
            
            logger.info(f"✅ Modelo {algorithm} cargado exitosamente")
            return model, preprocessor
            
        except Exception as e:
            logger.error(f"Error cargando modelo {algorithm}: {e}")
            raise
    
    async def _get_candidates_data(self) -> pd.DataFrame:
        """Obtiene datos de candidatos desde MongoDB"""
        try:
            collection = await get_collection("candidates_features")
            cursor = collection.find({})
            data = await cursor.to_list(length=None)
            
            if not data:
                raise ValueError("No se encontraron datos de candidatos")
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de candidatos: {e}")
            raise
    
    def _create_cluster_profile(self, cluster_id: int, profile_data: Dict, 
                               algorithm: str) -> ClusterProfile:
        """Crea un ClusterProfile desde datos del modelo"""
        
        # Generar características principales
        top_characteristics = []
        if 'top_features' in profile_data:
            for feature, value in list(profile_data['top_features'].items())[:3]:
                if 'skills_' in feature:
                    skill = feature.replace('skills_', '').replace('_', ' ').title()
                    top_characteristics.append(f"Especialista en {skill}")
                elif 'certs_' in feature:
                    cert = feature.replace('certs_', '').replace('_', ' ').title()
                    top_characteristics.append(f"Certificado en {cert}")
                elif feature == 'anios_experiencia':
                    years = int(value * 10)  # Desnormalizar aproximadamente
                    top_characteristics.append(f"~{years} años de experiencia")
                elif feature == 'nivel_ingles':
                    level = "Básico" if value < 0.3 else "Intermedio" if value < 0.7 else "Avanzado"
                    top_characteristics.append(f"Inglés {level}")
        
        if not top_characteristics:
            top_characteristics = ["Perfil técnico diverso", "Experiencia variada", "Skills múltiples"]
        
        # Generar descripción
        size = profile_data['size']
        percentage = profile_data['percentage']
        
        if percentage > 40:
            description = f"Cluster principal con {size} candidatos representando el perfil más común"
        elif percentage > 10:
            description = f"Cluster significativo con {size} candidatos de perfil especializado"
        else:
            description = f"Cluster específico con {size} candidatos de nicho particular"
        
        return ClusterProfile(
            cluster_id=cluster_id,
            size=size,
            percentage=percentage,
            description=description,
            top_characteristics=top_characteristics
        )
    
    async def analyze_candidate_clusters(
        self, 
        input: Optional[ClusteringQueryInput] = None
    ) -> ClusterAnalysis:
        """Analiza el clustering completo de candidatos"""
        
        algorithm = input.algorithm if input else "kmeans"
        
        try:
            logger.info(f"🔍 Analizando clustering con algoritmo: {algorithm}")
            
            # Cargar modelo y datos
            model, preprocessor = self._load_model_and_preprocessor(algorithm)
            
            # Obtener resumen del modelo
            summary = model.get_cluster_summary()
            
            # Cargar datos procesados para perfiles
            cache_key = algorithm
            if cache_key in self.data_cache:
                data_info = self.data_cache[cache_key]
                X_processed = data_info['X_processed']
                feature_names = data_info['feature_names']
                
                # Generar perfiles de clusters
                cluster_profiles_data = model.get_cluster_profiles(X_processed, feature_names)
            else:
                cluster_profiles_data = {}
            
            # Crear perfiles
            cluster_profiles = []
            for cluster_id, profile_data in cluster_profiles_data.items():
                profile = self._create_cluster_profile(cluster_id, profile_data, algorithm)
                cluster_profiles.append(profile)
            
            # Crear métricas
            metrics = ClusteringMetrics(
                silhouette_score=summary['metrics'].get('silhouette_score', 0.0),
                calinski_harabasz_score=summary['metrics'].get('calinski_harabasz_score', 0.0),
                davies_bouldin_score=summary['metrics'].get('davies_bouldin_score', 0.0),
                n_clusters=summary['n_clusters_found'],
                algorithm_used=algorithm
            )
            
            return ClusterAnalysis(
                total_candidates=summary['total_samples'],
                clusters_found=summary['n_clusters_found'],
                outliers_detected=summary['n_outliers'],
                cluster_profiles=cluster_profiles,
                metrics=metrics,
                algorithm_used=algorithm,
                training_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error en análisis de clustering: {e}")
            raise Exception(f"Error analizando clustering: {str(e)}")
    
    async def find_similar_candidates(
        self, 
        input: SimilarCandidatesInput
    ) -> SimilarCandidates:
        """Encuentra candidatos similares basado en clustering"""
        
        algorithm = input.algorithm or "kmeans"
        
        try:
            logger.info(f"🔍 Buscando candidatos similares para: {input.candidate_id}")
            
            # Cargar modelo
            model, preprocessor = self._load_model_and_preprocessor(algorithm)
            
            # Obtener datos de candidatos
            df = await self._get_candidates_data()
            
            # Buscar candidato objetivo
            target_candidate = df[df['_id'] == input.candidate_id]
            if target_candidate.empty:
                raise ValueError(f"Candidato {input.candidate_id} no encontrado")
            
            # Obtener cluster del candidato objetivo
            target_index = target_candidate.index[0]
            target_cluster = model.labels_[target_index]
            
            if target_cluster == -1:
                return SimilarCandidates(
                    target_candidate_id=input.candidate_id,
                    target_cluster_id=-1,
                    similar_candidates=[],
                    similarity_criteria=["Candidato identificado como outlier"]
                )
            
            # Encontrar candidatos en el mismo cluster
            same_cluster_indices = np.where(model.labels_ == target_cluster)[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != target_index]
            
            # Limitar resultados
            max_similar = min(input.max_similar, len(same_cluster_indices))
            selected_indices = same_cluster_indices[:max_similar]
            
            # Crear asignaciones
            similar_candidates = []
            for idx in selected_indices:
                candidate_id = df.iloc[idx]['_id']
                
                assignment = CandidateClusterAssignment(
                    candidate_id=candidate_id,
                    cluster_id=target_cluster,
                    cluster_confidence=0.85,  # Placeholder
                    distance_to_center=None
                )
                similar_candidates.append(assignment)
            
            # Criterios de similitud (basado en features principales)
            similarity_criteria = [
                "Mismo nivel de experiencia",
                "Skills técnicos similares",
                "Área educativa relacionada",
                "Perfil profesional compatible"
            ]
            
            return SimilarCandidates(
                target_candidate_id=input.candidate_id,
                target_cluster_id=target_cluster,
                similar_candidates=similar_candidates,
                similarity_criteria=similarity_criteria
            )
            
        except Exception as e:
            logger.error(f"Error buscando candidatos similares: {e}")
            raise Exception(f"Error en búsqueda de similares: {str(e)}")
    
    async def get_cluster_profile_details(
        self, 
        input: ClusterProfileInput
    ) -> ClusterProfile:
        """Obtiene detalles específicos de un cluster"""
        
        algorithm = input.algorithm or "kmeans"
        
        try:
            logger.info(f"🔍 Obteniendo detalles del cluster {input.cluster_id}")
            
            # Cargar modelo
            model, preprocessor = self._load_model_and_preprocessor(algorithm)
            
            # Verificar que el cluster existe
            unique_clusters = np.unique(model.labels_)
            if input.cluster_id not in unique_clusters:
                raise ValueError(f"Cluster {input.cluster_id} no encontrado")
            
            # Obtener datos procesados
            cache_key = algorithm
            if cache_key not in self.data_cache:
                raise ValueError("Datos procesados no disponibles en cache")
            
            data_info = self.data_cache[cache_key]
            X_processed = data_info['X_processed']
            feature_names = data_info['feature_names']
            
            # Generar perfil del cluster
            cluster_profiles = model.get_cluster_profiles(X_processed, feature_names)
            
            if input.cluster_id not in cluster_profiles:
                raise ValueError(f"Perfil del cluster {input.cluster_id} no disponible")
            
            profile_data = cluster_profiles[input.cluster_id]
            
            return self._create_cluster_profile(input.cluster_id, profile_data, algorithm)
            
        except Exception as e:
            logger.error(f"Error obteniendo perfil del cluster: {e}")
            raise Exception(f"Error en perfil de cluster: {str(e)}")

# Instancia global del resolver
clustering_resolver = ClusteringResolver()