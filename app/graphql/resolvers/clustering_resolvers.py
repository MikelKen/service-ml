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
    ClusteringQueryInput, SimilarCandidatesInput, ClusterProfileInput,
    CandidateInCluster, CandidatesInCluster, GetCandidatesInClusterInput
)
from app.config.mongodb_connection import get_collection
from app.config.connection import db
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
    
    async def _get_candidate_from_postgres(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos detallados de un candidato desde PostgreSQL"""
        try:
            if not await db.test_connection():
                logger.warning("PostgreSQL not connected, attempting to connect...")
                await db.connect()
            
            async with await db.get_connection() as conn:
                # Consulta para obtener candidato por ID
                query = """
                    SELECT 
                        id, 
                        nombre, 
                        email, 
                        telefono,
                        anios_experiencia,
                        nivel_educacion,
                        habilidades,
                        idiomas,
                        certificaciones
                    FROM postulaciones 
                    WHERE id = $1
                """
                
                result = await conn.fetchrow(query, candidate_id)
                
                if result:
                    # Convertir a diccionario
                    candidate_data = dict(result)
                    logger.info(f"✅ Datos de candidato {candidate_id} obtenidos de PostgreSQL")
                    return candidate_data
                else:
                    logger.warning(f"Candidato {candidate_id} no encontrado en PostgreSQL")
                    return None
                    
        except Exception as e:
            logger.error(f"Error obteniendo candidato de PostgreSQL: {e}")
            return None
    
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
    
    async def get_candidates_in_cluster(
        self,
        input: GetCandidatesInClusterInput
    ) -> CandidatesInCluster:
        """Obtiene todos los candidatos pertenecientes a un cluster específico
        
        Flujo:
        1. Obtiene IDs de candidatos del clustering
        2. Consulta PostgreSQL para obtener datos completos
        3. Enriquece con distancias del clustering
        """
        
        algorithm = input.algorithm or "kmeans"
        
        try:
            logger.info(f"📊 Obteniendo candidatos del cluster {input.cluster_id}")
            
            # 1. Cargar modelo de clustering
            model, preprocessor = self._load_model_and_preprocessor(algorithm)
            
            # 2. Verificar que el cluster existe
            unique_clusters = np.unique(model.labels_)
            if input.cluster_id not in unique_clusters:
                raise ValueError(f"Cluster {input.cluster_id} no encontrado")
            
            # 3. Obtener datos de candidatos desde MongoDB (para IDs y distancias)
            df = await self._get_candidates_data()
            
            # 4. Obtener datos procesados para distancias
            cache_key = algorithm
            distance_data = None
            if cache_key in self.data_cache:
                data_info = self.data_cache[cache_key]
                X_processed = data_info['X_processed']
                
                # Calcular distancias al centro del cluster si es kmeans
                if algorithm == "kmeans" and hasattr(model.model, 'cluster_centers_'):
                    cluster_center = model.model.cluster_centers_[input.cluster_id]
                    distance_data = {}
            
            # 5. Encontrar índices de candidatos en este cluster
            cluster_indices = np.where(model.labels_ == input.cluster_id)[0]
            
            logger.info(f"Found {len(cluster_indices)} candidates in cluster {input.cluster_id}")
            
            # 6. Limitar resultados si es necesario
            if input.limit and input.limit > 0:
                cluster_indices = cluster_indices[:input.limit]
            
            # 7. Extraer datos de candidatos
            candidates_data = []
            total_candidates = len(np.where(model.labels_ == input.cluster_id)[0])
            
            # Conectar a PostgreSQL una sola vez
            if not await db.test_connection():
                logger.info("Conectando a PostgreSQL...")
                await db.connect()
            
            for idx in cluster_indices:
                candidate_row = df.iloc[idx]
                candidate_id = str(candidate_row.get('_id', ''))
                
                logger.info(f"Procesando candidato: {candidate_id}")
                
                # Calcular distancia al centro si es posible
                distance_to_center = None
                if distance_data is not None and algorithm == "kmeans":
                    try:
                        sample_distance = np.linalg.norm(X_processed[idx] - cluster_center)
                        distance_to_center = float(sample_distance)
                    except Exception as e:
                        logger.warning(f"Error calculando distancia: {e}")
                        distance_to_center = None
                
                # **NUEVO: Obtener datos completos desde PostgreSQL**
                postgres_data = await self._get_candidate_from_postgres(candidate_id)
                
                if postgres_data:
                    # Usar datos de PostgreSQL (completos y actualizados)
                    candidate = CandidateInCluster(
                        candidate_id=candidate_id,
                        name=postgres_data.get('nombre', ''),
                        email=postgres_data.get('email', ''),
                        years_experience=postgres_data.get('anios_experiencia'),
                        education_area=postgres_data.get('nivel_educacion', ''),
                        work_area='',
                        skills=self._parse_list_field(postgres_data.get('habilidades')),
                        certifications=self._parse_list_field(postgres_data.get('certificaciones')),
                        english_level=postgres_data.get('idiomas'),
                        cluster_id=input.cluster_id,
                        distance_to_center=distance_to_center
                    )
                else:
                    # Fallback a datos de MongoDB si no está en PostgreSQL
                    logger.warning(f"Candidato {candidate_id} no encontrado en PostgreSQL, usando MongoDB")
                    
                    skills = []
                    if 'skills' in candidate_row and candidate_row['skills']:
                        if isinstance(candidate_row['skills'], (list, tuple)):
                            skills = list(candidate_row['skills'])
                        elif isinstance(candidate_row['skills'], str):
                            skills = [s.strip() for s in str(candidate_row['skills']).split(',')]
                    
                    certifications = []
                    if 'certifications' in candidate_row and candidate_row['certifications']:
                        if isinstance(candidate_row['certifications'], (list, tuple)):
                            certifications = list(candidate_row['certifications'])
                        elif isinstance(candidate_row['certifications'], str):
                            certifications = [c.strip() for c in str(candidate_row['certifications']).split(',')]
                    
                    english_level = None
                    if 'english_level' in candidate_row:
                        level_value = candidate_row['english_level']
                        if level_value == 0 or level_value == '0':
                            english_level = "No tiene"
                        elif level_value == 1 or level_value == '1':
                            english_level = "Básico"
                        elif level_value == 2 or level_value == '2':
                            english_level = "Intermedio"
                        elif level_value == 3 or level_value == '3':
                            english_level = "Avanzado"
                        else:
                            english_level = str(level_value)
                    
                    candidate = CandidateInCluster(
                        candidate_id=candidate_id,
                        name=str(candidate_row.get('name', candidate_row.get('full_name', ''))),
                        email=str(candidate_row.get('email', '')),
                        years_experience=float(candidate_row['anios_experiencia']) if 'anios_experiencia' in candidate_row else None,
                        education_area=str(candidate_row.get('area_educacion', '')),
                        work_area=str(candidate_row.get('area_trabajo', '')),
                        skills=skills if skills else None,
                        certifications=certifications if certifications else None,
                        english_level=english_level,
                        cluster_id=input.cluster_id,
                        distance_to_center=distance_to_center
                    )
                
                candidates_data.append(candidate)
            
            # 8. Calcular porcentaje
            if total_candidates > 0:
                cluster_percentage = (total_candidates / len(df)) * 100
            else:
                cluster_percentage = 0.0
            
            logger.info(f"✅ {len(candidates_data)} candidatos del cluster {input.cluster_id} procesados")
            
            return CandidatesInCluster(
                cluster_id=input.cluster_id,
                total_candidates=total_candidates,
                cluster_percentage=cluster_percentage,
                candidates=candidates_data
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo candidatos del cluster: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error obteniendo candidatos del cluster: {str(e)}")
    
    def _parse_list_field(self, field_value: Any) -> Optional[List[str]]:
        """Convierte un campo en lista si es necesario"""
        if not field_value:
            return None
        
        if isinstance(field_value, (list, tuple)):
            return list(field_value)
        elif isinstance(field_value, str):
            # Intentar parsear como JSON array
            import json
            try:
                parsed = json.loads(field_value)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            # Si no es JSON, dividir por comas
            return [s.strip() for s in field_value.split(',') if s.strip()]
        
        return None

# Instancia global del resolver
clustering_resolver = ClusteringResolver()