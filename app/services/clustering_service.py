"""
Servicio de Machine Learning para Clustering de Candidatos
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import pickle
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings
warnings.filterwarnings('ignore')

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateClusteringService:
    """Servicio principal de Clustering de Candidatos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.clustering_model = None
        self.processed_data = None
        self.original_data = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.feature_names = []
        self.silhouette_score_value = 0.0
        self.model_path = "trained_models/candidate_clustering_model.pkl"
        self.data_path = "candidatos_clustering_dataset.csv"
        self.is_model_trained = False
        
        # Intentar cargar modelo si existe
        self.load_model()
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Carga y prepara los datos de candidatos"""
        if data_path is None:
            data_path = self.data_path
            
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Datos cargados: {len(df)} candidatos")
            return df
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocesa los datos para clustering"""
        try:
            # Guardar datos originales
            self.original_data = df.copy()
            
            # Seleccionar características para clustering
            numerical_features = [
                'edad', 'años_experiencia', 'salario_esperado', 
                'liderazgo_equipos', 'proyectos_completados'
            ]
            
            categorical_features = [
                'nivel_educacion', 'area_especialidad', 'modalidad_trabajo',
                'ubicacion', 'industria_experiencia', 'disponibilidad_viajar',
                'educacion_continua'
            ]
            
            # Procesar características numéricas
            numerical_data = df[numerical_features].fillna(0)
            
            # Procesar características categóricas
            categorical_data = pd.DataFrame()
            for feature in categorical_features:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                
                # Llenar valores faltantes
                filled_values = df[feature].fillna('Desconocido')
                
                # Ajustar y transformar
                try:
                    encoded_values = self.label_encoders[feature].fit_transform(filled_values)
                except:
                    # Si hay nuevos valores, manejarlos
                    encoded_values = []
                    for val in filled_values:
                        try:
                            encoded_values.append(self.label_encoders[feature].transform([val])[0])
                        except:
                            encoded_values.append(0)  # Valor por defecto
                    encoded_values = np.array(encoded_values)
                
                categorical_data[feature] = encoded_values
            
            # Procesar habilidades como características binarias
            skills_features = self._process_skills(df)
            
            # Combinar todas las características
            combined_features = pd.concat([
                numerical_data,
                categorical_data,
                skills_features
            ], axis=1)
            
            # Guardar nombres de características
            self.feature_names = combined_features.columns.tolist()
            
            # Normalizar datos
            processed_data = self.scaler.fit_transform(combined_features)
            self.processed_data = processed_data
            
            logger.info(f"Datos preprocesados: {processed_data.shape[1]} características")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            raise
    
    def _process_skills(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa habilidades técnicas y blandas como características binarias"""
        skills_df = pd.DataFrame()
        
        # Combinar habilidades técnicas y blandas
        all_skills = set()
        for _, row in df.iterrows():
            tech_skills = str(row['habilidades_tecnicas']).split(',')
            soft_skills = str(row['habilidades_blandas']).split(',')
            all_skills.update([skill.strip() for skill in tech_skills + soft_skills])
        
        # Remover skills vacías o inválidas
        all_skills = {skill for skill in all_skills if skill and skill != 'nan'}
        
        # Crear características binarias para las habilidades más comunes (top 20)
        skill_counts = {}
        for _, row in df.iterrows():
            tech_skills = [s.strip() for s in str(row['habilidades_tecnicas']).split(',')]
            soft_skills = [s.strip() for s in str(row['habilidades_blandas']).split(',')]
            for skill in tech_skills + soft_skills:
                if skill and skill != 'nan':
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Tomar las 20 habilidades más comunes
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        top_skills = [skill[0] for skill in top_skills]
        
        # Crear columnas binarias
        for skill in top_skills:
            skills_df[f'skill_{skill.replace(" ", "_")}'] = df.apply(
                lambda row: 1 if skill in str(row['habilidades_tecnicas']) + ',' + str(row['habilidades_blandas']) else 0, 
                axis=1
            )
        
        return skills_df
    
    def perform_clustering(self, 
                          data: np.ndarray, 
                          algorithm: str = "kmeans",
                          n_clusters: int = None,
                          **kwargs) -> Tuple[np.ndarray, float]:
        """Realiza clustering en los datos"""
        try:
            if algorithm.lower() == "kmeans":
                if n_clusters is None:
                    n_clusters = self._determine_optimal_clusters(data)
                
                model = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
                
            elif algorithm.lower() == "hierarchical":
                if n_clusters is None:
                    n_clusters = self._determine_optimal_clusters(data)
                
                linkage = kwargs.get('linkage', 'ward')
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage
                )
                
            elif algorithm.lower() == "dbscan":
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                
            else:
                raise ValueError(f"Algoritmo no soportado: {algorithm}")
            
            # Entrenar modelo
            cluster_labels = model.fit_predict(data)
            self.cluster_labels = cluster_labels
            self.clustering_model = model
            
            # Calcular centros de clusters (solo para KMeans)
            if algorithm.lower() == "kmeans":
                self.cluster_centers = model.cluster_centers_
            else:
                self.cluster_centers = self._calculate_cluster_centers(data, cluster_labels)
            
            # Calcular silhouette score
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(data, cluster_labels)
            else:
                silhouette_avg = 0.0
            
            self.silhouette_score_value = silhouette_avg
            self.is_model_trained = True
            
            logger.info(f"Clustering completado: {len(set(cluster_labels))} clusters, "
                       f"Silhouette Score: {silhouette_avg:.3f}")
            
            return cluster_labels, silhouette_avg
            
        except Exception as e:
            logger.error(f"Error en clustering: {str(e)}")
            raise
    
    def _determine_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Determina el número óptimo de clusters usando el método del codo"""
        try:
            if len(data) < max_k:
                max_k = len(data) - 1
            
            if max_k < 2:
                return 2
            
            inertias = []
            k_range = range(2, min(max_k + 1, len(data)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            # Método del codo simplificado
            if len(inertias) >= 3:
                # Encontrar el punto donde la reducción de inercia es menor
                deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                optimal_idx = 0
                for i in range(1, len(deltas)):
                    if deltas[i] < deltas[optimal_idx] * 0.7:  # 30% de reducción
                        optimal_idx = i
                        break
                return k_range[optimal_idx]
            else:
                return 3  # Valor por defecto
                
        except Exception as e:
            logger.warning(f"Error determinando clusters óptimos: {str(e)}")
            return 3  # Valor por defecto
    
    def _calculate_cluster_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calcula centros de clusters para algoritmos que no los calculan automáticamente"""
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            if label == -1:  # Ruido en DBSCAN
                continue
            cluster_points = data[labels == label]
            center = np.mean(cluster_points, axis=0)
            centers.append(center)
        
        return np.array(centers)
    
    def analyze_clusters(self) -> List[Dict[str, Any]]:
        """Analiza las características de cada cluster"""
        if not self.is_model_trained or self.original_data is None:
            raise ValueError("Modelo no entrenado o datos no disponibles")
        
        try:
            clusters_info = []
            unique_labels = np.unique(self.cluster_labels)
            
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Ruido en DBSCAN
                    continue
                
                # Obtener candidatos del cluster
                cluster_mask = self.cluster_labels == cluster_id
                cluster_candidates = self.original_data[cluster_mask]
                
                # Calcular estadísticas
                info = {
                    'cluster_id': int(cluster_id),
                    'cluster_name': f"Cluster {cluster_id}",
                    'candidate_count': len(cluster_candidates),
                    'description': self._generate_cluster_description(cluster_candidates),
                    'key_characteristics': self._extract_key_characteristics(cluster_candidates),
                    'avg_experience_years': float(cluster_candidates['años_experiencia'].mean()),
                    'avg_salary_expectation': float(cluster_candidates['salario_esperado'].mean()),
                    'common_skills': self._get_common_skills(cluster_candidates),
                    'common_industries': self._get_common_values(cluster_candidates, 'industria_experiencia'),
                    'education_levels': self._get_common_values(cluster_candidates, 'nivel_educacion')
                }
                
                clusters_info.append(info)
            
            return clusters_info
            
        except Exception as e:
            logger.error(f"Error analizando clusters: {str(e)}")
            raise
    
    def _generate_cluster_description(self, cluster_data: pd.DataFrame) -> str:
        """Genera descripción automática del cluster"""
        try:
            avg_age = cluster_data['edad'].mean()
            avg_exp = cluster_data['años_experiencia'].mean()
            most_common_industry = cluster_data['industria_experiencia'].mode().iloc[0]
            most_common_education = cluster_data['nivel_educacion'].mode().iloc[0]
            
            description = f"Profesionales con {avg_exp:.1f} años de experiencia promedio, "
            description += f"edad promedio {avg_age:.1f} años, "
            description += f"principalmente en {most_common_industry} "
            description += f"con {most_common_education}"
            
            return description
        except:
            return "Cluster de profesionales con características similares"
    
    def _extract_key_characteristics(self, cluster_data: pd.DataFrame) -> List[str]:
        """Extrae características clave del cluster"""
        characteristics = []
        
        try:
            # Experiencia
            avg_exp = cluster_data['años_experiencia'].mean()
            if avg_exp < 3:
                characteristics.append("Profesionales junior")
            elif avg_exp < 7:
                characteristics.append("Profesionales con experiencia media")
            else:
                characteristics.append("Profesionales senior")
            
            # Salario
            avg_salary = cluster_data['salario_esperado'].mean()
            if avg_salary < 50000:
                characteristics.append("Expectativas salariales bajas-medias")
            elif avg_salary < 80000:
                characteristics.append("Expectativas salariales medias")
            else:
                characteristics.append("Expectativas salariales altas")
            
            # Modalidad de trabajo más común
            common_modality = cluster_data['modalidad_trabajo'].mode().iloc[0]
            characteristics.append(f"Prefieren trabajo {common_modality.lower()}")
            
            # Disponibilidad para viajar
            travel_yes = (cluster_data['disponibilidad_viajar'] == 'Sí').sum()
            travel_no = (cluster_data['disponibilidad_viajar'] == 'No').sum()
            if travel_yes > travel_no:
                characteristics.append("Disponibles para viajar")
            
        except Exception as e:
            logger.warning(f"Error extrayendo características: {str(e)}")
        
        return characteristics[:5]  # Limitar a 5 características principales
    
    def _get_common_skills(self, cluster_data: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Obtiene habilidades más comunes en el cluster"""
        all_skills = []
        
        for _, row in cluster_data.iterrows():
            tech_skills = str(row['habilidades_tecnicas']).split(',')
            soft_skills = str(row['habilidades_blandas']).split(',')
            all_skills.extend([skill.strip() for skill in tech_skills + soft_skills])
        
        # Contar frecuencias
        skill_counts = {}
        for skill in all_skills:
            if skill and skill != 'nan':
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Devolver las más comunes
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        return [skill[0] for skill in sorted_skills[:top_n]]
    
    def _get_common_values(self, cluster_data: pd.DataFrame, column: str, top_n: int = 3) -> List[str]:
        """Obtiene valores más comunes para una columna"""
        try:
            value_counts = cluster_data[column].value_counts()
            return value_counts.head(top_n).index.tolist()
        except:
            return []
    
    def get_candidate_cluster_assignments(self) -> List[Dict[str, Any]]:
        """Obtiene asignaciones de candidatos a clusters"""
        if not self.is_model_trained or self.original_data is None:
            raise ValueError("Modelo no entrenado o datos no disponibles")
        
        try:
            assignments = []
            
            for idx, (_, candidate) in enumerate(self.original_data.iterrows()):
                cluster_id = int(self.cluster_labels[idx])
                
                # Calcular distancia al centro del cluster
                if self.cluster_centers is not None and cluster_id >= 0:
                    candidate_features = self.processed_data[idx]
                    if cluster_id < len(self.cluster_centers):
                        distance = euclidean_distances(
                            [candidate_features], 
                            [self.cluster_centers[cluster_id]]
                        )[0][0]
                    else:
                        distance = 0.0
                else:
                    distance = 0.0
                
                # Calcular score de similitud (inverso de la distancia normalizada)
                similarity_score = 1.0 / (1.0 + distance)
                
                assignment = {
                    'candidate_id': int(candidate['id']),
                    'candidate_name': candidate['nombre'],
                    'cluster_id': cluster_id,
                    'cluster_name': f"Cluster {cluster_id}" if cluster_id >= 0 else "Sin Cluster",
                    'similarity_score': float(similarity_score),
                    'distance_to_centroid': float(distance),
                    'profile_summary': f"{candidate['area_especialidad']} - {candidate['años_experiencia']} años exp."
                }
                
                assignments.append(assignment)
            
            return assignments
            
        except Exception as e:
            logger.error(f"Error obteniendo asignaciones: {str(e)}")
            raise
    
    def find_similar_candidates(self, 
                               candidate_profile: Dict[str, Any], 
                               max_results: int = 10) -> List[Dict[str, Any]]:
        """Encuentra candidatos similares a un perfil dado"""
        if not self.is_model_trained:
            raise ValueError("Modelo no entrenado")
        
        try:
            # Crear DataFrame temporal con el nuevo candidato
            temp_df = pd.DataFrame([candidate_profile])
            
            # Agregar a los datos originales temporalmente
            combined_df = pd.concat([self.original_data, temp_df], ignore_index=True)
            
            # Preprocesar incluyendo el nuevo candidato
            processed_combined = self._preprocess_single_candidate(candidate_profile)
            
            # Predecir cluster para el nuevo candidato
            if hasattr(self.clustering_model, 'predict'):
                predicted_cluster = self.clustering_model.predict([processed_combined])[0]
            else:
                # Para DBSCAN, encontrar el cluster más cercano
                distances = euclidean_distances([processed_combined], self.cluster_centers)
                predicted_cluster = np.argmin(distances[0])
            
            # Encontrar candidatos en el mismo cluster
            same_cluster_mask = self.cluster_labels == predicted_cluster
            same_cluster_indices = np.where(same_cluster_mask)[0]
            
            # Calcular similitudes
            similarities = []
            for idx in same_cluster_indices:
                candidate_features = self.processed_data[idx]
                similarity = cosine_similarity([processed_combined], [candidate_features])[0][0]
                
                similarities.append({
                    'index': idx,
                    'similarity': similarity,
                    'candidate_data': self.original_data.iloc[idx]
                })
            
            # Ordenar por similitud y tomar los mejores
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_similar = similarities[:max_results]
            
            # Formatear resultados
            results = []
            for item in top_similar:
                candidate_data = item['candidate_data']
                result = {
                    'candidate_id': int(candidate_data['id']),
                    'candidate_name': candidate_data['nombre'],
                    'cluster_id': predicted_cluster,
                    'cluster_name': f"Cluster {predicted_cluster}",
                    'similarity_score': float(item['similarity']),
                    'distance_to_centroid': 0.0,  # Se podría calcular
                    'profile_summary': f"{candidate_data['area_especialidad']} - {candidate_data['años_experiencia']} años exp."
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error encontrando candidatos similares: {str(e)}")
            raise
    
    def _preprocess_single_candidate(self, candidate_profile: Dict[str, Any]) -> np.ndarray:
        """Preprocesa un solo candidato para predicción"""
        try:
            # Crear DataFrame temporal
            temp_df = pd.DataFrame([candidate_profile])
            
            # Aplicar el mismo preprocesamiento que en entrenamiento
            numerical_features = [
                'edad', 'años_experiencia', 'salario_esperado', 
                'liderazgo_equipos', 'proyectos_completados'
            ]
            
            categorical_features = [
                'nivel_educacion', 'area_especialidad', 'modalidad_trabajo',
                'ubicacion', 'industria_experiencia', 'disponibilidad_viajar',
                'educacion_continua'
            ]
            
            # Procesar características numéricas
            numerical_data = temp_df[numerical_features].fillna(0)
            
            # Procesar características categóricas
            categorical_data = pd.DataFrame()
            for feature in categorical_features:
                value = temp_df[feature].iloc[0]
                if feature in self.label_encoders:
                    try:
                        encoded_value = self.label_encoders[feature].transform([value])[0]
                    except:
                        encoded_value = 0  # Valor por defecto para valores no vistos
                else:
                    encoded_value = 0
                
                categorical_data[feature] = [encoded_value]
            
            # Procesar habilidades (simplificado)
            skills_data = pd.DataFrame()
            for feature_name in self.feature_names:
                if feature_name.startswith('skill_'):
                    skill_name = feature_name.replace('skill_', '').replace('_', ' ')
                    tech_skills = str(candidate_profile.get('habilidades_tecnicas', ''))
                    soft_skills = str(candidate_profile.get('habilidades_blandas', ''))
                    has_skill = 1 if skill_name in tech_skills + ',' + soft_skills else 0
                    skills_data[feature_name] = [has_skill]
            
            # Combinar características
            combined_features = pd.concat([
                numerical_data,
                categorical_data,
                skills_data
            ], axis=1)
            
            # Asegurar que todas las columnas estén presentes
            for feature_name in self.feature_names:
                if feature_name not in combined_features.columns:
                    combined_features[feature_name] = 0
            
            # Reordenar columnas
            combined_features = combined_features[self.feature_names]
            
            # Normalizar
            processed_features = self.scaler.transform(combined_features)
            
            return processed_features[0]
            
        except Exception as e:
            logger.error(f"Error preprocesando candidato individual: {str(e)}")
            raise
    
    def save_model(self) -> bool:
        """Guarda el modelo entrenado"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'clustering_model': self.clustering_model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'cluster_centers': self.cluster_centers,
                'feature_names': self.feature_names,
                'silhouette_score': self.silhouette_score_value,
                'is_trained': self.is_model_trained,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Modelo guardado en: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """Carga el modelo entrenado"""
        try:
            if not os.path.exists(self.model_path):
                logger.info("No se encontró modelo guardado")
                return False
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.clustering_model = model_data['clustering_model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.cluster_centers = model_data['cluster_centers']
            self.feature_names = model_data['feature_names']
            self.silhouette_score_value = model_data['silhouette_score']
            self.is_model_trained = model_data['is_trained']
            
            logger.info("Modelo cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            return False
    
    def train_clustering_model(self, 
                              algorithm: str = "kmeans",
                              n_clusters: int = None,
                              **kwargs) -> Dict[str, Any]:
        """Entrena el modelo completo de clustering"""
        start_time = time.time()
        
        try:
            # 1. Cargar datos
            df = self.load_data()
            
            # 2. Preprocesar datos
            processed_data = self.preprocess_data(df)
            
            # 3. Realizar clustering
            cluster_labels, silhouette_score = self.perform_clustering(
                processed_data, algorithm, n_clusters, **kwargs
            )
            
            # 4. Analizar clusters
            clusters_info = self.analyze_clusters()
            
            # 5. Obtener asignaciones
            candidate_assignments = self.get_candidate_cluster_assignments()
            
            # 6. Guardar modelo
            self.save_model()
            
            # Calcular tiempo de procesamiento
            processing_time = (time.time() - start_time) * 1000
            
            # Preparar resultado
            result = {
                'total_candidates': len(df),
                'num_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'silhouette_score': float(silhouette_score),
                'clusters': clusters_info,
                'candidate_assignments': candidate_assignments,
                'processing_time_ms': float(processing_time),
                'model_parameters': {
                    'algorithm': algorithm,
                    'n_clusters': n_clusters or len(set(cluster_labels)),
                    **kwargs
                }
            }
            
            logger.info(f"Entrenamiento completado en {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            raise


# Instancia global del servicio
clustering_service = CandidateClusteringService()


# Funciones de utilidad para GraphQL
def train_candidate_clustering(algorithm: str = "kmeans", 
                             n_clusters: int = None,
                             **kwargs) -> Dict[str, Any]:
    """Entrena modelo de clustering de candidatos"""
    return clustering_service.train_clustering_model(algorithm, n_clusters, **kwargs)


def get_clustering_results() -> Dict[str, Any]:
    """Obtiene resultados del clustering actual"""
    if not clustering_service.is_model_trained:
        raise ValueError("Modelo no entrenado. Entrene el modelo primero.")
    
    clusters_info = clustering_service.analyze_clusters()
    candidate_assignments = clustering_service.get_candidate_cluster_assignments()
    
    return {
        'total_candidates': len(clustering_service.original_data),
        'num_clusters': len(clusters_info),
        'silhouette_score': clustering_service.silhouette_score_value,
        'clusters': clusters_info,
        'candidate_assignments': candidate_assignments,
        'processing_time_ms': 0.0,
        'model_parameters': {'algorithm': 'loaded_model'}
    }


def find_similar_candidates_to_profile(candidate_profile: Dict[str, Any], 
                                     max_results: int = 10) -> Dict[str, Any]:
    """Encuentra candidatos similares a un perfil"""
    similar_candidates = clustering_service.find_similar_candidates(
        candidate_profile, max_results
    )
    
    # Crear objeto CandidateProfile para referencia
    reference_profile = {
        'id': 0,
        'nombre': candidate_profile['nombre'],
        **candidate_profile
    }
    
    return {
        'reference_candidate': reference_profile,
        'similar_candidates': similar_candidates,
        'similarity_criteria': ['experiencia', 'habilidades', 'industria', 'educacion'],
        'total_found': len(similar_candidates)
    }


def get_clustering_analytics() -> Dict[str, Any]:
    """Obtiene analíticas de clustering"""
    if not clustering_service.is_model_trained:
        raise ValueError("Modelo no entrenado")
    
    try:
        df = clustering_service.original_data
        labels = clustering_service.cluster_labels
        
        # Distribución de clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_distribution = {f"Cluster {label}": int(count) 
                              for label, count in zip(unique_labels, counts) if label >= 0}
        
        # Distribución de industrias
        industry_counts = df['industria_experiencia'].value_counts().head(10)
        industry_distribution = {str(k): int(v) for k, v in industry_counts.items()}
        
        # Distribución de educación
        education_counts = df['nivel_educacion'].value_counts()
        education_distribution = {str(k): int(v) for k, v in education_counts.items()}
        
        # Frecuencia de habilidades (simplificado)
        all_skills = []
        for _, row in df.iterrows():
            tech_skills = str(row['habilidades_tecnicas']).split(',')
            soft_skills = str(row['habilidades_blandas']).split(',')
            all_skills.extend([skill.strip() for skill in tech_skills + soft_skills])
        
        skill_counts = {}
        for skill in all_skills:
            if skill and skill != 'nan':
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        top_skills = dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15])
        
        # Rangos salariales por cluster
        salary_ranges = {}
        experience_ranges = {}
        
        for cluster_id in unique_labels:
            if cluster_id >= 0:
                cluster_data = df[labels == cluster_id]
                salary_min = cluster_data['salario_esperado'].min()
                salary_max = cluster_data['salario_esperado'].max()
                salary_ranges[f"Cluster {cluster_id}"] = f"${salary_min:,.0f} - ${salary_max:,.0f}"
                
                exp_min = cluster_data['años_experiencia'].min()
                exp_max = cluster_data['años_experiencia'].max()
                experience_ranges[f"Cluster {cluster_id}"] = f"{exp_min:.0f} - {exp_max:.0f} años"
        
        return {
            'cluster_distribution': cluster_distribution,
            'skill_frequency': top_skills,
            'industry_distribution': industry_distribution,
            'education_distribution': education_distribution,
            'salary_ranges_by_cluster': salary_ranges,
            'experience_ranges_by_cluster': experience_ranges
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo analíticas: {str(e)}")
        raise