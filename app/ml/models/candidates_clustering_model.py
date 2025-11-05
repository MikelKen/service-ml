#!/usr/bin/env python3
"""
🎯 MODELO DE CLUSTERING PARA CANDIDATOS
Implementa K-Means y otros algoritmos para agrupar candidatos por similitud
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandidatesClusteringModel:
    """Modelo de clustering para agrupar candidatos por similitud de perfil"""
    
    def __init__(self, algorithm: str = 'kmeans'):
        self.algorithm = algorithm
        self.model = None
        self.n_clusters = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.metrics = {}
        self.pca = None
        self.is_fitted = False
        
        # Configuraciones por algoritmo
        self.algorithm_configs = {
            'kmeans': {
                'n_clusters': 8,
                'random_state': 42,
                'n_init': 10,
                'max_iter': 300
            },
            'dbscan': {
                'eps': 0.5,
                'min_samples': 30
            },
            'hierarchical': {
                'n_clusters': 8,
                'linkage': 'ward'
            }
        }
    
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 15) -> Dict[str, Any]:
        """Encuentra el número óptimo de clusters usando múltiples métricas"""
        logger.info("🔍 Buscando número óptimo de clusters...")
        
        cluster_range = range(2, max_clusters + 1)
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        inertias = []
        
        for n_clusters in cluster_range:
            logger.info(f"  📊 Evaluando {n_clusters} clusters...")
            
            # K-Means temporal
            temp_kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            temp_labels = temp_kmeans.fit_predict(X)
            
            # Calcular métricas
            silhouette_avg = silhouette_score(X, temp_labels)
            calinski_score = calinski_harabasz_score(X, temp_labels)
            davies_bouldin = davies_bouldin_score(X, temp_labels)
            
            silhouette_scores.append(silhouette_avg)
            calinski_scores.append(calinski_score)
            davies_bouldin_scores.append(davies_bouldin)
            inertias.append(temp_kmeans.inertia_)
            
            logger.info(f"    ✅ Silhouette: {silhouette_avg:.3f}, Calinski: {calinski_score:.2f}")
        
        # Encontrar óptimos
        optimal_silhouette = cluster_range[np.argmax(silhouette_scores)]
        optimal_calinski = cluster_range[np.argmax(calinski_scores)]
        optimal_davies = cluster_range[np.argmin(davies_bouldin_scores)]
        
        # Método del codo (elbow)
        optimal_elbow = self._find_elbow_point(list(cluster_range), inertias)
        
        results = {
            'cluster_range': list(cluster_range),
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'inertias': inertias,
            'optimal_silhouette': optimal_silhouette,
            'optimal_calinski': optimal_calinski,
            'optimal_davies': optimal_davies,
            'optimal_elbow': optimal_elbow,
            'recommended': optimal_silhouette  # Usar silhouette como principal
        }
        
        logger.info("📈 RESULTADOS DE OPTIMIZACIÓN:")
        logger.info(f"  🎯 Silhouette óptimo: {optimal_silhouette} clusters")
        logger.info(f"  📊 Calinski óptimo: {optimal_calinski} clusters")
        logger.info(f"  📉 Davies-Bouldin óptimo: {optimal_davies} clusters")
        logger.info(f"  📍 Método del codo: {optimal_elbow} clusters")
        logger.info(f"  🏆 RECOMENDADO: {results['recommended']} clusters")
        
        return results
    
    def fit(self, X: np.ndarray, n_clusters: Optional[int] = None, 
            find_optimal: bool = True) -> 'CandidatesClusteringModel':
        """Entrena el modelo de clustering"""
        logger.info(f"🚀 ENTRENANDO MODELO DE CLUSTERING: {self.algorithm}")
        logger.info(f"📊 Datos de entrada: {X.shape}")
        
        # Encontrar número óptimo si es necesario
        if find_optimal and self.algorithm == 'kmeans':
            optimization_results = self.find_optimal_clusters(X)
            optimal_clusters = optimization_results['recommended']
            
            if n_clusters is None:
                n_clusters = optimal_clusters
                logger.info(f"🎯 Usando número óptimo: {n_clusters} clusters")
            else:
                logger.info(f"⚠️ Número especificado: {n_clusters}, óptimo sugerido: {optimal_clusters}")
        
        # Configurar modelo según algoritmo
        if self.algorithm == 'kmeans':
            config = self.algorithm_configs['kmeans'].copy()
            if n_clusters:
                config['n_clusters'] = n_clusters
            
            self.model = KMeans(**config)
            self.n_clusters = config['n_clusters']
            
        elif self.algorithm == 'dbscan':
            config = self.algorithm_configs['dbscan']
            self.model = DBSCAN(**config)
            
        elif self.algorithm == 'hierarchical':
            config = self.algorithm_configs['hierarchical'].copy()
            if n_clusters:
                config['n_clusters'] = n_clusters
            
            self.model = AgglomerativeClustering(**config)
            self.n_clusters = config['n_clusters']
        
        # Entrenar modelo
        logger.info("🔄 Entrenando modelo...")
        self.labels_ = self.model.fit_predict(X)
        
        # Obtener centros de clusters (solo para K-Means)
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers_ = self.model.cluster_centers_
        
        # Calcular métricas
        self._calculate_metrics(X, self.labels_)
        
        # PCA para visualización
        self._fit_pca_for_visualization(X)
        
        self.is_fitted = True
        
        logger.info("✅ ENTRENAMIENTO COMPLETADO")
        self._log_training_results()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clusters para nuevos datos"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Usar fit() primero.")
        
        if self.algorithm == 'kmeans':
            return self.model.predict(X)
        else:
            # Para DBSCAN y otros, necesitamos usar fit_predict en todo el dataset
            logger.warning("⚠️ Este algoritmo no soporta predicción incremental")
            return self.model.fit_predict(X)
    
    def get_cluster_profiles(self, X: np.ndarray, feature_names: List[str]) -> Dict[int, Dict]:
        """Genera perfiles descriptivos de cada cluster"""
        logger.info("📊 Generando perfiles de clusters...")
        
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")
        
        profiles = {}
        unique_labels = np.unique(self.labels_)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Outliers en DBSCAN
                continue
            
            # Datos del cluster
            cluster_mask = self.labels_ == cluster_id
            cluster_data = X[cluster_mask]
            
            # Estadísticas básicas
            profile = {
                'cluster_id': int(cluster_id),
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(X) * 100),
                'feature_means': {},
                'feature_stds': {},
                'top_features': {}
            }
            
            # Calcular estadísticas por feature
            for i, feature_name in enumerate(feature_names):
                profile['feature_means'][feature_name] = float(np.mean(cluster_data[:, i]))
                profile['feature_stds'][feature_name] = float(np.std(cluster_data[:, i]))
            
            # Identificar features más distintivas
            if hasattr(self.model, 'cluster_centers_'):
                center = self.cluster_centers_[cluster_id]
                # Features con mayor desviación del promedio global
                global_mean = np.mean(X, axis=0)
                deviations = np.abs(center - global_mean)
                top_indices = np.argsort(deviations)[-5:]  # Top 5
                
                profile['top_features'] = {
                    feature_names[i]: float(center[i]) 
                    for i in top_indices
                }
            
            profiles[cluster_id] = profile
        
        logger.info(f"✅ Perfiles generados para {len(profiles)} clusters")
        return profiles
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del clustering"""
        if not self.is_fitted:
            return {"error": "Modelo no entrenado"}
        
        unique_labels = np.unique(self.labels_)
        n_clusters_found = len(unique_labels)
        n_outliers = np.sum(self.labels_ == -1) if -1 in unique_labels else 0
        
        return {
            'algorithm': self.algorithm,
            'n_clusters_found': int(n_clusters_found),
            'n_outliers': int(n_outliers),
            'total_samples': len(self.labels_),
            'metrics': self.metrics,
            'cluster_sizes': {
                int(label): int(np.sum(self.labels_ == label))
                for label in unique_labels if label != -1
            }
        }
    
    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray):
        """Calcula métricas de calidad del clustering"""
        try:
            # Filtrar outliers para métricas
            valid_mask = labels != -1
            if np.sum(valid_mask) < 2:
                logger.warning("⚠️ Muy pocas muestras para calcular métricas")
                return
            
            X_valid = X[valid_mask]
            labels_valid = labels[valid_mask]
            
            if len(np.unique(labels_valid)) < 2:
                logger.warning("⚠️ Solo un cluster encontrado")
                return
            
            self.metrics = {
                'silhouette_score': float(silhouette_score(X_valid, labels_valid)),
                'calinski_harabasz_score': float(calinski_harabasz_score(X_valid, labels_valid)),
                'davies_bouldin_score': float(davies_bouldin_score(X_valid, labels_valid)),
                'n_clusters': len(np.unique(labels_valid)),
                'n_outliers': int(np.sum(labels == -1))
            }
            
            if hasattr(self.model, 'inertia_'):
                self.metrics['inertia'] = float(self.model.inertia_)
                
        except Exception as e:
            logger.error(f"❌ Error calculando métricas: {e}")
            self.metrics = {'error': str(e)}
    
    def _fit_pca_for_visualization(self, X: np.ndarray):
        """Ajusta PCA para visualización 2D"""
        try:
            self.pca = PCA(n_components=2, random_state=42)
            self.pca.fit(X)
            logger.info(f"📊 PCA ajustado - Varianza explicada: {self.pca.explained_variance_ratio_.sum():.3f}")
        except Exception as e:
            logger.warning(f"⚠️ Error en PCA: {e}")
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Encuentra el punto del codo en la curva de inercia"""
        try:
            # Método de la segunda derivada
            if len(inertias) < 3:
                return k_values[0]
            
            # Calcular segunda derivada
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_deriv)
            
            # Encontrar máximo de segunda derivada
            elbow_idx = np.argmax(second_derivatives) + 1
            return k_values[elbow_idx]
            
        except Exception:
            # Fallback: retornar punto medio
            return k_values[len(k_values) // 2]
    
    def _log_training_results(self):
        """Log de resultados del entrenamiento"""
        logger.info("📊 RESULTADOS DEL CLUSTERING:")
        logger.info(f"  🎯 Algoritmo: {self.algorithm}")
        
        if self.metrics:
            if 'silhouette_score' in self.metrics:
                logger.info(f"  📈 Silhouette Score: {self.metrics['silhouette_score']:.3f}")
            if 'calinski_harabasz_score' in self.metrics:
                logger.info(f"  📊 Calinski-Harabasz: {self.metrics['calinski_harabasz_score']:.2f}")
            if 'davies_bouldin_score' in self.metrics:
                logger.info(f"  📉 Davies-Bouldin: {self.metrics['davies_bouldin_score']:.3f}")
        
        unique_labels = np.unique(self.labels_)
        n_outliers = np.sum(self.labels_ == -1) if -1 in unique_labels else 0
        
        logger.info(f"  🏷️ Clusters encontrados: {len(unique_labels)}")
        logger.info(f"  ⚠️ Outliers: {n_outliers}")
        
        # Tamaños de clusters
        for label in unique_labels:
            if label != -1:
                size = np.sum(self.labels_ == label)
                percentage = size / len(self.labels_) * 100
                logger.info(f"    Cluster {label}: {size} candidatos ({percentage:.1f}%)")
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        model_data = {
            'algorithm': self.algorithm,
            'model': self.model,
            'n_clusters': self.n_clusters,
            'labels_': self.labels_,
            'cluster_centers_': self.cluster_centers_,
            'metrics': self.metrics,
            'pca': self.pca,
            'is_fitted': self.is_fitted,
            'algorithm_configs': self.algorithm_configs,
            'training_date': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Modelo guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Carga un modelo entrenado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(algorithm=model_data['algorithm'])
        model.model = model_data['model']
        model.n_clusters = model_data['n_clusters']
        model.labels_ = model_data['labels_']
        model.cluster_centers_ = model_data['cluster_centers_']
        model.metrics = model_data['metrics']
        model.pca = model_data['pca']
        model.is_fitted = model_data['is_fitted']
        model.algorithm_configs = model_data['algorithm_configs']
        
        logger.info(f"📂 Modelo cargado desde: {filepath}")
        logger.info(f"🗓️ Entrenado el: {model_data.get('training_date', 'fecha desconocida')}")
        
        return model

if __name__ == "__main__":
    # Ejemplo de uso
    print("🎯 CandidatesClusteringModel creado")
    print("✅ Algoritmos disponibles: kmeans, dbscan, hierarchical")
    print("🚀 Listo para entrenar clustering de candidatos")