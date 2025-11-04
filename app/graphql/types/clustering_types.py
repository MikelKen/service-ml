#!/usr/bin/env python3
"""
🧬 TIPOS GRAPHQL PARA CLUSTERING DE CANDIDATOS
Definiciones de tipos para consultas de clustering no supervisado
"""

import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime

@strawberry.type
class ClusterProfile:
    """Perfil descriptivo de un cluster de candidatos"""
    cluster_id: int
    size: int
    percentage: float
    description: str
    top_characteristics: List[str]
    
    @strawberry.field
    def summary(self) -> str:
        return f"Cluster {self.cluster_id}: {self.size} candidatos ({self.percentage:.1f}%)"

@strawberry.type
class ClusteringMetrics:
    """Métricas de calidad del clustering"""
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    n_clusters: int
    algorithm_used: str

@strawberry.type
class CandidateClusterAssignment:
    """Asignación de candidato a cluster"""
    candidate_id: str
    cluster_id: int
    cluster_confidence: Optional[float] = None
    distance_to_center: Optional[float] = None

@strawberry.type
class ClusterAnalysis:
    """Análisis completo de clustering"""
    total_candidates: int
    clusters_found: int
    outliers_detected: int
    cluster_profiles: List[ClusterProfile]
    metrics: ClusteringMetrics
    algorithm_used: str
    training_date: str

@strawberry.type
class SimilarCandidates:
    """Candidatos similares basados en clustering"""
    target_candidate_id: str
    target_cluster_id: int
    similar_candidates: List[CandidateClusterAssignment]
    similarity_criteria: List[str]

@strawberry.input
class ClusteringQueryInput:
    """Input para consultas de clustering"""
    algorithm: Optional[str] = "kmeans"  # kmeans, dbscan
    max_results: Optional[int] = 10
    include_outliers: Optional[bool] = False

@strawberry.input
class SimilarCandidatesInput:
    """Input para búsqueda de candidatos similares"""
    candidate_id: str
    max_similar: Optional[int] = 5
    algorithm: Optional[str] = "kmeans"
    include_metrics: Optional[bool] = True

@strawberry.input
class ClusterProfileInput:
    """Input para análisis de perfil de cluster"""
    cluster_id: int
    algorithm: Optional[str] = "kmeans"
    include_details: Optional[bool] = True