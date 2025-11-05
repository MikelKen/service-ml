#!/usr/bin/env python3
"""
🎓 ENTRENAMIENTO PASO A PASO - CLUSTERING DE CANDIDATOS
Entrena modelo de clustering no supervisado y guarda archivos .pkl
"""

import asyncio
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.mongodb_connection import get_collection
from app.ml.preprocessing.candidates_clustering_preprocessor import CandidatesClusteringPreprocessor
from app.ml.models.candidates_clustering_model import CandidatesClusteringModel

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CandidatesClusteringTrainer:
    """Entrenador completo para clustering de candidatos"""
    
    def __init__(self):
        self.models_dir = "trained_models/clustering"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Nombres de archivos con identificador clustering
        self.files = {
            'preprocessor': f"{self.models_dir}/candidates_clustering_preprocessor_{self.timestamp}.pkl",
            'kmeans_model': f"{self.models_dir}/candidates_clustering_kmeans_{self.timestamp}.pkl",
            'dbscan_model': f"{self.models_dir}/candidates_clustering_dbscan_{self.timestamp}.pkl",
            'data_processed': f"{self.models_dir}/candidates_clustering_data_{self.timestamp}.pkl"
        }
        
        # Crear directorio
        os.makedirs(self.models_dir, exist_ok=True)
    
    async def step_1_load_data(self) -> pd.DataFrame:
        """PASO 1: Cargar datos de MongoDB"""
        logger.info("🔥 PASO 1: CARGANDO DATOS DE MONGODB")
        logger.info("="*60)
        
        try:
            collection = await get_collection("candidates_features")
            
            logger.info("📊 Obteniendo datos de candidatos...")
            cursor = collection.find({})
            data = await cursor.to_list(length=None)
            
            if not data:
                raise ValueError("No se encontraron datos en candidates_features")
            
            # Convertir a DataFrame
            df = pd.DataFrame(data)
            
            logger.info(f"✅ Datos cargados: {len(df)} candidatos")
            logger.info(f"📋 Columnas: {list(df.columns)}")
            
            # Estadísticas básicas
            logger.info("\n📈 ESTADÍSTICAS BÁSICAS:")
            logger.info(f"  • Total candidatos: {len(df)}")
            logger.info(f"  • Experiencia promedio: {df['anios_experiencia'].mean():.1f} años")
            logger.info(f"  • Niveles educación únicos: {df['nivel_educacion'].nunique()}")
            logger.info(f"  • Candidatos con certificaciones: {df['certificaciones'].notna().sum()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            raise
    
    def step_2_preprocess_data(self, df: pd.DataFrame) -> tuple:
        """PASO 2: Preprocessar datos para clustering"""
        logger.info("\n🔧 PASO 2: PREPROCESSING DE DATOS")
        logger.info("="*60)
        
        try:
            # Crear preprocessor
            preprocessor = CandidatesClusteringPreprocessor()
            
            # Preprocessar datos
            logger.info("🔄 Iniciando preprocessing...")
            X_processed = preprocessor.fit_transform(df)
            
            logger.info(f"✅ Preprocessing completado")
            logger.info(f"📊 Matriz procesada: {X_processed.shape}")
            logger.info(f"🏷️ Features extraídas: {len(preprocessor.feature_names)}")
            
            # Guardar preprocessor
            preprocessor.save_preprocessor(self.files['preprocessor'])
            
            # Guardar datos procesados
            processed_data = {
                'X_processed': X_processed,
                'feature_names': preprocessor.feature_names,
                'original_data': df,
                'preprocessing_date': datetime.now().isoformat()
            }
            
            import pickle
            with open(self.files['data_processed'], 'wb') as f:
                pickle.dump(processed_data, f)
            
            logger.info(f"💾 Datos procesados guardados en: {self.files['data_processed']}")
            
            return X_processed, preprocessor
            
        except Exception as e:
            logger.error(f"❌ Error en preprocessing: {e}")
            raise
    
    def step_3_train_kmeans(self, X: np.ndarray, feature_names: list) -> CandidatesClusteringModel:
        """PASO 3: Entrenar modelo K-Means"""
        logger.info("\n🎯 PASO 3: ENTRENAMIENTO K-MEANS")
        logger.info("="*60)
        
        try:
            # Crear modelo K-Means
            kmeans_model = CandidatesClusteringModel(algorithm='kmeans')
            
            # Entrenar con búsqueda de clusters óptimos
            logger.info("🔍 Buscando número óptimo de clusters...")
            kmeans_model.fit(X, find_optimal=True)
            
            # Obtener perfiles de clusters
            logger.info("📊 Generando perfiles de clusters...")
            cluster_profiles = kmeans_model.get_cluster_profiles(X, feature_names)
            
            # Mostrar resultados
            logger.info("\n📈 RESULTADOS K-MEANS:")
            summary = kmeans_model.get_cluster_summary()
            logger.info(f"  🎯 Clusters encontrados: {summary['n_clusters_found']}")
            logger.info(f"  📊 Silhouette Score: {summary['metrics'].get('silhouette_score', 'N/A')}")
            
            for cluster_id, profile in cluster_profiles.items():
                logger.info(f"  🏷️ Cluster {cluster_id}: {profile['size']} candidatos ({profile['percentage']:.1f}%)")
            
            # Guardar modelo
            kmeans_model.save_model(self.files['kmeans_model'])
            
            return kmeans_model
            
        except Exception as e:
            logger.error(f"❌ Error entrenando K-Means: {e}")
            raise
    
    def step_4_train_dbscan(self, X: np.ndarray, feature_names: list) -> CandidatesClusteringModel:
        """PASO 4: Entrenar modelo DBSCAN"""
        logger.info("\n🌐 PASO 4: ENTRENAMIENTO DBSCAN")
        logger.info("="*60)
        
        try:
            # Crear modelo DBSCAN
            dbscan_model = CandidatesClusteringModel(algorithm='dbscan')
            
            # Entrenar
            logger.info("🔄 Entrenando DBSCAN...")
            dbscan_model.fit(X, find_optimal=False)
            
            # Obtener perfiles
            cluster_profiles = dbscan_model.get_cluster_profiles(X, feature_names)
            
            # Mostrar resultados
            logger.info("\n📈 RESULTADOS DBSCAN:")
            summary = dbscan_model.get_cluster_summary()
            logger.info(f"  🎯 Clusters encontrados: {summary['n_clusters_found']}")
            logger.info(f"  ⚠️ Outliers: {summary['n_outliers']}")
            
            for cluster_id, profile in cluster_profiles.items():
                logger.info(f"  🏷️ Cluster {cluster_id}: {profile['size']} candidatos ({profile['percentage']:.1f}%)")
            
            # Guardar modelo
            dbscan_model.save_model(self.files['dbscan_model'])
            
            return dbscan_model
            
        except Exception as e:
            logger.error(f"❌ Error entrenando DBSCAN: {e}")
            raise
    
    def step_5_generate_summary(self, kmeans_model: CandidatesClusteringModel, 
                               dbscan_model: CandidatesClusteringModel,
                               df: pd.DataFrame):
        """PASO 5: Generar resumen final"""
        logger.info("\n📊 PASO 5: RESUMEN FINAL")
        logger.info("="*60)
        
        # Resumen de entrenamiento
        summary = {
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(df),
            'models_trained': ['kmeans', 'dbscan'],
            'files_generated': self.files,
            'kmeans_results': kmeans_model.get_cluster_summary(),
            'dbscan_results': dbscan_model.get_cluster_summary()
        }
        
        # Guardar resumen
        summary_file = f"{self.models_dir}/training_summary_{self.timestamp}.json"
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("\n📁 ARCHIVOS GENERADOS:")
        for name, filepath in self.files.items():
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"  ✅ {name}: {filepath} ({size_mb:.1f} MB)")
        
        logger.info(f"\n📊 RESUMEN:")
        logger.info(f"  📈 Dataset: {len(df)} candidatos procesados")
        logger.info(f"  🎯 K-Means: {summary['kmeans_results']['n_clusters_found']} clusters")
        logger.info(f"  🌐 DBSCAN: {summary['dbscan_results']['n_clusters_found']} clusters")
        logger.info(f"  💾 Archivos: {len([f for f in self.files.values() if os.path.exists(f)])} generados")
        
        return summary
    
    async def train_complete_pipeline(self):
        """Ejecuta el pipeline completo de entrenamiento"""
        logger.info("🚀 INICIANDO ENTRENAMIENTO COMPLETO DE CLUSTERING")
        logger.info("🎯 Objetivo: Agrupar candidatos por similitud de perfil")
        logger.info("⏱️ Timestamp: " + self.timestamp)
        logger.info("="*80)
        
        try:
            # PASO 1: Cargar datos
            df = await self.step_1_load_data()
            
            # PASO 2: Preprocessar
            X_processed, preprocessor = self.step_2_preprocess_data(df)
            
            # PASO 3: Entrenar K-Means
            kmeans_model = self.step_3_train_kmeans(X_processed, preprocessor.feature_names)
            
            # PASO 4: Entrenar DBSCAN
            dbscan_model = self.step_4_train_dbscan(X_processed, preprocessor.feature_names)
            
            # PASO 5: Resumen final
            summary = self.step_5_generate_summary(kmeans_model, dbscan_model, df)
            
            logger.info("\n🎉 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
            logger.info("🚀 Siguiente paso: Implementar GraphQL para consultas")
            
            return summary
            
        except Exception as e:
            logger.error(f"💥 Error en el entrenamiento: {e}")
            raise

async def main():
    """Función principal"""
    trainer = CandidatesClusteringTrainer()
    await trainer.train_complete_pipeline()

if __name__ == "__main__":
    asyncio.run(main())