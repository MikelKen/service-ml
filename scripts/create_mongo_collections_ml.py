#!/usr/bin/env python3
"""
🗃️ DISEÑO DE COLECCIONES MONGODB PARA ML SEMI-SUPERVISADO
Define esquemas y estructura de datos optimizada para machine learning
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import sys

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.mongodb_connection import get_mongodb_sync, mongodb_connection

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoMLCollectionDesigner:
    """Diseñador de colecciones MongoDB para ML"""
    
    def __init__(self):
        self.db = None
        self.collections_schema = self._define_collections_schema()
    
    def _define_collections_schema(self) -> Dict[str, Dict]:
        """Define el esquema de las colecciones MongoDB para ML"""
        
        schemas = {
            # Colección principal de candidatos para ML
            "ml_candidates": {
                "description": "Datos de candidatos procesados para machine learning",
                "indexes": [
                    {"fields": [("candidate_id", 1)], "unique": True},
                    {"fields": [("email", 1)], "unique": True},
                    {"fields": [("created_at", -1)]},
                    {"fields": [("ml_label", 1)]},
                    {"fields": [("anos_experiencia", 1)]},
                    {"fields": [("nivel_educacion", 1)]},
                ],
                "schema": {
                    "candidate_id": "str",  # UUID del candidato
                    "nombre": "str",
                    "email": "str",
                    "telefono": "str",
                    "anos_experiencia": "int",
                    "nivel_educacion": "str",
                    "habilidades": ["str"],  # Lista de habilidades
                    "habilidades_raw": "str",  # Texto original
                    "idiomas": ["str"],  # Lista de idiomas
                    "idiomas_raw": "str",  # Texto original
                    "certificaciones": ["str"],  # Lista de certificaciones
                    "certificaciones_raw": "str",  # Texto original
                    "puesto_actual": "str",
                    "url_cv": "str",
                    
                    # Features procesadas para ML
                    "features": {
                        "experiencia_normalizada": "float",
                        "nivel_educacion_encoded": "int",
                        "num_habilidades": "int",
                        "num_idiomas": "int",
                        "num_certificaciones": "int",
                        "skills_vector": ["float"],  # Vector TF-IDF de habilidades
                        "profile_completeness": "float",  # Completitud del perfil (0-1)
                    },
                    
                    # Labels para aprendizaje semi-supervisado
                    "ml_label": "str",  # 'hired', 'rejected', 'unlabeled'
                    "ml_confidence": "float",  # Confianza en la etiqueta (0-1)
                    "label_source": "str",  # 'manual', 'model_prediction', 'rule_based'
                    
                    # Metadatos
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "version": "int"
                }
            },
            
            # Colección de ofertas de trabajo para ML
            "ml_job_offers": {
                "description": "Ofertas de trabajo procesadas para machine learning",
                "indexes": [
                    {"fields": [("offer_id", 1)], "unique": True},
                    {"fields": [("empresa_id", 1)]},
                    {"fields": [("fecha_publicacion", -1)]},
                    {"fields": [("activa", 1)]},
                    {"fields": [("salario", 1)]},
                ],
                "schema": {
                    "offer_id": "str",  # UUID de la oferta
                    "titulo": "str",
                    "descripcion": "str",
                    "salario": "float",
                    "ubicacion": "str",
                    "requisitos": "str",
                    "fecha_publicacion": "datetime",
                    "activa": "bool",
                    
                    # Información de la empresa
                    "empresa_id": "str",
                    "empresa_nombre": "str",
                    "empresa_rubro": "str",
                    
                    # Features procesadas para ML
                    "features": {
                        "requisitos_vector": ["float"],  # Vector TF-IDF de requisitos
                        "salario_normalizado": "float",
                        "dias_desde_publicacion": "int",
                        "nivel_requisitos": "str",  # 'junior', 'mid', 'senior'
                        "tipo_contrato": "str",  # Inferido del texto
                        "modalidad_trabajo": "str",  # 'presencial', 'remoto', 'hibrido'
                    },
                    
                    # Metadatos
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "version": "int"
                }
            },
            
            # Colección de postulaciones para ML
            "ml_applications": {
                "description": "Postulaciones con features para machine learning",
                "indexes": [
                    {"fields": [("application_id", 1)], "unique": True},
                    {"fields": [("candidate_id", 1)]},
                    {"fields": [("offer_id", 1)]},
                    {"fields": [("estado", 1)]},
                    {"fields": [("fecha_postulacion", -1)]},
                    {"fields": [("ml_prediction", 1)]},
                ],
                "schema": {
                    "application_id": "str",  # UUID de la postulación
                    "candidate_id": "str",  # Referencia a ml_candidates
                    "offer_id": "str",  # Referencia a ml_job_offers
                    "fecha_postulacion": "datetime",
                    "estado": "str",  # Estado original
                    
                    # Features de compatibilidad
                    "compatibility_features": {
                        "skill_match_score": "float",  # Similitud de habilidades (0-1)
                        "experience_match": "float",  # Compatibilidad de experiencia
                        "education_match": "float",  # Compatibilidad educativa
                        "location_match": "bool",  # Coincidencia de ubicación
                        "salary_expectation_match": "float",  # Expectativa salarial
                        "overall_compatibility": "float",  # Score general de compatibilidad
                    },
                    
                    # Labels y predicciones ML
                    "ml_target": "int",  # 0: rechazado, 1: aceptado/contratado
                    "ml_prediction": "int",  # Predicción del modelo
                    "ml_probability": "float",  # Probabilidad de contratación
                    "ml_confidence": "str",  # 'high', 'medium', 'low'
                    
                    # Semi-supervised learning
                    "is_labeled": "bool",  # True si tiene etiqueta conocida
                    "label_quality": "str",  # 'high', 'medium', 'low', 'predicted'
                    
                    # Metadatos
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "processed_at": "datetime",
                    "version": "int"
                }
            },
            
            # Colección de entrevistas y evaluaciones
            "ml_interviews": {
                "description": "Datos de entrevistas para enriquecer el modelo",
                "indexes": [
                    {"fields": [("interview_id", 1)], "unique": True},
                    {"fields": [("application_id", 1)]},
                    {"fields": [("fecha", -1)]},
                ],
                "schema": {
                    "interview_id": "str",
                    "application_id": "str",  # Referencia a ml_applications
                    "fecha": "datetime",
                    "duracion_min": "int",
                    "objetivos_totales": "str",
                    "objetivos_cubiertos": "str",
                    "entrevistador": "str",
                    
                    # Evaluaciones
                    "evaluaciones": [{
                        "evaluacion_id": "str",
                        "calificacion_tecnica": "float",
                        "calificacion_actitud": "float",
                        "calificacion_general": "float",
                        "comentarios": "str"
                    }],
                    
                    # Features derivadas
                    "features": {
                        "promedio_calificaciones": "float",
                        "cobertura_objetivos": "float",  # % objetivos cubiertos
                        "duracion_normalizada": "float",
                        "sentiment_comentarios": "float",  # Análisis de sentimiento
                    },
                    
                    # Metadatos
                    "created_at": "datetime",
                    "updated_at": "datetime"
                }
            },
            
            # Colección para tracking del modelo ML
            "ml_model_tracking": {
                "description": "Tracking de entrenamientos y métricas del modelo",
                "indexes": [
                    {"fields": [("model_id", 1)], "unique": True},
                    {"fields": [("created_at", -1)]},
                    {"fields": [("model_type", 1)]},
                    {"fields": [("is_active", 1)]},
                ],
                "schema": {
                    "model_id": "str",
                    "model_name": "str",
                    "model_type": "str",  # 'semi_supervised', 'supervised', 'clustering'
                    "algorithm": "str",  # 'label_propagation', 'self_training', etc.
                    "version": "str",
                    "is_active": "bool",
                    
                    # Configuración del entrenamiento
                    "training_config": {
                        "labeled_samples": "int",
                        "unlabeled_samples": "int",
                        "test_samples": "int",
                        "features_used": ["str"],
                        "hyperparameters": "dict"
                    },
                    
                    # Métricas del modelo
                    "metrics": {
                        "accuracy": "float",
                        "precision": "float",
                        "recall": "float",
                        "f1_score": "float",
                        "roc_auc": "float",
                        "pseudo_label_accuracy": "float",  # Para semi-supervisado
                        "label_propagation_confidence": "float"
                    },
                    
                    # Información del dataset
                    "dataset_info": {
                        "total_samples": "int",
                        "labeled_ratio": "float",
                        "class_distribution": "dict",
                        "feature_importance": "dict"
                    },
                    
                    # Metadatos
                    "trained_at": "datetime",
                    "created_at": "datetime",
                    "model_path": "str",
                    "preprocessor_path": "str"
                }
            }
        }
        
        return schemas
    
    def connect(self):
        """Conectar a MongoDB"""
        try:
            mongodb_connection.connect_sync()
            self.db = get_mongodb_sync()
            logger.info("✅ Conectado a MongoDB")
        except Exception as e:
            logger.error(f"❌ Error conectando a MongoDB: {e}")
            raise
    
    def disconnect(self):
        """Desconectar de MongoDB"""
        try:
            mongodb_connection.disconnect_sync()
            logger.info("🔌 Desconectado de MongoDB")
        except Exception as e:
            logger.error(f"❌ Error desconectando: {e}")
    
    def create_collections(self):
        """Crear todas las colecciones con sus índices"""
        logger.info("🗃️ Creando colecciones MongoDB...")
        
        for collection_name, config in self.collections_schema.items():
            try:
                # Crear colección si no existe
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
                    logger.info(f"  ✅ Colección creada: {collection_name}")
                else:
                    logger.info(f"  ℹ️ Colección ya existe: {collection_name}")
                
                # Crear índices
                collection = self.db[collection_name]
                
                # Eliminar índices existentes (excepto _id)
                existing_indexes = list(collection.list_indexes())
                for index in existing_indexes:
                    if index['name'] != '_id_':
                        collection.drop_index(index['name'])
                
                # Crear nuevos índices
                for index_config in config['indexes']:
                    fields = index_config['fields']
                    options = {k: v for k, v in index_config.items() if k != 'fields'}
                    
                    try:
                        collection.create_index(fields, **options)
                        logger.info(f"    📊 Índice creado en {collection_name}: {fields}")
                    except Exception as e:
                        logger.warning(f"    ⚠️ Error creando índice {fields}: {e}")
                
            except Exception as e:
                logger.error(f"❌ Error creando colección {collection_name}: {e}")
    
    def validate_collections(self) -> Dict[str, bool]:
        """Validar que todas las colecciones existan"""
        logger.info("🔍 Validando colecciones...")
        
        results = {}
        existing_collections = self.db.list_collection_names()
        
        for collection_name in self.collections_schema.keys():
            exists = collection_name in existing_collections
            results[collection_name] = exists
            
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {collection_name}: {'Existe' if exists else 'No existe'}")
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Obtener información de las colecciones"""
        logger.info("📊 Obteniendo información de colecciones...")
        
        info = {}
        
        for collection_name in self.collections_schema.keys():
            try:
                collection = self.db[collection_name]
                
                # Estadísticas básicas
                stats = self.db.command("collStats", collection_name)
                
                info[collection_name] = {
                    "count": stats.get("count", 0),
                    "size_bytes": stats.get("size", 0),
                    "avg_obj_size": stats.get("avgObjSize", 0),
                    "indexes": len(list(collection.list_indexes())),
                    "description": self.collections_schema[collection_name]["description"]
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Error obteniendo info de {collection_name}: {e}")
                info[collection_name] = {"error": str(e)}
        
        return info
    
    def print_schema_documentation(self):
        """Imprimir documentación de los esquemas"""
        logger.info("📋 DOCUMENTACIÓN DE ESQUEMAS MONGODB:")
        logger.info("=" * 60)
        
        for collection_name, config in self.collections_schema.items():
            logger.info(f"\n🗃️ Colección: {collection_name}")
            logger.info(f"   Descripción: {config['description']}")
            logger.info(f"   Índices: {len(config['indexes'])}")
            logger.info(f"   Campos principales:")
            
            # Mostrar campos principales del esquema
            schema = config['schema']
            for field, field_type in list(schema.items())[:10]:  # Primeros 10 campos
                if isinstance(field_type, dict):
                    logger.info(f"     - {field}: objeto con {len(field_type)} subcampos")
                elif isinstance(field_type, list):
                    logger.info(f"     - {field}: array de {field_type[0] if field_type else 'any'}")
                else:
                    logger.info(f"     - {field}: {field_type}")
            
            if len(schema) > 10:
                logger.info(f"     ... y {len(schema) - 10} campos más")


def main():
    """Función principal"""
    designer = MongoMLCollectionDesigner()
    
    try:
        # Conectar
        designer.connect()
        
        # Imprimir documentación
        designer.print_schema_documentation()
        
        # Crear colecciones
        designer.create_collections()
        
        # Validar
        validation_results = designer.validate_collections()
        
        # Mostrar información
        collection_info = designer.get_collection_info()
        
        logger.info("\n📊 RESUMEN DE COLECCIONES:")
        for name, info in collection_info.items():
            if 'error' not in info:
                logger.info(f"  {name}: {info['count']} documentos, {info['indexes']} índices")
        
        logger.info("🎉 Diseño de colecciones MongoDB completado")
        
    finally:
        designer.disconnect()


if __name__ == "__main__":
    main()