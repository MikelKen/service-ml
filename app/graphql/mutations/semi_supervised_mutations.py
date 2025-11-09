#!/usr/bin/env python3
"""
🔧 MUTACIONES GRAPHQL PARA MODELO SEMI-SUPERVISADO
Implementa mutaciones para entrenar modelos, realizar predicciones y gestionar el pipeline ML
"""

import strawberry
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import os
import uuid
import json

from app.graphql.types.semi_supervised_types import (
    ModelTrainingResult, BatchPredictionResult, OperationResult,
    SemiSupervisedPrediction, ModelInfo,
    # Input types
    TrainingParameters, PredictionInput, BatchPredictionInput,
    # Enums
    SemiSupervisedAlgorithm, ConfidenceLevel, PredictionStatus
)

from app.graphql.resolvers.semi_supervised_resolvers import semi_supervised_resolvers
from app.ml.models.semi_supervised_model import SemiSupervisedClassifier, train_semi_supervised_model
from app.ml.preprocessing.semi_supervised_preprocessor import semi_supervised_preprocessor
from app.config.mongodb_connection import get_mongodb_sync, mongodb_connection
import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@strawberry.type
class SemiSupervisedMLMutations:
    """Mutaciones para operaciones de machine learning semi-supervisado"""
    
    @strawberry.mutation
    async def train_semi_supervised_model(
        self,
        parameters: TrainingParameters
    ) -> ModelTrainingResult:
        """Entrenar un nuevo modelo semi-supervisado"""
        logger.info(f"🚀 Iniciando entrenamiento de modelo: {parameters.algorithm.value}")
        
        training_id = f"training_{uuid.uuid4().hex[:8]}"
        training_started = datetime.now(timezone.utc)
        
        try:
            # Configurar parámetros de entrenamiento
            algorithm = parameters.algorithm.value
            
            # Entrenar modelo
            training_record = train_semi_supervised_model(
                algorithm=algorithm,
                save_path=f"trained_models/semi_supervised/{algorithm}_model.pkl"
            )
            
            training_completed = datetime.now(timezone.utc)
            training_time = (training_completed - training_started).total_seconds()
            
            # Guardar registro en MongoDB
            await self._save_training_record(training_id, algorithm, training_record, training_started, training_completed)
            
            # Convertir métricas
            from app.graphql.types.semi_supervised_types import ModelMetrics, PseudoLabelStats, TrainingConfig
            
            metrics_data = training_record.get('metrics', {})
            metrics = ModelMetrics(
                train_accuracy=metrics_data.get('train_accuracy', 0.0),
                train_precision=metrics_data.get('train_precision', 0.0),
                train_recall=metrics_data.get('train_recall', 0.0),
                train_f1=metrics_data.get('train_f1', 0.0),
                val_accuracy=metrics_data.get('val_accuracy'),
                val_precision=metrics_data.get('val_precision'),
                val_recall=metrics_data.get('val_recall'),
                val_f1=metrics_data.get('val_f1'),
                val_roc_auc=metrics_data.get('val_roc_auc'),
                cv_f1_mean=metrics_data.get('cv_f1_mean'),
                cv_f1_std=metrics_data.get('cv_f1_std')
            )
            
            # Estadísticas de pseudo-etiquetas
            pseudo_stats_data = training_record.get('pseudo_label_stats', {})
            pseudo_label_stats = PseudoLabelStats(
                total_unlabeled=pseudo_stats_data.get('total_unlabeled', 0),
                positive_pseudo_labels=pseudo_stats_data.get('pseudo_label_distribution', {}).get('positive', 0),
                negative_pseudo_labels=pseudo_stats_data.get('pseudo_label_distribution', {}).get('negative', 0),
                mean_confidence=pseudo_stats_data.get('confidence_stats', {}).get('mean_confidence'),
                median_confidence=pseudo_stats_data.get('confidence_stats', {}).get('median_confidence'),
                high_confidence_samples=pseudo_stats_data.get('confidence_stats', {}).get('high_confidence_samples'),
                low_confidence_samples=pseudo_stats_data.get('confidence_stats', {}).get('low_confidence_samples')
            )
            
            # Configuración de entrenamiento
            training_config = TrainingConfig(
                algorithm=parameters.algorithm,
                labeled_samples=training_record.get('labeled_samples', 0),
                unlabeled_samples=training_record.get('unlabeled_samples', 0),
                validation_split=parameters.validation_split,
                features_used=[],  # Se podría obtener del preprocessor
                hyperparameters={}  # Se podría obtener de la configuración del algoritmo
            )
            
            result = ModelTrainingResult(
                training_id=training_id,
                algorithm=parameters.algorithm,
                training_started_at=training_started,
                training_completed_at=training_completed,
                training_time_seconds=training_time,
                training_config=training_config,
                metrics=metrics,
                pseudo_label_stats=pseudo_label_stats,
                success=True,
                model_path=f"trained_models/semi_supervised/{algorithm}_model.pkl",
                preprocessor_path="trained_models/semi_supervised_preprocessor.pkl"
            )
            
            logger.info(f"✅ Entrenamiento completado exitosamente: {training_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            
            return ModelTrainingResult(
                training_id=training_id,
                algorithm=parameters.algorithm,
                training_started_at=training_started,
                training_completed_at=datetime.now(timezone.utc),
                training_time_seconds=0,
                training_config=TrainingConfig(
                    algorithm=parameters.algorithm,
                    labeled_samples=0,
                    unlabeled_samples=0,
                    validation_split=parameters.validation_split,
                    features_used=[],
                    hyperparameters={}
                ),
                metrics=ModelMetrics(
                    train_accuracy=0.0,
                    train_precision=0.0,
                    train_recall=0.0,
                    train_f1=0.0
                ),
                pseudo_label_stats=PseudoLabelStats(
                    total_unlabeled=0,
                    positive_pseudo_labels=0,
                    negative_pseudo_labels=0
                ),
                success=False,
                error_message=str(e)
            )
    
    @strawberry.mutation
    async def predict_batch_applications(
        self,
        batch_input: BatchPredictionInput
    ) -> BatchPredictionResult:
        """Realizar predicciones en lote para múltiples aplicaciones"""
        logger.info(f"🔮 Iniciando predicciones en lote: {len(batch_input.application_ids)} aplicaciones")
        
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        processed_at = datetime.now(timezone.utc)
        
        try:
            # Obtener datos de aplicaciones
            mongodb_connection.connect_sync()
            db = get_mongodb_sync()
            
            applications_collection = db['ml_applications']
            candidates_collection = db['ml_candidates']
            offers_collection = db['ml_job_offers']
            
            # Buscar aplicaciones
            applications_cursor = applications_collection.find({
                'application_id': {'$in': batch_input.application_ids}
            })
            applications_data = list(applications_cursor)
            
            if not applications_data:
                raise ValueError("No se encontraron aplicaciones con los IDs proporcionados")
            
            # Obtener candidatos y ofertas relacionadas
            candidate_ids = [app['candidate_id'] for app in applications_data]
            offer_ids = [app['offer_id'] for app in applications_data]
            
            candidates_cursor = candidates_collection.find({'candidate_id': {'$in': candidate_ids}})
            candidates_data = list(candidates_cursor)
            
            offers_cursor = offers_collection.find({'offer_id': {'$in': offer_ids}})
            offers_data = list(offers_cursor)
            
            # Crear DataFrames
            applications_df = pd.DataFrame(applications_data)
            candidates_df = pd.DataFrame(candidates_data)
            offers_df = pd.DataFrame(offers_data)
            
            # Cargar modelo
            model_path = "trained_models/semi_supervised/label_propagation_model.pkl"
            if not os.path.exists(model_path):
                raise ValueError("No hay modelo entrenado disponible")
            
            classifier = SemiSupervisedClassifier.load_model(model_path)
            
            # Transformar datos
            X = semi_supervised_preprocessor.transform(applications_df, candidates_df, offers_df)
            
            # Realizar predicciones
            predictions, probabilities, confidence_levels = classifier.predict_with_confidence(X)
            
            # Procesar resultados
            batch_predictions = []
            successful_predictions = 0
            failed_predictions = 0
            errors = []
            
            # Contadores para estadísticas
            high_confidence_count = 0
            medium_confidence_count = 0
            low_confidence_count = 0
            predicted_positive = 0
            predicted_negative = 0
            
            for i, app_data in enumerate(applications_data):
                try:
                    prediction = predictions[i]
                    probability = probabilities[i]
                    confidence = confidence_levels[i]
                    
                    # Convertir enums
                    prediction_status = PredictionStatus.ACCEPTED if prediction == 1 else PredictionStatus.REJECTED
                    confidence_level = ConfidenceLevel(confidence)
                    
                    # Actualizar contadores
                    if confidence == 'high':
                        high_confidence_count += 1
                    elif confidence == 'medium':
                        medium_confidence_count += 1
                    else:
                        low_confidence_count += 1
                    
                    if prediction == 1:
                        predicted_positive += 1
                    else:
                        predicted_negative += 1
                    
                    batch_prediction = SemiSupervisedPrediction(
                        application_id=app_data['application_id'],
                        candidate_id=app_data['candidate_id'],
                        offer_id=app_data['offer_id'],
                        prediction=prediction_status,
                        probability=float(probability),
                        confidence_level=confidence_level,
                        compatibility_score=app_data.get('compatibility_features', {}).get('overall_compatibility', 0.0),
                        predicted_at=processed_at,
                        model_algorithm=SemiSupervisedAlgorithm(classifier.algorithm),
                        model_version=classifier.model_info.get('version', '1.0')
                    )
                    
                    batch_predictions.append(batch_prediction)
                    successful_predictions += 1
                    
                    # Actualizar base de datos si se solicita
                    if batch_input.update_database:
                        applications_collection.update_one(
                            {'application_id': app_data['application_id']},
                            {
                                '$set': {
                                    'ml_prediction': int(prediction),
                                    'ml_probability': float(probability),
                                    'ml_confidence': confidence,
                                    'updated_at': processed_at,
                                    'model_algorithm': classifier.algorithm,
                                    'model_version': classifier.model_info.get('version', '1.0')
                                }
                            }
                        )
                    
                except Exception as e:
                    failed_predictions += 1
                    errors.append(f"Error procesando aplicación {app_data.get('application_id', 'unknown')}: {str(e)}")
            
            result = BatchPredictionResult(
                batch_id=batch_id,
                processed_at=processed_at,
                total_predictions=len(batch_input.application_ids),
                successful_predictions=successful_predictions,
                failed_predictions=failed_predictions,
                high_confidence_predictions=high_confidence_count,
                medium_confidence_predictions=medium_confidence_count,
                low_confidence_predictions=low_confidence_count,
                predicted_positive=predicted_positive,
                predicted_negative=predicted_negative,
                predictions=batch_predictions,
                errors=errors
            )
            
            logger.info(f"✅ Predicciones en lote completadas: {successful_predictions}/{len(batch_input.application_ids)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en predicciones en lote: {e}")
            
            return BatchPredictionResult(
                batch_id=batch_id,
                processed_at=processed_at,
                total_predictions=len(batch_input.application_ids),
                successful_predictions=0,
                failed_predictions=len(batch_input.application_ids),
                high_confidence_predictions=0,
                medium_confidence_predictions=0,
                low_confidence_predictions=0,
                predicted_positive=0,
                predicted_negative=0,
                predictions=[],
                errors=[str(e)]
            )
        
        finally:
            mongodb_connection.disconnect_sync()
    
    @strawberry.mutation
    async def retrain_model_with_new_labels(
        self,
        algorithm: SemiSupervisedAlgorithm,
        confidence_threshold: float = 0.8
    ) -> ModelTrainingResult:
        """Re-entrenar modelo incorporando predicciones de alta confianza como nuevas etiquetas"""
        logger.info(f"🔄 Re-entrenando modelo con nuevas etiquetas: {algorithm.value}")
        
        training_id = f"retrain_{uuid.uuid4().hex[:8]}"
        training_started = datetime.now(timezone.utc)
        
        try:
            # Obtener aplicaciones con predicciones de alta confianza
            mongodb_connection.connect_sync()
            db = get_mongodb_sync()
            applications_collection = db['ml_applications']
            
            # Actualizar aplicaciones con predicciones de alta confianza
            high_confidence_apps = applications_collection.find({
                'ml_confidence': 'high',
                'ml_probability': {'$gte': confidence_threshold},
                'is_labeled': False
            })
            
            updated_count = 0
            for app in high_confidence_apps:
                # Promover a etiquetado
                applications_collection.update_one(
                    {'_id': app['_id']},
                    {
                        '$set': {
                            'is_labeled': True,
                            'label_quality': 'predicted',
                            'ml_target': app.get('ml_prediction', -1),
                            'updated_at': datetime.now(timezone.utc)
                        }
                    }
                )
                updated_count += 1
            
            logger.info(f"📊 Promovidas {updated_count} predicciones a etiquetas")
            
            # Re-entrenar modelo
            training_record = train_semi_supervised_model(
                algorithm=algorithm.value,
                save_path=f"trained_models/semi_supervised/{algorithm.value}_retrained_model.pkl"
            )
            
            training_completed = datetime.now(timezone.utc)
            training_time = (training_completed - training_started).total_seconds()
            
            # Guardar registro
            await self._save_training_record(training_id, algorithm.value, training_record, training_started, training_completed)
            
            # Convertir resultado
            from app.graphql.types.semi_supervised_types import ModelMetrics, PseudoLabelStats, TrainingConfig
            
            metrics_data = training_record.get('metrics', {})
            metrics = ModelMetrics(
                train_accuracy=metrics_data.get('train_accuracy', 0.0),
                train_precision=metrics_data.get('train_precision', 0.0),
                train_recall=metrics_data.get('train_recall', 0.0),
                train_f1=metrics_data.get('train_f1', 0.0),
                val_f1=metrics_data.get('val_f1')
            )
            
            pseudo_stats_data = training_record.get('pseudo_label_stats', {})
            pseudo_label_stats = PseudoLabelStats(
                total_unlabeled=pseudo_stats_data.get('total_unlabeled', 0),
                positive_pseudo_labels=pseudo_stats_data.get('pseudo_label_distribution', {}).get('positive', 0),
                negative_pseudo_labels=pseudo_stats_data.get('pseudo_label_distribution', {}).get('negative', 0)
            )
            
            training_config = TrainingConfig(
                algorithm=algorithm,
                labeled_samples=training_record.get('labeled_samples', 0),
                unlabeled_samples=training_record.get('unlabeled_samples', 0),
                validation_split=0.2,
                features_used=[],
                hyperparameters={'confidence_threshold': confidence_threshold, 'promoted_labels': updated_count}
            )
            
            result = ModelTrainingResult(
                training_id=training_id,
                algorithm=algorithm,
                training_started_at=training_started,
                training_completed_at=training_completed,
                training_time_seconds=training_time,
                training_config=training_config,
                metrics=metrics,
                pseudo_label_stats=pseudo_label_stats,
                success=True,
                model_path=f"trained_models/semi_supervised/{algorithm.value}_retrained_model.pkl"
            )
            
            logger.info(f"✅ Re-entrenamiento completado exitosamente: {training_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en re-entrenamiento: {e}")
            raise
        
        finally:
            mongodb_connection.disconnect_sync()
    
    @strawberry.mutation
    async def activate_model(self, model_id: str) -> OperationResult:
        """Activar un modelo específico como el modelo en producción"""
        logger.info(f"🔄 Activando modelo: {model_id}")
        
        try:
            mongodb_connection.connect_sync()
            db = get_mongodb_sync()
            models_collection = db['ml_model_tracking']
            
            # Desactivar todos los modelos activos
            models_collection.update_many(
                {'is_active': True},
                {'$set': {'is_active': False, 'updated_at': datetime.now(timezone.utc)}}
            )
            
            # Activar el modelo seleccionado
            result = models_collection.update_one(
                {'model_id': model_id},
                {'$set': {'is_active': True, 'updated_at': datetime.now(timezone.utc)}}
            )
            
            if result.modified_count == 0:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            return OperationResult(
                success=True,
                message=f"Modelo {model_id} activado exitosamente",
                operation_id=f"activate_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"❌ Error activando modelo: {e}")
            return OperationResult(
                success=False,
                message=f"Error activando modelo: {str(e)}",
                timestamp=datetime.now(timezone.utc)
            )
        
        finally:
            mongodb_connection.disconnect_sync()
    
    @strawberry.mutation
    async def migrate_data_from_postgres(self) -> OperationResult:
        """Ejecutar migración de datos desde PostgreSQL a MongoDB"""
        logger.info("🔄 Iniciando migración de datos desde PostgreSQL...")
        
        try:
            # Importar y ejecutar migración
            from scripts.migrate_postgres_to_mongo_ml import PostgresToMongoMigrator
            
            migrator = PostgresToMongoMigrator()
            await migrator.migrate_data()
            
            return OperationResult(
                success=True,
                message="Migración de datos completada exitosamente",
                operation_id=f"migrate_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc),
                details={"source": "postgresql", "target": "mongodb"}
            )
            
        except Exception as e:
            logger.error(f"❌ Error en migración: {e}")
            return OperationResult(
                success=False,
                message=f"Error en migración: {str(e)}",
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _save_training_record(self, training_id: str, algorithm: str, 
                                  training_record: Dict, started_at: datetime, 
                                  completed_at: datetime):
        """Guardar registro de entrenamiento en MongoDB"""
        try:
            mongodb_connection.connect_sync()
            db = get_mongodb_sync()
            models_collection = db['ml_model_tracking']
            
            # Desactivar modelos anteriores del mismo algoritmo
            models_collection.update_many(
                {'algorithm': algorithm, 'is_active': True},
                {'$set': {'is_active': False}}
            )
            
            # Crear registro
            record = {
                'model_id': training_id,
                'model_name': f"SemiSupervised_{algorithm}",
                'model_type': 'semi_supervised',
                'algorithm': algorithm,
                'version': '1.0',
                'is_active': True,
                'training_config': {
                    'labeled_samples': training_record.get('labeled_samples', 0),
                    'unlabeled_samples': training_record.get('unlabeled_samples', 0),
                    'features_used': [],
                    'hyperparameters': {}
                },
                'metrics': training_record.get('metrics', {}),
                'dataset_info': {
                    'total_samples': training_record.get('labeled_samples', 0) + training_record.get('unlabeled_samples', 0),
                    'labeled_ratio': training_record.get('labeled_samples', 0) / max(training_record.get('labeled_samples', 0) + training_record.get('unlabeled_samples', 0), 1),
                    'feature_importance': {}
                },
                'trained_at': started_at,
                'created_at': completed_at,
                'model_path': f"trained_models/semi_supervised/{algorithm}_model.pkl",
                'preprocessor_path': "trained_models/semi_supervised_preprocessor.pkl"
            }
            
            models_collection.insert_one(record)
            logger.info(f"✅ Registro de entrenamiento guardado: {training_id}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando registro: {e}")
        finally:
            mongodb_connection.disconnect_sync()


# Instancia de mutaciones
semi_supervised_mutations = SemiSupervisedMLMutations()