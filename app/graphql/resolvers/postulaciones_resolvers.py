"""
Resolvers GraphQL para el modelo semi-supervisado de postulaciones
"""
import strawberry
from typing import List, Optional, Dict, Any
import logging
import asyncio
import json
from datetime import datetime
import os

from app.graphql.types.postulaciones_types import (
    EstadoDistribution, PostulacionPrediction, PostulacionesModelMetrics, PostulacionesTrainingDataSummary,
    SemiSupervisedModelInfo, PostulacionPredictionResult, PredictionBatch,
    TrainingResult, PostulacionInput, PredictionFilter, TrainingConfig,
    PostulacionFeatures, PostulacionesDatasetStats
)
from app.ml.data.postgres_extractor import postgres_extractor
from app.ml.training.postulaciones_semi_supervised_trainer import postulaciones_trainer
from app.config.connection import mongodb
from app.config.settings import settings

logger = logging.getLogger(__name__)


@strawberry.type
class PostulacionesQuery:
    """Consultas para el modelo de postulaciones"""
    
    @strawberry.field
    async def get_estados_distribution(self) -> List[EstadoDistribution]:
        """Obtiene la distribución actual de estados de postulaciones"""
        try:
            stats_df = await postgres_extractor.get_estado_distribution()
            
            if stats_df is None or stats_df.empty:
                return []
            
            # Calcular total para porcentajes
            total_records = stats_df['cantidad'].sum()
            
            return [
                EstadoDistribution(
                    estado=row['estado'], 
                    cantidad=row['cantidad'],
                    percentage=round((row['cantidad'] / total_records) * 100, 2) if total_records > 0 else 0.0
                )
                for _, row in stats_df.iterrows()
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo distribución de estados: {e}")
            return []
    
    @strawberry.field
    async def get_model_info(self) -> Optional[SemiSupervisedModelInfo]:
        """Obtiene información del modelo actual"""
        try:
            # Verificar si hay modelo cargado
            if not postulaciones_trainer.is_trained or postulaciones_trainer.best_model is None:
                # Intentar cargar modelo guardado
                model_path = os.path.join(settings.ml_models_path, "postulaciones", "semi_supervised_model.pkl")
                if os.path.exists(model_path):
                    postulaciones_trainer.load_model(model_path)
                else:
                    return None
            
            # Crear métricas
            metrics = PostulacionesModelMetrics(
                accuracy=postulaciones_trainer.model_metrics.get('accuracy', 0.0),
                precision_macro=postulaciones_trainer.model_metrics.get('precision_macro', 0.0),
                recall_macro=postulaciones_trainer.model_metrics.get('recall_macro', 0.0),
                f1_macro=postulaciones_trainer.model_metrics.get('f1_macro', 0.0),
                precision_weighted=postulaciones_trainer.model_metrics.get('precision_weighted', 0.0),
                recall_weighted=postulaciones_trainer.model_metrics.get('recall_weighted', 0.0),
                f1_weighted=postulaciones_trainer.model_metrics.get('f1_weighted', 0.0)
            )
            
            # Obtener distribución de estados (para data_summary)
            estados_dist = await self.get_estados_distribution()
            
            # Crear resumen de datos
            data_summary = PostulacionesTrainingDataSummary(
                total_records=len(postulaciones_trainer.complete_data) if postulaciones_trainer.complete_data is not None else 0,
                labeled_records=len(postulaciones_trainer.labeled_data) if postulaciones_trainer.labeled_data is not None else 0,
                unlabeled_records=len(postulaciones_trainer.unlabeled_data) if postulaciones_trainer.unlabeled_data is not None else 0,
                features_count=len(postulaciones_trainer.processed_features.columns) - 4 if postulaciones_trainer.processed_features is not None else 0,
                estado_distribution=estados_dist
            )
            
            return SemiSupervisedModelInfo(
                model_name=postulaciones_trainer.best_model_name or "unknown",
                model_type="semi_supervised",
                training_date=datetime.now().isoformat(),  # Podríamos guardar esto en el modelo
                metrics=metrics,
                data_summary=data_summary,
                is_trained=postulaciones_trainer.is_trained
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo información del modelo: {e}")
            return None
    
    @strawberry.field
    async def postulaciones_dataset_stats(self) -> Optional[PostulacionesDatasetStats]:
        """Obtiene estadísticas del dataset de postulaciones"""
        try:
            # Obtener distribución de estados
            estados_dist = await self.get_estados_distribution()
            
            # Conectar a MongoDB si no está conectado
            if mongodb.database is None:
                await mongodb.connect()
            
            # Obtener estadísticas de la colección de datos completos
            complete_collection = mongodb.get_collection("postulaciones_completas")
            labeled_collection = mongodb.get_collection("postulaciones_labeled")
            
            # Contar registros
            total_records = await complete_collection.count_documents({})
            labeled_records = await labeled_collection.count_documents({})
            unlabeled_records = total_records - labeled_records
            
            # Obtener features count (asumiendo que es fijo del último entrenamiento)
            features_count = 568  # Del último entrenamiento exitoso
            
            # Obtener fecha de última actualización
            last_doc = await complete_collection.find_one(
                {}, 
                sort=[("_id", -1)]
            )
            last_update = None
            if last_doc and "_id" in last_doc:
                last_update = last_doc["_id"].generation_time.isoformat()
            
            return PostulacionesDatasetStats(
                total_records=total_records,
                labeled_records=labeled_records,
                unlabeled_records=unlabeled_records,
                state_distribution=estados_dist,
                features_count=features_count,
                last_update=last_update
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas del dataset: {e}")
            return None
    
    @strawberry.field
    async def get_predictions(self, filters: Optional[PredictionFilter] = None) -> List[PostulacionPrediction]:
        """Obtiene predicciones guardadas con filtros opcionales"""
        try:
            # Conectar a MongoDB si no está conectado
            if mongodb.database is None:
                await mongodb.connect()
            
            collection = mongodb.get_collection("postulaciones_predictions")
            
            # Construir query de MongoDB
            mongo_query = {}
            
            if filters:
                if filters.estado_predicho:
                    mongo_query['estado_predicho'] = filters.estado_predicho
                
                if filters.model_used:
                    mongo_query['model_used'] = filters.model_used
                
                if filters.min_accuracy:
                    mongo_query['model_accuracy'] = {'$gte': filters.min_accuracy}
            
            # Ejecutar query
            cursor = collection.find(mongo_query)
            
            if filters and filters.limit:
                cursor = cursor.limit(filters.limit)
            
            predictions = []
            async for doc in cursor:
                prediction = PostulacionPrediction(
                    postulacion_id=str(doc.get('postulacion_id', '')),
                    estado_original=doc.get('estado_original'),
                    estado_predicho=doc.get('estado_predicho', ''),
                    estado_predicho_encoded=doc.get('estado_predicho_encoded', 0),
                    model_used=doc.get('model_used', ''),
                    prediction_date=doc.get('prediction_date', ''),
                    accuracy=doc.get('model_accuracy')
                )
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error obteniendo predicciones: {e}")
            return []
    
    @strawberry.field
    async def predict_single_postulacion(self, postulacion_data: PostulacionInput) -> Optional[PostulacionPredictionResult]:
        """Predice el estado de una postulación individual"""
        try:
            # Verificar si el modelo está cargado
            if not postulaciones_trainer.is_trained:
                model_path = os.path.join(settings.ml_models_path, "postulaciones", "semi_supervised_model.pkl")
                if os.path.exists(model_path):
                    postulaciones_trainer.load_model(model_path)
                else:
                    logger.error("No hay modelo entrenado disponible")
                    return None
            
            # Convertir input a diccionario
            input_dict = {
                'nombre': postulacion_data.nombre,
                'anios_experiencia': postulacion_data.anios_experiencia,
                'nivel_educacion': postulacion_data.nivel_educacion,
                'habilidades': postulacion_data.habilidades,
                'idiomas': postulacion_data.idiomas,
                'certificaciones': postulacion_data.certificaciones,
                'puesto_actual': postulacion_data.puesto_actual,
                'url_cv': postulacion_data.url_cv,
                'oferta_titulo': postulacion_data.oferta_titulo,
                'oferta_descripcion': postulacion_data.oferta_descripcion,
                'salario': postulacion_data.salario,
                'ubicacion': postulacion_data.ubicacion,
                'requisitos': postulacion_data.requisitos,
                'empresa_rubro': postulacion_data.empresa_rubro
            }
            
            # Realizar predicción
            prediction_result = postulaciones_trainer.predict_estado(input_dict)
            
            # Crear objetos de respuesta
            postulacion_features = PostulacionFeatures(
                postulacion_id="temp_" + datetime.now().strftime('%Y%m%d_%H%M%S'),
                nombre=postulacion_data.nombre,
                anios_experiencia=postulacion_data.anios_experiencia,
                nivel_educacion=postulacion_data.nivel_educacion,
                habilidades=postulacion_data.habilidades,
                idiomas=postulacion_data.idiomas,
                certificaciones=postulacion_data.certificaciones,
                puesto_actual=postulacion_data.puesto_actual,
                oferta_titulo=postulacion_data.oferta_titulo,
                salario=postulacion_data.salario,
                ubicacion=postulacion_data.ubicacion,
                requisitos=postulacion_data.requisitos,
                empresa_rubro=postulacion_data.empresa_rubro
            )
            
            prediction = PostulacionPrediction(
                postulacion_id=postulacion_features.postulacion_id,
                estado_original=None,
                estado_predicho=prediction_result['estado_predicho'],
                estado_predicho_encoded=prediction_result['estado_encoded'],
                model_used=prediction_result['model_used'],
                prediction_date=prediction_result['prediction_date'],
                accuracy=None
            )
            
            # Calcular confidence score si hay probabilidades
            confidence_score = None
            probabilities_json = None
            
            if prediction_result.get('probabilities'):
                probs = prediction_result['probabilities']
                confidence_score = max(probs.values()) if probs else None
                probabilities_json = json.dumps(probs)
            
            return PostulacionPredictionResult(
                postulacion_data=postulacion_features,
                prediction=prediction,
                confidence_score=confidence_score,
                probabilities=probabilities_json
            )
            
        except Exception as e:
            logger.error(f"Error en predicción individual: {e}")
            return None
    
    @strawberry.field
    async def get_all_predictions_batch(self) -> Optional[PredictionBatch]:
        """Obtiene todas las predicciones en un lote con información del modelo"""
        try:
            # Obtener predicciones
            predictions = await self.get_predictions()
            
            # Obtener información del modelo
            model_info = await self.get_model_info()
            
            if model_info is None:
                return None
            
            return PredictionBatch(
                predictions=predictions,
                total_predictions=len(predictions),
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo lote de predicciones: {e}")
            return None


@strawberry.type  
class PostulacionesMutation:
    """Mutaciones para el modelo de postulaciones"""
    
    @strawberry.mutation
    async def train_semi_supervised_model(self, config: Optional[TrainingConfig] = None) -> TrainingResult:
        """Entrena el modelo semi-supervisado"""
        start_time = datetime.now()
        
        try:
            logger.info("Iniciando entrenamiento de modelo semi-supervisado via GraphQL")
            
            # Verificar si ya hay modelo entrenado y no se fuerza reentrenamiento
            if config and not config.force_retrain:
                model_path = os.path.join(settings.ml_models_path, "postulaciones", "semi_supervised_model.pkl")
                if os.path.exists(model_path) and postulaciones_trainer.is_trained:
                    return TrainingResult(
                        success=False,
                        message="El modelo ya está entrenado. Use force_retrain=true para reentrenar.",
                        model_info=None,
                        training_duration_seconds=0.0,
                        files_created=[]
                    )
            
            # Ejecutar entrenamiento
            training_summary = await postulaciones_trainer.train_full_pipeline()
            
            # Calcular duración
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Obtener información del modelo entrenado
            model_info = await PostulacionesQuery().get_model_info()
            
            # Archivos creados
            files_created = [
                training_summary.get('model_path', ''),
                training_summary.get('preprocessor_path', '')
            ]
            
            return TrainingResult(
                success=True,
                message=f"Modelo entrenado exitosamente. Mejor modelo: {training_summary['best_model']}",
                model_info=model_info,
                training_duration_seconds=duration,
                files_created=[f for f in files_created if f]
            )
            
        except Exception as e:
            logger.error(f"Error en entrenamiento via GraphQL: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            
            return TrainingResult(
                success=False,
                message=f"Error durante el entrenamiento: {str(e)}",
                model_info=None,
                training_duration_seconds=duration,
                files_created=[]
            )
    
    @strawberry.mutation
    async def load_trained_model(self, model_path: Optional[str] = None) -> TrainingResult:
        """Carga un modelo previamente entrenado"""
        try:
            if model_path is None:
                model_path = os.path.join(settings.ml_models_path, "postulaciones", "semi_supervised_model.pkl")
            
            if not os.path.exists(model_path):
                return TrainingResult(
                    success=False,
                    message=f"Archivo de modelo no encontrado: {model_path}",
                    model_info=None,
                    training_duration_seconds=0.0,
                    files_created=[]
                )
            
            # Cargar modelo
            postulaciones_trainer.load_model(model_path)
            
            # Obtener información del modelo cargado
            model_info = await PostulacionesQuery().get_model_info()
            
            return TrainingResult(
                success=True,
                message=f"Modelo cargado exitosamente desde: {model_path}",
                model_info=model_info,
                training_duration_seconds=0.0,
                files_created=[]
            )
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            
            return TrainingResult(
                success=False,
                message=f"Error cargando modelo: {str(e)}",
                model_info=None,
                training_duration_seconds=0.0,
                files_created=[]
            )
    
    @strawberry.mutation
    async def extract_data_from_postgres(self) -> TrainingResult:
        """Extrae datos desde PostgreSQL y los guarda en MongoDB"""
        start_time = datetime.now()
        
        try:
            logger.info("Extrayendo datos desde PostgreSQL via GraphQL")
            
            # Extraer datos
            complete_df = await postgres_extractor.extract_complete_dataset()
            
            if complete_df.empty:
                return TrainingResult(
                    success=False,
                    message="No se pudieron extraer datos de PostgreSQL",
                    model_info=None,
                    training_duration_seconds=0.0,
                    files_created=[]
                )
            
            # Guardar en MongoDB
            await postgres_extractor.save_to_mongo(complete_df, "postulaciones_completas")
            
            # Solo datos etiquetados
            labeled_df = complete_df[complete_df['estado'].notna()]
            await postgres_extractor.save_to_mongo(labeled_df, "postulaciones_labeled")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TrainingResult(
                success=True,
                message=f"Datos extraídos exitosamente. Total: {len(complete_df)} registros, Etiquetados: {len(labeled_df)}",
                model_info=None,
                training_duration_seconds=duration,
                files_created=["MongoDB: postulaciones_completas", "MongoDB: postulaciones_labeled"]
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo datos via GraphQL: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            
            return TrainingResult(
                success=False,
                message=f"Error extrayendo datos: {str(e)}",
                model_info=None,
                training_duration_seconds=duration,
                files_created=[]
            )