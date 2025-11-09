"""
Resolvers de GraphQL para modelo semi-supervisado de postulaciones
"""
import strawberry
import asyncio
import logging
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.graphql.types.ml_types import (
    SemiSupervisedTrainingInput,
    SemiSupervisedTrainingResult,
    SemiSupervisedModelInfo,
    SemiSupervisedDataSummary,
    PostulacionEstadoPredictionInput,
    PostulacionEstadoPrediction,
    BatchEstadoPredictionInput,
    BatchEstadoPredictionResult,
    SemiSupervisedModelComparison,
    UnlabeledDataInsights,
    RetrainModelInput,
    RetrainModelResult,
    PredictionConfidenceAnalysis,
    KeyIntValuePair,
    KeyFloatValuePair,
    KeyValuePair
)

from app.ml.training.semi_supervised_trainer import semi_supervised_trainer
from app.ml.data.postgres_extractor import postgres_extractor
from app.ml.models.semi_supervised_model import SemiSupervisedPostulacionModel
from app.config.connection import init_database, mongodb

logger = logging.getLogger(__name__)

@strawberry.type
class SemiSupervisedMLResolver:
    """Resolver para operaciones de Machine Learning semi-supervisado"""
    
    @strawberry.field
    async def train_semi_supervised_models(
        self, 
        config: Optional[SemiSupervisedTrainingInput] = None
    ) -> SemiSupervisedTrainingResult:
        """Entrena modelos semi-supervisados para predicción de estados de postulaciones"""
        try:
            logger.info("Iniciando entrenamiento de modelos semi-supervisados")
            
            # Configuración por defecto
            if config is None:
                config = SemiSupervisedTrainingInput()
            
            # Inicializar conexiones
            await init_database()
            
            # Entrenar modelos
            training_summary = await semi_supervised_trainer.train_all_models(
                save_to_mongo=config.save_to_mongo or True
            )
            
            if not training_summary:
                return SemiSupervisedTrainingResult(
                    success=False,
                    message="Error en el entrenamiento de modelos",
                    total_samples=0,
                    labeled_samples=0,
                    unlabeled_samples=0,
                    features_count=0,
                    classes_found=[],
                    models_trained=[],
                    models_info=[]
                )
            
            # Convertir resultados
            models_info = []
            for model_type, results in training_summary['training_results'].items():
                if 'error' not in results:
                    model_info = SemiSupervisedModelInfo(
                        model_type=model_type,
                        is_trained=True,
                        training_timestamp=results.get('timestamp'),
                        train_accuracy=results.get('train_accuracy'),
                        val_accuracy=results.get('val_accuracy'),
                        labeled_samples=results.get('training_samples'),
                        unlabeled_samples=results.get('unlabeled_samples'),
                        total_samples=results.get('total_samples'),
                        classes=results.get('classes', []),
                        unlabeled_predictions_count=results.get('total_predictions'),
                        prediction_confidence_mean=results.get('prediction_confidence', {}).get('mean'),
                        prediction_distribution=results.get('unlabeled_predictions_distribution'),
                        model_path=f"trained_models/semi_supervised/{model_type}_model.pkl",
                        metrics_available=True
                    )
                    models_info.append(model_info)
            
            # Archivos generados
            files_generated = []
            if 'files_generated' in training_summary:
                files_generated.extend(training_summary['files_generated'].get('models', {}).values())
                if 'preprocessor' in training_summary['files_generated']:
                    files_generated.append(training_summary['files_generated']['preprocessor'])
            
            return SemiSupervisedTrainingResult(
                success=True,
                message="Modelos semi-supervisados entrenados exitosamente",
                total_samples=training_summary['data_info']['total_samples'],
                labeled_samples=training_summary['data_info']['labeled_samples'],
                unlabeled_samples=training_summary['data_info']['unlabeled_samples'],
                features_count=training_summary['data_info']['features_count'],
                classes_found=list(training_summary['data_info']['classes'].keys()),
                models_trained=list(training_summary['training_results'].keys()),
                models_info=models_info,
                best_model_type=training_summary['best_model']['model_type'],
                best_model_score=training_summary['best_model']['score'],
                files_generated=files_generated,
                unlabeled_predictions_generated=training_summary.get('best_model', {}).get('results', {}).get('total_predictions'),
                high_confidence_predictions=training_summary.get('best_model', {}).get('results', {}).get('high_confidence_count')
            )
            
        except Exception as e:
            logger.error(f"Error entrenando modelos semi-supervisados: {str(e)}")
            return SemiSupervisedTrainingResult(
                success=False,
                message=f"Error en entrenamiento: {str(e)}",
                total_samples=0,
                labeled_samples=0,
                unlabeled_samples=0,
                features_count=0,
                classes_found=[],
                models_trained=[],
                models_info=[],
                errors=[str(e)]
            )
    
    @strawberry.field
    async def predict_postulacion_estado(
        self, 
        input_data: PostulacionEstadoPredictionInput,
        model_type: Optional[str] = None
    ) -> PostulacionEstadoPrediction:
        """Predice el estado de una postulación usando modelo semi-supervisado"""
        try:
            start_time = datetime.now()
            
            # Cargar modelo y preprocesador
            if model_type is None:
                # Buscar el mejor modelo disponible
                model_files = [f for f in os.listdir("trained_models/semi_supervised") if f.endswith("_model.pkl")]
                if not model_files:
                    return PostulacionEstadoPrediction(
                        predicted_estado="error",
                        confidence=0.0,
                        error="No hay modelos entrenados disponibles"
                    )
                model_type = model_files[0].replace("_model.pkl", "")
            
            # Cargar modelo
            model = semi_supervised_trainer.load_trained_model(model_type)
            
            # Cargar preprocesador
            preprocessor_path = "trained_models/semi_supervised/preprocessor.pkl"
            if not os.path.exists(preprocessor_path):
                return PostulacionEstadoPrediction(
                    predicted_estado="error",
                    confidence=0.0,
                    error="Preprocesador no encontrado"
                )
            
            semi_supervised_trainer.preprocessor.load(preprocessor_path)
            
            # Preparar datos para predicción
            if input_data.postulacion_id:
                # Extraer datos de BD usando ID
                await init_database()
                df = await postgres_extractor.extract_postulaciones_with_features()
                
                # Filtrar por ID
                postulacion_df = df[df['postulacion_id'].astype(str) == input_data.postulacion_id]
                
                if postulacion_df.empty:
                    return PostulacionEstadoPrediction(
                        postulacion_id=input_data.postulacion_id,
                        predicted_estado="error",
                        confidence=0.0,
                        error="Postulación no encontrada"
                    )
            else:
                # Usar datos manuales
                data_dict = {
                    'nombre': input_data.nombre or '',
                    'anios_experiencia': input_data.anios_experiencia or 0,
                    'nivel_educacion': input_data.nivel_educacion or '',
                    'habilidades': input_data.habilidades or '',
                    'idiomas': input_data.idiomas or '',
                    'certificaciones': input_data.certificaciones or '',
                    'puesto_actual': input_data.puesto_actual or '',
                    'oferta_titulo': input_data.oferta_titulo or '',
                    'oferta_salario': input_data.oferta_salario or 0,
                    'oferta_requisitos': input_data.oferta_requisitos or '',
                    'empresa_rubro': input_data.empresa_rubro or '',
                    # Valores por defecto para campos requeridos
                    'total_entrevistas': 0,
                    'promedio_duracion_entrevistas': 0,
                    'promedio_calificacion_tecnica': 0,
                    'promedio_calificacion_actitud': 0,
                    'promedio_calificacion_general': 0,
                    'fecha_postulacion': datetime.now()
                }
                postulacion_df = pd.DataFrame([data_dict])
            
            # Procesar datos
            X = semi_supervised_trainer.preprocessor.transform(postulacion_df)
            
            # Realizar predicción
            prediction = model.predict(X)[0]
            confidence = model.get_prediction_confidence(X)[0]
            
            # Convertir predicción numérica a etiqueta
            label_encoder = semi_supervised_trainer.preprocessor.label_encoder
            predicted_estado = label_encoder.inverse_transform([prediction])[0]
            
            # Obtener distribución de probabilidades si está disponible
            probability_distribution = None
            try:
                proba = model.predict_proba(X)[0]
                probability_distribution = dict(zip(
                    label_encoder.classes_, 
                    [float(p) for p in proba]
                ))
            except:
                pass
            
            # Determinar nivel de confianza
            confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            
            # Calcular factores clave (simplificado)
            key_factors = []
            if input_data.anios_experiencia and input_data.anios_experiencia > 5:
                key_factors.append("Experiencia alta")
            if input_data.habilidades and len(input_data.habilidades) > 50:
                key_factors.append("Habilidades detalladas")
            if input_data.certificaciones:
                key_factors.append("Certificaciones disponibles")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return PostulacionEstadoPrediction(
                postulacion_id=input_data.postulacion_id,
                predicted_estado=predicted_estado,
                confidence=float(confidence),
                probability_distribution=probability_distribution,
                confidence_level=confidence_level,
                model_used=model_type,
                prediction_timestamp=datetime.now().isoformat(),
                key_factors=key_factors,
                experience_score=min(input_data.anios_experiencia / 10.0, 1.0) if input_data.anios_experiencia else 0.0,
                skills_score=min(len(input_data.habilidades or '') / 100.0, 1.0),
                education_score=0.8 if input_data.nivel_educacion in ['Universitario', 'Postgrado'] else 0.5,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error en predicción de estado: {str(e)}")
            return PostulacionEstadoPrediction(
                postulacion_id=input_data.postulacion_id,
                predicted_estado="error",
                confidence=0.0,
                error=str(e)
            )
    
    @strawberry.field
    async def predict_batch_estados(
        self, 
        input_data: BatchEstadoPredictionInput
    ) -> BatchEstadoPredictionResult:
        """Predice estados de múltiples postulaciones en batch"""
        try:
            start_time = datetime.now()
            
            predictions = []
            success_count = 0
            error_count = 0
            
            for postulacion_input in input_data.postulaciones:
                try:
                    prediction = await self.predict_postulacion_estado(
                        postulacion_input, 
                        input_data.model_type
                    )
                    predictions.append(prediction)
                    
                    if prediction.error:
                        error_count += 1
                    else:
                        success_count += 1
                        
                except Exception as e:
                    error_prediction = PostulacionEstadoPrediction(
                        postulacion_id=postulacion_input.postulacion_id,
                        predicted_estado="error",
                        confidence=0.0,
                        error=str(e)
                    )
                    predictions.append(error_prediction)
                    error_count += 1
            
            # Estadísticas del resumen
            summary_stats = {}
            for pred in predictions:
                if pred.predicted_estado != "error":
                    summary_stats[pred.predicted_estado] = summary_stats.get(pred.predicted_estado, 0) + 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return BatchEstadoPredictionResult(
                predictions=predictions,
                total_processed=len(input_data.postulaciones),
                success_count=success_count,
                error_count=error_count,
                model_used=input_data.model_type or "auto_selected",
                processing_time=processing_time,
                summary_stats=summary_stats
            )
            
        except Exception as e:
            logger.error(f"Error en predicción batch: {str(e)}")
            return BatchEstadoPredictionResult(
                predictions=[],
                total_processed=len(input_data.postulaciones) if input_data.postulaciones else 0,
                success_count=0,
                error_count=len(input_data.postulaciones) if input_data.postulaciones else 0,
                model_used="error",
                processing_time=0.0,
                summary_stats={"error": len(input_data.postulaciones) if input_data.postulaciones else 0}
            )
    
    @strawberry.field
    async def get_semi_supervised_data_summary(self) -> SemiSupervisedDataSummary:
        """Obtiene resumen de datos para modelo semi-supervisado"""
        try:
            await init_database()
            
            # Extraer estadísticas de tablas
            table_stats = await postgres_extractor.get_table_stats()
            
            # Extraer distribución de estados
            estado_distribution = await postgres_extractor.extract_estado_distribution()
            
            # Extraer datos completos para análisis
            df = await postgres_extractor.extract_postulaciones_with_features()
            
            if df.empty:
                return SemiSupervisedDataSummary(
                    total_postulaciones=0,
                    labeled_postulaciones=0,
                    unlabeled_postulaciones=0,
                    labeled_percentage=0.0,
                    estado_distribution=[],
                    can_train_semi_supervised=False,
                    recommendations=["No hay datos disponibles para entrenar"],
                    table_stats=[]
                )
            
            # Calcular estadísticas
            total_postulaciones = len(df)
            labeled_postulaciones = df['estado'].notna().sum()
            unlabeled_postulaciones = total_postulaciones - labeled_postulaciones
            labeled_percentage = (labeled_postulaciones / total_postulaciones) * 100
            
            # Calcular calidad de datos
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            completeness_score = 100 - missing_percentage
            
            # Determinar si se puede entrenar
            can_train = labeled_postulaciones >= 5 and unlabeled_postulaciones > 0
            
            # Generar recomendaciones
            recommendations = []
            if labeled_postulaciones < 5:
                recommendations.append("Se necesitan al menos 5 muestras etiquetadas para entrenamiento semi-supervisado")
            if unlabeled_postulaciones == 0:
                recommendations.append("No hay datos no etiquetados para aprendizaje semi-supervisado")
            if labeled_percentage < 10:
                recommendations.append("Se recomienda tener al menos 10% de datos etiquetados")
            if missing_percentage > 30:
                recommendations.append("Alto porcentaje de datos faltantes. Considerar limpieza de datos")
            if can_train:
                recommendations.append("Los datos son adecuados para entrenamiento semi-supervisado")
            
            # Convertir diccionarios a tipos GraphQL
            estado_distribution_pairs = [
                KeyIntValuePair(key=str(k), value=int(v)) 
                for k, v in estado_distribution.items()
            ]
            
            table_stats_pairs = [
                KeyValuePair(key=str(k), value=str(v)) 
                for k, v in table_stats.items()
            ]
            
            return SemiSupervisedDataSummary(
                total_postulaciones=total_postulaciones,
                labeled_postulaciones=int(labeled_postulaciones),
                unlabeled_postulaciones=unlabeled_postulaciones,
                labeled_percentage=float(labeled_percentage),
                estado_distribution=estado_distribution_pairs,
                missing_data_percentage=float(missing_percentage),
                completeness_score=float(completeness_score),
                can_train_semi_supervised=bool(can_train),
                recommendations=recommendations,
                table_stats=table_stats_pairs
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen de datos: {str(e)}")
            return SemiSupervisedDataSummary(
                total_postulaciones=0,
                labeled_postulaciones=0,
                unlabeled_postulaciones=0,
                labeled_percentage=0.0,
                estado_distribution=[],
                can_train_semi_supervised=False,
                recommendations=[f"Error obteniendo datos: {str(e)}"],
                table_stats=[]
            )
    
    @strawberry.field
    async def get_trained_models_info(self) -> List[SemiSupervisedModelInfo]:
        """Obtiene información de modelos semi-supervisados entrenados"""
        try:
            models_info = []
            models_dir = "trained_models/semi_supervised"
            
            if not os.path.exists(models_dir):
                return models_info
            
            # Buscar archivos de modelos
            model_files = [f for f in os.listdir(models_dir) if f.endswith("_model.pkl")]
            
            for model_file in model_files:
                model_type = model_file.replace("_model.pkl", "")
                model_path = os.path.join(models_dir, model_file)
                
                try:
                    # Cargar modelo para obtener información
                    model = SemiSupervisedPostulacionModel(model_type=model_type)
                    model.load(model_path)
                    
                    # Buscar archivo de métricas
                    metrics_file = model_file.replace("_model.pkl", "_model_metrics.json")
                    metrics_path = os.path.join(models_dir, metrics_file)
                    
                    metrics_data = {}
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            metrics_data = json.load(f)
                    
                    model_info = SemiSupervisedModelInfo(
                        model_type=model_type,
                        is_trained=model.is_trained,
                        training_timestamp=metrics_data.get('timestamp'),
                        train_accuracy=metrics_data.get('train_accuracy'),
                        val_accuracy=metrics_data.get('val_accuracy'),
                        labeled_samples=metrics_data.get('training_samples'),
                        unlabeled_samples=metrics_data.get('unlabeled_samples'),
                        total_samples=metrics_data.get('total_samples'),
                        classes=metrics_data.get('classes', []),
                        unlabeled_predictions_count=metrics_data.get('total_predictions'),
                        prediction_confidence_mean=metrics_data.get('prediction_confidence', {}).get('mean'),
                        prediction_distribution=metrics_data.get('unlabeled_predictions_distribution'),
                        model_path=model_path,
                        metrics_available=os.path.exists(metrics_path)
                    )
                    
                    models_info.append(model_info)
                    
                except Exception as e:
                    logger.warning(f"Error cargando modelo {model_type}: {str(e)}")
                    continue
            
            return models_info
            
        except Exception as e:
            logger.error(f"Error obteniendo información de modelos: {str(e)}")
            return []
    
    @strawberry.field
    async def analyze_unlabeled_data(self) -> UnlabeledDataInsights:
        """Analiza datos no etiquetados y proporciona insights"""
        try:
            await init_database()
            
            # Extraer datos no etiquetados
            unlabeled_df = await postgres_extractor.extract_missing_estado_postulaciones()
            
            if unlabeled_df.empty:
                return UnlabeledDataInsights(
                    total_unlabeled=0,
                    predicted_estados={},
                    confidence_stats=PredictionConfidenceAnalysis(
                        total_predictions=0,
                        high_confidence_count=0,
                        medium_confidence_count=0,
                        low_confidence_count=0,
                        confidence_distribution={},
                        reliable_predictions=0,
                        review_needed_predictions=0,
                        manual_verification_needed=0
                    ),
                    common_patterns=["No hay datos no etiquetados disponibles"],
                    labeling_strategy="No se requiere etiquetado adicional"
                )
            
            # Buscar modelo entrenado
            model_files = [f for f in os.listdir("trained_models/semi_supervised") if f.endswith("_model.pkl")]
            if not model_files:
                return UnlabeledDataInsights(
                    total_unlabeled=len(unlabeled_df),
                    predicted_estados={},
                    confidence_stats=PredictionConfidenceAnalysis(
                        total_predictions=0,
                        high_confidence_count=0,
                        medium_confidence_count=0,
                        low_confidence_count=0,
                        confidence_distribution={},
                        reliable_predictions=0,
                        review_needed_predictions=0,
                        manual_verification_needed=0
                    ),
                    common_patterns=["No hay modelos entrenados para hacer predicciones"],
                    labeling_strategy="Entrenar modelo semi-supervisado primero"
                )
            
            # Cargar modelo y hacer predicciones
            model_type = model_files[0].replace("_model.pkl", "")
            model = semi_supervised_trainer.load_trained_model(model_type)
            
            # Cargar preprocesador
            preprocessor_path = "trained_models/semi_supervised/preprocessor.pkl"
            semi_supervised_trainer.preprocessor.load(preprocessor_path)
            
            # Procesar y predecir
            X = semi_supervised_trainer.preprocessor.transform(unlabeled_df)
            predictions = model.predict(X)
            confidence_scores = model.get_prediction_confidence(X)
            
            # Convertir predicciones
            label_encoder = semi_supervised_trainer.preprocessor.label_encoder
            predicted_labels = label_encoder.inverse_transform(predictions)
            
            # Análisis de distribución
            unique_preds, counts = np.unique(predicted_labels, return_counts=True)
            predicted_estados = dict(zip(unique_preds.tolist(), counts.tolist()))
            
            # Análisis de confianza
            high_conf = np.sum(confidence_scores > 0.8)
            medium_conf = np.sum((confidence_scores > 0.6) & (confidence_scores <= 0.8))
            low_conf = np.sum(confidence_scores <= 0.6)
            
            confidence_distribution = {
                "high": float(high_conf / len(confidence_scores)),
                "medium": float(medium_conf / len(confidence_scores)),
                "low": float(low_conf / len(confidence_scores))
            }
            
            confidence_stats = PredictionConfidenceAnalysis(
                total_predictions=len(predictions),
                high_confidence_count=int(high_conf),
                medium_confidence_count=int(medium_conf),
                low_confidence_count=int(low_conf),
                confidence_distribution=confidence_distribution,
                reliable_predictions=int(high_conf),
                review_needed_predictions=int(medium_conf),
                manual_verification_needed=int(low_conf)
            )
            
            # Patrones comunes (simplificado)
            common_patterns = []
            if len(unlabeled_df) > 0:
                # Analizar experiencia
                exp_mean = unlabeled_df['anios_experiencia'].mean() if 'anios_experiencia' in unlabeled_df.columns else 0
                if exp_mean > 5:
                    common_patterns.append("Candidatos con alta experiencia promedio")
                elif exp_mean < 2:
                    common_patterns.append("Candidatos junior predominantes")
                
                # Analizar rubros más comunes
                if 'empresa_rubro' in unlabeled_df.columns:
                    top_rubro = unlabeled_df['empresa_rubro'].mode().iloc[0] if len(unlabeled_df['empresa_rubro'].mode()) > 0 else None
                    if top_rubro:
                        common_patterns.append(f"Rubro predominante: {top_rubro}")
            
            # Estrategia de etiquetado
            labeling_strategy = "Priorizar etiquetado manual de predicciones con baja confianza"
            if low_conf > len(predictions) * 0.5:
                labeling_strategy = "Alto porcentaje de predicciones con baja confianza. Considerar recolectar más datos etiquetados"
            elif high_conf > len(predictions) * 0.8:
                labeling_strategy = "Predicciones mayormente confiables. Etiquetado selectivo recomendado"
            
            return UnlabeledDataInsights(
                total_unlabeled=len(unlabeled_df),
                predicted_estados=predicted_estados,
                confidence_stats=confidence_stats,
                common_patterns=common_patterns,
                labeling_strategy=labeling_strategy
            )
            
        except Exception as e:
            logger.error(f"Error analizando datos no etiquetados: {str(e)}")
            return UnlabeledDataInsights(
                total_unlabeled=0,
                predicted_estados={},
                confidence_stats=PredictionConfidenceAnalysis(
                    total_predictions=0,
                    high_confidence_count=0,
                    medium_confidence_count=0,
                    low_confidence_count=0,
                    confidence_distribution={},
                    reliable_predictions=0,
                    review_needed_predictions=0,
                    manual_verification_needed=0
                ),
                common_patterns=[f"Error en análisis: {str(e)}"]
            )

# Instancia del resolver
semi_supervised_resolver = SemiSupervisedMLResolver()