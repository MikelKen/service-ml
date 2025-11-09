#!/usr/bin/env python3
"""
🔗 RESOLVERS GRAPHQL PARA MODELO SEMI-SUPERVISADO
Implementa resolvers para entrenamiento, predicciones y consultas del modelo semi-supervisado
"""

import strawberry
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import os
import uuid

from app.graphql.types.semi_supervised_types import (
    SemiSupervisedPrediction, ApplicationWithPrediction, ModelTrainingResult,
    ModelInfo, BatchPredictionResult, ModelAnalysis, DatasetStatistics,
    ModelComparison, OperationResult, ValidationResult, PaginatedApplications,
    FeatureImportance, CompatibilityFeatures, CandidateFeatures, OfferFeatures,
    ModelMetrics, PseudoLabelStats, TrainingConfig,
    # Input types
    TrainingParameters, PredictionInput, BatchPredictionInput,
    ModelSelectionCriteria, ApplicationFilter, PaginationInput,
    # Enums
    SemiSupervisedAlgorithm, ConfidenceLevel, LabelQuality, PredictionStatus
)

from app.ml.models.semi_supervised_model import SemiSupervisedClassifier, train_semi_supervised_model
from app.ml.preprocessing.semi_supervised_preprocessor import semi_supervised_preprocessor
from app.config.mongodb_connection import get_mongodb_sync, mongodb_connection
import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiSupervisedMLResolvers:
    """Resolvers para operaciones de machine learning semi-supervisado"""
    
    def __init__(self):
        self.current_model: Optional[SemiSupervisedClassifier] = None
        self.db = None
    
    async def _get_db(self):
        """Obtener conexión a MongoDB"""
        if not self.db:
            mongodb_connection.connect_sync()
            self.db = get_mongodb_sync()
        return self.db
    
    async def get_dataset_statistics(self) -> DatasetStatistics:
        """Obtener estadísticas del dataset"""
        logger.info("📊 Obteniendo estadísticas del dataset...")
        
        try:
            db = await self._get_db()
            applications_collection = db['ml_applications']
            
            # Contar totales
            total_applications = applications_collection.count_documents({})
            labeled_applications = applications_collection.count_documents({'is_labeled': True})
            unlabeled_applications = total_applications - labeled_applications
            
            # Distribución por estado de predicción
            accepted_applications = applications_collection.count_documents({'ml_target': 1})
            rejected_applications = applications_collection.count_documents({'ml_target': 0})
            pending_applications = applications_collection.count_documents({'ml_target': -1})
            
            # Distribución temporal
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            year_ago = now - timedelta(days=365)
            
            applications_last_week = applications_collection.count_documents({
                'fecha_postulacion': {'$gte': week_ago}
            })
            applications_last_month = applications_collection.count_documents({
                'fecha_postulacion': {'$gte': month_ago}
            })
            applications_last_year = applications_collection.count_documents({
                'fecha_postulacion': {'$gte': year_ago}
            })
            
            # Calidad de datos (simplificado)
            candidates_collection = db['ml_candidates']
            total_candidates = candidates_collection.count_documents({})
            complete_profiles = candidates_collection.count_documents({
                'features.profile_completeness': {'$gte': 0.8}
            })
            
            return DatasetStatistics(
                total_applications=total_applications,
                labeled_applications=labeled_applications,
                unlabeled_applications=unlabeled_applications,
                labeled_ratio=labeled_applications / max(total_applications, 1),
                accepted_applications=accepted_applications,
                rejected_applications=rejected_applications,
                pending_applications=pending_applications,
                applications_last_week=applications_last_week,
                applications_last_month=applications_last_month,
                applications_last_year=applications_last_year,
                complete_profiles=complete_profiles,
                incomplete_profiles=total_candidates - complete_profiles,
                missing_features_count=0  # Calcular si es necesario
            )
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            raise
    
    async def get_applications_with_predictions(
        self, 
        filter_params: Optional[ApplicationFilter] = None,
        pagination: Optional[PaginationInput] = None
    ) -> PaginatedApplications:
        """Obtener aplicaciones con predicciones paginadas"""
        logger.info("📋 Obteniendo aplicaciones con predicciones...")
        
        try:
            db = await self._get_db()
            applications_collection = db['ml_applications']
            candidates_collection = db['ml_candidates']
            offers_collection = db['ml_job_offers']
            
            # Construir filtro de consulta
            query = {}
            if filter_params:
                if filter_params.candidate_ids:
                    query['candidate_id'] = {'$in': filter_params.candidate_ids}
                if filter_params.offer_ids:
                    query['offer_id'] = {'$in': filter_params.offer_ids}
                if filter_params.date_from or filter_params.date_to:
                    date_filter = {}
                    if filter_params.date_from:
                        date_filter['$gte'] = filter_params.date_from
                    if filter_params.date_to:
                        date_filter['$lte'] = filter_params.date_to
                    query['fecha_postulacion'] = date_filter
                if filter_params.is_labeled is not None:
                    query['is_labeled'] = filter_params.is_labeled
                if filter_params.confidence_level:
                    query['ml_confidence'] = filter_params.confidence_level.value
                if filter_params.prediction_status:
                    query['ml_prediction'] = filter_params.prediction_status.value
            
            # Configurar paginación
            if not pagination:
                pagination = PaginationInput()
            
            skip = (pagination.page - 1) * pagination.page_size
            sort_order = 1 if pagination.sort_order == "ASC" else -1
            
            # Obtener aplicaciones
            applications_cursor = applications_collection.find(query).skip(skip).limit(pagination.page_size).sort(pagination.sort_by, sort_order)
            applications_data = list(applications_cursor)
            
            # Contar total
            total_count = applications_collection.count_documents(query)
            
            # Crear mapas para joins
            candidate_ids = [app['candidate_id'] for app in applications_data]
            offer_ids = [app['offer_id'] for app in applications_data]
            
            candidates_cursor = candidates_collection.find({'candidate_id': {'$in': candidate_ids}})
            candidates_map = {c['candidate_id']: c for c in candidates_cursor}
            
            offers_cursor = offers_collection.find({'offer_id': {'$in': offer_ids}})
            offers_map = {o['offer_id']: o for o in offers_cursor}
            
            # Construir resultado
            applications = []
            for app_data in applications_data:
                candidate = candidates_map.get(app_data['candidate_id'], {})
                offer = offers_map.get(app_data['offer_id'], {})
                
                # Extraer features de compatibilidad
                compat_features = app_data.get('compatibility_features', {})
                compatibility_features = CompatibilityFeatures(
                    skill_match_score=compat_features.get('skill_match_score', 0.0),
                    experience_match=compat_features.get('experience_match', 0.0),
                    education_match=compat_features.get('education_match', 0.0),
                    location_match=compat_features.get('location_match', False),
                    salary_expectation_match=compat_features.get('salary_expectation_match', 0.0),
                    overall_compatibility=compat_features.get('overall_compatibility', 0.0)
                )
                
                # Convertir enums
                ml_prediction = None
                if app_data.get('ml_prediction') is not None:
                    pred_value = app_data['ml_prediction']
                    if pred_value == 1:
                        ml_prediction = PredictionStatus.ACCEPTED
                    elif pred_value == 0:
                        ml_prediction = PredictionStatus.REJECTED
                    else:
                        ml_prediction = PredictionStatus.UNKNOWN
                
                ml_confidence = None
                if app_data.get('ml_confidence'):
                    conf_value = app_data['ml_confidence']
                    if conf_value == 'high':
                        ml_confidence = ConfidenceLevel.HIGH
                    elif conf_value == 'medium':
                        ml_confidence = ConfidenceLevel.MEDIUM
                    else:
                        ml_confidence = ConfidenceLevel.LOW
                
                label_quality = LabelQuality.UNLABELED
                if app_data.get('label_quality'):
                    quality_value = app_data['label_quality']
                    if quality_value == 'high':
                        label_quality = LabelQuality.HIGH
                    elif quality_value == 'medium':
                        label_quality = LabelQuality.MEDIUM
                    elif quality_value == 'low':
                        label_quality = LabelQuality.LOW
                    elif quality_value == 'predicted':
                        label_quality = LabelQuality.PREDICTED
                
                application = ApplicationWithPrediction(
                    application_id=app_data['application_id'],
                    candidate_id=app_data['candidate_id'],
                    offer_id=app_data['offer_id'],
                    fecha_postulacion=app_data.get('fecha_postulacion', datetime.now(timezone.utc)),
                    estado_original=app_data.get('estado', ''),
                    is_labeled=app_data.get('is_labeled', False),
                    label_quality=label_quality,
                    compatibility_features=compatibility_features,
                    ml_prediction=ml_prediction,
                    ml_probability=app_data.get('ml_probability'),
                    ml_confidence=ml_confidence,
                    candidate_name=candidate.get('nombre', ''),
                    candidate_email=candidate.get('email', ''),
                    candidate_experience_years=candidate.get('anos_experiencia', 0),
                    candidate_skills=candidate.get('habilidades', []),
                    offer_title=offer.get('titulo', ''),
                    offer_company=offer.get('empresa_nombre', ''),
                    offer_salary=offer.get('salario'),
                    offer_location=offer.get('ubicacion', '')
                )
                
                applications.append(application)
            
            # Calcular paginación
            total_pages = (total_count + pagination.page_size - 1) // pagination.page_size
            has_next_page = pagination.page < total_pages
            has_previous_page = pagination.page > 1
            
            return PaginatedApplications(
                applications=applications,
                total_count=total_count,
                page=pagination.page,
                page_size=pagination.page_size,
                total_pages=total_pages,
                has_next_page=has_next_page,
                has_previous_page=has_previous_page
            )
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo aplicaciones: {e}")
            raise
    
    async def get_active_model_info(self) -> Optional[ModelInfo]:
        """Obtener información del modelo activo"""
        logger.info("ℹ️ Obteniendo información del modelo activo...")
        
        try:
            db = await self._get_db()
            models_collection = db['ml_model_tracking']
            
            # Buscar modelo activo
            active_model = models_collection.find_one(
                {'is_active': True, 'model_type': 'semi_supervised'},
                sort=[('created_at', -1)]
            )
            
            if not active_model:
                return None
            
            # Obtener estadísticas del dataset
            dataset_stats = await self.get_dataset_statistics()
            
            # Convertir métricas
            metrics_data = active_model.get('metrics', {})
            metrics = ModelMetrics(
                train_accuracy=metrics_data.get('accuracy', 0.0),
                train_precision=metrics_data.get('precision', 0.0),
                train_recall=metrics_data.get('recall', 0.0),
                train_f1=metrics_data.get('f1_score', 0.0),
                val_roc_auc=metrics_data.get('roc_auc')
            )
            
            # Convertir algoritmo
            algorithm_str = active_model.get('algorithm', 'label_propagation')
            algorithm = SemiSupervisedAlgorithm(algorithm_str)
            
            return ModelInfo(
                model_id=active_model['model_id'],
                algorithm=algorithm,
                version=active_model.get('version', '1.0'),
                created_at=active_model.get('created_at', datetime.now(timezone.utc)),
                is_active=active_model['is_active'],
                performance_metrics=metrics,
                total_samples=dataset_stats.total_applications,
                labeled_samples=dataset_stats.labeled_applications,
                unlabeled_samples=dataset_stats.unlabeled_applications,
                labeled_ratio=dataset_stats.labeled_ratio,
                positive_samples=dataset_stats.accepted_applications,
                negative_samples=dataset_stats.rejected_applications,
                n_features=active_model.get('dataset_info', {}).get('total_features', 0),
                feature_categories=active_model.get('dataset_info', {}).get('feature_categories', {})
            )
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo info del modelo: {e}")
            return None
    
    async def predict_single_application(self, prediction_input: PredictionInput) -> SemiSupervisedPrediction:
        """Realizar predicción para una aplicación específica"""
        logger.info(f"🔮 Prediciendo aplicación: {prediction_input.candidate_id} -> {prediction_input.offer_id}")
        
        try:
            # Cargar modelo si no está cargado
            if not self.current_model or not self.current_model.is_trained:
                model_path = "trained_models/semi_supervised/label_propagation_model.pkl"
                if os.path.exists(model_path):
                    self.current_model = SemiSupervisedClassifier.load_model(model_path)
                else:
                    raise ValueError("No hay modelo entrenado disponible")
            
            # Obtener datos de la aplicación desde MongoDB
            db = await self._get_db()
            applications_collection = db['ml_applications']
            
            # Buscar aplicación existente o crear una temporal
            application = applications_collection.find_one({
                'candidate_id': prediction_input.candidate_id,
                'offer_id': prediction_input.offer_id
            })
            
            if not application:
                # Crear aplicación temporal para predicción
                application = {
                    'application_id': f"temp_{uuid.uuid4()}",
                    'candidate_id': prediction_input.candidate_id,
                    'offer_id': prediction_input.offer_id,
                    'fecha_postulacion': datetime.now(timezone.utc),
                    'estado': 'prediccion',
                    'is_labeled': False,
                    'compatibility_features': {
                        'overall_compatibility': prediction_input.override_compatibility_score or 0.5
                    }
                }
            
            # Preparar datos para predicción
            candidates_collection = db['ml_candidates']
            offers_collection = db['ml_job_offers']
            
            candidate = candidates_collection.find_one({'candidate_id': prediction_input.candidate_id})
            offer = offers_collection.find_one({'offer_id': prediction_input.offer_id})
            
            if not candidate or not offer:
                raise ValueError("Candidato u oferta no encontrados")
            
            # Crear DataFrames temporales
            applications_df = pd.DataFrame([application])
            candidates_df = pd.DataFrame([candidate])
            offers_df = pd.DataFrame([offer])
            
            # Transformar datos
            X = semi_supervised_preprocessor.transform(applications_df, candidates_df, offers_df)
            
            # Realizar predicción
            prediction, probability, confidence = self.current_model.predict_with_confidence(X)
            
            # Convertir resultados
            prediction_status = PredictionStatus.ACCEPTED if prediction[0] == 1 else PredictionStatus.REJECTED
            confidence_level = ConfidenceLevel(confidence[0])
            
            result = SemiSupervisedPrediction(
                application_id=application['application_id'],
                candidate_id=prediction_input.candidate_id,
                offer_id=prediction_input.offer_id,
                prediction=prediction_status,
                probability=float(probability[0]),
                confidence_level=confidence_level,
                compatibility_score=application['compatibility_features'].get('overall_compatibility', 0.0),
                predicted_at=datetime.now(timezone.utc),
                model_algorithm=SemiSupervisedAlgorithm(self.current_model.algorithm),
                model_version=self.current_model.model_info.get('version', '1.0')
            )
            
            logger.info(f"✅ Predicción completada: {prediction_status.name} (confianza: {confidence_level.name})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            raise
    
    async def validate_dataset(self) -> ValidationResult:
        """Validar calidad y consistencia del dataset"""
        logger.info("🔍 Validando dataset...")
        
        try:
            db = await self._get_db()
            
            errors = []
            warnings = []
            
            # Validar colecciones existen
            collections = db.list_collection_names()
            required_collections = ['ml_applications', 'ml_candidates', 'ml_job_offers']
            
            for collection in required_collections:
                if collection not in collections:
                    errors.append(f"Colección requerida '{collection}' no existe")
            
            if errors:
                return ValidationResult(
                    is_valid=False,
                    validation_errors=errors,
                    warnings=warnings,
                    data_quality_score=0.0,
                    recommendations=["Ejecutar migración de datos completa"]
                )
            
            # Validar datos
            applications_collection = db['ml_applications']
            candidates_collection = db['ml_candidates']
            offers_collection = db['ml_job_offers']
            
            total_apps = applications_collection.count_documents({})
            if total_apps == 0:
                errors.append("No hay aplicaciones en el dataset")
            
            labeled_apps = applications_collection.count_documents({'is_labeled': True})
            if labeled_apps < 50:
                warnings.append(f"Pocas aplicaciones etiquetadas ({labeled_apps}), se recomiendan al menos 50")
            
            # Validar integridad referencial
            app_sample = applications_collection.find().limit(100)
            missing_candidates = 0
            missing_offers = 0
            
            for app in app_sample:
                if not candidates_collection.find_one({'candidate_id': app['candidate_id']}):
                    missing_candidates += 1
                if not offers_collection.find_one({'offer_id': app['offer_id']}):
                    missing_offers += 1
            
            if missing_candidates > 0:
                warnings.append(f"Se encontraron {missing_candidates} referencias a candidatos faltantes")
            if missing_offers > 0:
                warnings.append(f"Se encontraron {missing_offers} referencias a ofertas faltantes")
            
            # Calcular score de calidad
            quality_factors = []
            quality_factors.append(min(labeled_apps / 100, 1.0))  # Al menos 100 etiquetadas = 100%
            quality_factors.append(1.0 if missing_candidates == 0 else 0.8)
            quality_factors.append(1.0 if missing_offers == 0 else 0.8)
            
            data_quality_score = sum(quality_factors) / len(quality_factors)
            
            # Recomendaciones
            recommendations = []
            if labeled_apps < 100:
                recommendations.append("Etiquetar más aplicaciones para mejorar el modelo")
            if missing_candidates > 0 or missing_offers > 0:
                recommendations.append("Limpiar referencias rotas en el dataset")
            if data_quality_score < 0.8:
                recommendations.append("Revisar calidad general de los datos")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                validation_errors=errors,
                warnings=warnings,
                data_quality_score=data_quality_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"❌ Error validando dataset: {e}")
            return ValidationResult(
                is_valid=False,
                validation_errors=[f"Error interno: {str(e)}"],
                warnings=[],
                data_quality_score=0.0,
                recommendations=["Revisar configuración de base de datos"]
            )


# Instancia global de resolvers
semi_supervised_resolvers = SemiSupervisedMLResolvers()