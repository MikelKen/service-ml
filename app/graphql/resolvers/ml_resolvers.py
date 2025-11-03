"""
Resolvers de GraphQL para operaciones de Machine Learning
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import time

from app.graphql.types.ml_types import (
    CompatibilityPredictionInput, BatchCompatibilityInput, TopCandidatesInput,
    CompatibilityPrediction, BatchCompatibilityResult, ModelTrainingResult,
    ModelInfo, FeatureImportance, ModelFeatureImportance, PredictionExplanation,
    TrainingDataSummary, ModelPerformanceMetrics, TrainingConfigInput,
    TrainingMetrics, DataSummary, ModelMetrics, PredictionFactors, MetadataInfo
)

from app.ml.models.predictor import compatibility_predictor
from app.ml.training.model_trainer import compatibility_trainer, train_compatibility_model
from app.ml.data.data_extractor import data_extractor

logger = logging.getLogger(__name__)


async def predict_compatibility(input_data: CompatibilityPredictionInput) -> CompatibilityPrediction:
    """Predice compatibilidad entre un candidato y una oferta"""
    
    try:
        # Realizar predicción
        result = compatibility_predictor.predict_compatibility(
            input_data.candidate_id, 
            input_data.offer_id
        )
        
        return CompatibilityPrediction(
            candidate_id=result['candidate_id'],
            offer_id=result['offer_id'],
            probability=result['probability'],
            prediction=result['prediction'],
            confidence=result['confidence'],
            model_used=result.get('model_used'),
            prediction_date=result.get('prediction_date'),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error en predicción de compatibilidad: {e}")
        return CompatibilityPrediction(
            candidate_id=input_data.candidate_id,
            offer_id=input_data.offer_id,
            probability=0.0,
            prediction=False,
            confidence='Error',
            error=str(e)
        )


async def predict_batch_compatibility(input_data: BatchCompatibilityInput) -> BatchCompatibilityResult:
    """Predice compatibilidad para múltiples pares candidato-oferta"""
    
    start_time = time.time()
    predictions = []
    success_count = 0
    error_count = 0
    
    try:
        for pair in input_data.pairs:
            try:
                result = compatibility_predictor.predict_compatibility(
                    pair.candidate_id, 
                    pair.offer_id
                )
                
                prediction = CompatibilityPrediction(
                    candidate_id=result['candidate_id'],
                    offer_id=result['offer_id'],
                    probability=result['probability'],
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    model_used=result.get('model_used'),
                    prediction_date=result.get('prediction_date'),
                    error=result.get('error')
                )
                
                predictions.append(prediction)
                
                if result.get('error'):
                    error_count += 1
                else:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error procesando par {pair.candidate_id}-{pair.offer_id}: {e}")
                predictions.append(CompatibilityPrediction(
                    candidate_id=pair.candidate_id,
                    offer_id=pair.offer_id,
                    probability=0.0,
                    prediction=False,
                    confidence='Error',
                    error=str(e)
                ))
                error_count += 1
        
        processing_time = time.time() - start_time
        
        return BatchCompatibilityResult(
            predictions=predictions,
            total_processed=len(input_data.pairs),
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        return BatchCompatibilityResult(
            predictions=[],
            total_processed=0,
            success_count=0,
            error_count=len(input_data.pairs),
            processing_time=time.time() - start_time
        )


async def get_top_candidates_for_offer(input_data: TopCandidatesInput) -> List[CompatibilityPrediction]:
    """Obtiene los mejores candidatos para una oferta específica"""
    
    try:
        results = compatibility_predictor.predict_candidates_for_offer(
            input_data.offer_id, 
            input_data.top_n
        )
        
        predictions = []
        for result in results:
            prediction = CompatibilityPrediction(
                candidate_id=result['candidate_id'],
                offer_id=result['offer_id'],
                probability=result['probability'],
                prediction=result['prediction'],
                confidence=result['confidence'],
                ranking=result.get('ranking')
            )
            predictions.append(prediction)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error obteniendo top candidatos para oferta {input_data.offer_id}: {e}")
        return []


async def train_model(config: Optional[TrainingConfigInput] = None) -> ModelTrainingResult:
    """Entrena un nuevo modelo de compatibilidad"""
    
    start_time = time.time()
    
    try:
        logger.info("Iniciando entrenamiento de modelo desde GraphQL")
        
        # Ejecutar entrenamiento
        training_result = await asyncio.to_thread(train_compatibility_model)
        
        training_time = time.time() - start_time
        
        # Convertir métricas a tipos específicos
        metrics = None
        if training_result.get('best_metrics'):
            metrics_dict = training_result['best_metrics']
            metrics = TrainingMetrics(
                accuracy=metrics_dict.get('accuracy'),
                precision=metrics_dict.get('precision'),
                recall=metrics_dict.get('recall'),
                f1_score=metrics_dict.get('f1_score'),
                roc_auc=metrics_dict.get('roc_auc')
            )
        
        # Convertir resumen de datos
        data_summary = None
        if training_result.get('data_summary'):
            summary_dict = training_result['data_summary']
            data_summary = DataSummary(
                total_samples=summary_dict.get('total_samples'),
                positive_samples=summary_dict.get('positive_samples'),
                negative_samples=summary_dict.get('negative_samples'),
                features_count=summary_dict.get('features_count')
            )
        
        return ModelTrainingResult(
            success=True,
            message="Modelo entrenado exitosamente",
            best_model=training_result.get('best_model'),
            metrics=metrics,
            training_time=training_time,
            data_summary=data_summary
        )
        
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        return ModelTrainingResult(
            success=False,
            message=f"Error en entrenamiento: {str(e)}",
            training_time=time.time() - start_time
        )


async def get_model_info() -> ModelInfo:
    """Obtiene información del modelo actual"""
    
    try:
        info = compatibility_predictor.get_model_info()
        
        # Convertir métricas a tipo específico
        metrics = None
        if info.get('metrics'):
            metrics_dict = info['metrics']
            metrics = ModelMetrics(
                accuracy=metrics_dict.get('accuracy'),
                precision=metrics_dict.get('precision'),
                recall=metrics_dict.get('recall'),
                f1_score=metrics_dict.get('f1_score')
            )
        
        return ModelInfo(
            model_name=info.get('model_name'),
            model_type=info.get('model_type'),
            is_loaded=info.get('is_loaded', False),
            metrics=metrics,
            feature_importance_count=info.get('feature_importance_count'),
            top_features=info.get('top_features')
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo información del modelo: {e}")
        return ModelInfo(
            is_loaded=False
        )


async def get_feature_importance(top_n: Optional[int] = 20) -> ModelFeatureImportance:
    """Obtiene la importancia de features del modelo"""
    
    try:
        importance_dict = compatibility_predictor.get_feature_importance(top_n or 20)
        
        features = [
            FeatureImportance(feature_name=name, importance=importance)
            for name, importance in importance_dict.items()
        ]
        
        return ModelFeatureImportance(
            features=features,
            total_features=len(features)
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo importancia de features: {e}")
        return ModelFeatureImportance(
            features=[],
            total_features=0
        )


async def explain_prediction(candidate_id: str, offer_id: str) -> PredictionExplanation:
    """Proporciona explicación detallada de una predicción"""
    
    try:
        explanation = compatibility_predictor.explain_prediction(candidate_id, offer_id)
        
        # Convertir prediction
        pred_data = explanation.get('prediction', {})
        prediction = CompatibilityPrediction(
            candidate_id=pred_data.get('candidate_id', candidate_id),
            offer_id=pred_data.get('offer_id', offer_id),
            probability=pred_data.get('probability', 0.0),
            prediction=pred_data.get('prediction', False),
            confidence=pred_data.get('confidence', 'Unknown')
        )
        
        # Convertir feature importance
        feature_imp_dict = explanation.get('feature_importance', {})
        feature_importance = [
            FeatureImportance(feature_name=name, importance=importance)
            for name, importance in feature_imp_dict.items()
        ]
        
        # Convertir key factors
        factors_dict = explanation.get('key_factors', {})
        key_factors = PredictionFactors(
            experience_match=factors_dict.get('experience_match'),
            skills_overlap=factors_dict.get('skills_overlap'),
            education_fit=factors_dict.get('education_fit'),
            location_match=factors_dict.get('location_match')
        )
        
        return PredictionExplanation(
            prediction=prediction,
            key_factors=key_factors,
            feature_importance=feature_importance,
            recommendation=explanation.get('recommendation', 'No hay recomendación disponible')
        )
        
    except Exception as e:
        logger.error(f"Error explicando predicción: {e}")
        # Retornar explicación básica
        prediction = CompatibilityPrediction(
            candidate_id=candidate_id,
            offer_id=offer_id,
            probability=0.0,
            prediction=False,
            confidence='Error',
            error=str(e)
        )
        
        return PredictionExplanation(
            prediction=prediction,
            key_factors=PredictionFactors(),
            feature_importance=[],
            recommendation=f"Error generando explicación: {str(e)}"
        )


async def get_training_data_summary() -> TrainingDataSummary:
    """Obtiene resumen de los datos de entrenamiento disponibles"""
    
    try:
        # Extraer datos para obtener estadísticas
        training_data = data_extractor.create_training_dataset()
        
        if training_data.empty:
            return TrainingDataSummary(
                total_records=0,
                positive_samples=0,
                negative_samples=0,
                features_count=0
            )
        
        target_counts = training_data['target'].value_counts().to_dict()
        
        return TrainingDataSummary(
            total_records=len(training_data),
            positive_samples=target_counts.get(1, 0),
            negative_samples=target_counts.get(0, 0),
            features_count=len(training_data.columns) - 1  # Excluir target
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo resumen de datos: {e}")
        return TrainingDataSummary(
            total_records=0,
            positive_samples=0,
            negative_samples=0,
            features_count=0
        )


async def get_model_performance() -> ModelPerformanceMetrics:
    """Obtiene métricas de rendimiento del modelo actual"""
    
    try:
        info = compatibility_predictor.get_model_info()
        metrics = info.get('metrics', {})
        
        return ModelPerformanceMetrics(
            accuracy=metrics.get('accuracy'),
            precision=metrics.get('precision'),
            recall=metrics.get('recall'),
            f1_score=metrics.get('f1_score'),
            roc_auc=metrics.get('roc_auc'),
            confusion_matrix=metrics.get('confusion_matrix')
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas de rendimiento: {e}")
        return ModelPerformanceMetrics()


# Funciones auxiliares para verificar estado
async def is_model_loaded() -> bool:
    """Verifica si hay un modelo cargado"""
    return compatibility_predictor.is_loaded


async def get_model_status() -> Dict[str, Any]:
    """Obtiene estado completo del sistema ML"""
    
    try:
        info = compatibility_predictor.get_model_info()
        
        # Verificar datos disponibles
        candidates_count = 0
        offers_count = 0
        companies_count = 0
        
        try:
            candidates_df = data_extractor.extract_candidates()
            offers_df = data_extractor.extract_job_offers()
            companies_df = data_extractor.extract_companies()
            
            candidates_count = len(candidates_df)
            offers_count = len(offers_df)
            companies_count = len(companies_df)
            
        except Exception as e:
            logger.warning(f"Error verificando datos: {e}")
        
        return {
            'model_loaded': info.get('is_loaded', False),
            'model_name': info.get('model_name'),
            'model_type': info.get('model_type'),
            'preprocessor_fitted': info.get('preprocessor_fitted', False),
            'data_available': {
                'candidates': candidates_count,
                'offers': offers_count,
                'companies': companies_count
            },
            'system_ready': (
                info.get('is_loaded', False) and 
                info.get('preprocessor_fitted', False) and 
                candidates_count > 0 and 
                offers_count > 0
            )
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        return {
            'model_loaded': False,
            'system_ready': False,
            'error': str(e)
        }