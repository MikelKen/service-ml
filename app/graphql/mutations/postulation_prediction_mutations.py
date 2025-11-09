#!/usr/bin/env python3
"""
🎯 MUTATIONS PARA PREDICCIÓN DE POSTULACIONES
Mutations GraphQL para entrenar y gestionar el modelo semi-supervisado
"""

import logging
import strawberry
from strawberry.types import Info
from typing import List, Optional
import asyncio
import os
import sys

# Agregar path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..', '..')
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.graphql.types.postulation_prediction_types import (
    TrainModelResponse,
    ReloadModelResponse,
    TrainingConfig,
    PostulationPrediction,
    CandidateInput,
    OfferInput,
    BatchPredictionResult
)

logger = logging.getLogger(__name__)

class PostulationPredictionMutations:
    """Mutations para el sistema de predicción de postulaciones"""
    
    @staticmethod
    async def train_semi_supervised_model(
        config: Optional[TrainingConfig] = None,
        info: Info = None
    ) -> TrainModelResponse:
        """
        Entrena un nuevo modelo semi-supervisado
        """
        try:
            logger.info("🚀 Iniciando entrenamiento del modelo semi-supervisado")
            
            # Importar el entrenador
            from train_semi_supervised_step_by_step import SemiSupervisedTrainer
            
            # Configuración por defecto si no se proporciona
            if config is None:
                n_samples = 1000
                labeled_ratio = 0.3
            else:
                n_samples = config.n_samples
                labeled_ratio = config.labeled_ratio
            
            # Crear entrenador
            trainer = SemiSupervisedTrainer()
            
            # Ejecutar entrenamiento en un hilo separado para no bloquear
            # En una implementación real, esto se haría con Celery o similar
            results = trainer.run_complete_training(n_samples=n_samples)
            
            if results['success']:
                logger.info(f"✅ Entrenamiento completado: {results['best_model']}")
                
                return TrainModelResponse(
                    success=True,
                    message=f"Modelo entrenado exitosamente: {results['best_model']}",
                    model_path=results['model_path'],
                    training_summary=f"Mejor modelo: {results['best_model']}"
                )
            else:
                logger.error(f"❌ Error en entrenamiento: {results['error']}")
                
                return TrainModelResponse(
                    success=False,
                    message=f"Error en entrenamiento: {results['error']}"
                )
                
        except Exception as e:
            logger.error(f"❌ Error en mutation de entrenamiento: {e}")
            return TrainModelResponse(
                success=False,
                message=f"Error inesperado: {str(e)}"
            )
    
    @staticmethod
    async def reload_model(info: Info) -> ReloadModelResponse:
        """
        Recarga el modelo más reciente disponible
        """
        try:
            logger.info("🔄 Recargando modelo predictor...")
            
            from app.ml.models.postulation_status_predictor import reload_model, get_predictor_status
            
            # Recargar modelo
            reload_model()
            
            # Verificar estado
            status = get_predictor_status()
            
            if status['is_loaded']:
                return ReloadModelResponse(
                    success=True,
                    message="Modelo recargado exitosamente"
                )
            else:
                return ReloadModelResponse(
                    success=False,
                    message="No se pudo cargar el modelo"
                )
                
        except Exception as e:
            logger.error(f"❌ Error recargando modelo: {e}")
            return ReloadModelResponse(
                success=False,
                message=f"Error recargando modelo: {str(e)}"
            )
    
    @staticmethod
    async def predict_batch_postulations(
        candidates: List[CandidateInput],
        offers: List[OfferInput],
        info: Info
    ) -> BatchPredictionResult:
        """
        Realiza predicciones en lote para múltiples combinaciones candidato-oferta
        """
        try:
            logger.info(f"📊 Ejecutando predicciones en lote: {len(candidates)} candidatos x {len(offers)} ofertas")
            
            from app.ml.models.postulation_status_predictor import predict_postulation_status
            
            predictions = []
            errors = []
            successful = 0
            failed = 0
            
            # Predecir todas las combinaciones
            for candidate in candidates:
                for offer in offers:
                    try:
                        # Convertir inputs a diccionarios
                        candidate_data = {
                            'id': candidate.id,
                            'aniosExperiencia': candidate.anios_experiencia,
                            'nivelEducacion': candidate.nivel_educacion,
                            'habilidades': candidate.habilidades,
                            'idiomas': candidate.idiomas,
                            'certificaciones': candidate.certificaciones,
                            'puestoActual': candidate.puesto_actual
                        }
                        
                        offer_data = {
                            'id': offer.id,
                            'titulo': offer.titulo,
                            'salario': offer.salario,
                            'ubicacion': offer.ubicacion,
                            'requisitos': offer.requisitos,
                            'empresaId': offer.empresa_id
                        }
                        
                        # Realizar predicción
                        result = predict_postulation_status(candidate_data, offer_data)
                        
                        # Convertir a tipo GraphQL
                        prediction = PostulationPrediction(
                            prediction=result['prediction'],
                            prediction_numeric=result['prediction_numeric'],
                            confidence=result['confidence'],
                            prob_aceptado=result['probabilities']['ACEPTADO'],
                            prob_rechazado=result['probabilities']['RECHAZADO'],
                            recommendation=result['recommendation'],
                            candidate_id=result.get('candidate_id'),
                            offer_id=result.get('offer_id'),
                            prediction_timestamp=result.get('prediction_timestamp')
                        )
                        
                        predictions.append(prediction)
                        successful += 1
                        
                    except Exception as e:
                        error_msg = f"Error prediciendo candidato {candidate.id} -> oferta {offer.id}: {str(e)}"
                        errors.append(error_msg)
                        failed += 1
                        logger.error(error_msg)
            
            total = successful + failed
            
            logger.info(f"✅ Predicciones en lote completadas: {successful}/{total} exitosas")
            
            return BatchPredictionResult(
                total_predictions=total,
                successful_predictions=successful,
                failed_predictions=failed,
                predictions=predictions,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"❌ Error en predicciones en lote: {e}")
            return BatchPredictionResult(
                total_predictions=0,
                successful_predictions=0,
                failed_predictions=0,
                predictions=[],
                errors=[f"Error general: {str(e)}"]
            )

# Resolver functions para usar en el schema principal
async def train_semi_supervised_model_mutation(
    config: Optional[TrainingConfig] = None,
    info: Info = None
) -> TrainModelResponse:
    """Mutation para entrenar modelo"""
    return await PostulationPredictionMutations.train_semi_supervised_model(config, info)

async def reload_model_mutation(info: Info) -> ReloadModelResponse:
    """Mutation para recargar modelo"""
    return await PostulationPredictionMutations.reload_model(info)

async def predict_batch_postulations_mutation(
    candidates: List[CandidateInput],
    offers: List[OfferInput],
    info: Info
) -> BatchPredictionResult:
    """Mutation para predicciones en lote"""
    return await PostulationPredictionMutations.predict_batch_postulations(candidates, offers, info)