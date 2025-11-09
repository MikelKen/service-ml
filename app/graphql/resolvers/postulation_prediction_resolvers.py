#!/usr/bin/env python3
"""
🎯 RESOLVERS PARA PREDICCIÓN DE ESTADO DE POSTULACIONES
Resolvers GraphQL para el modelo semi-supervisado
"""

import logging
from typing import Dict, Any, List, Optional
import strawberry
from strawberry.types import Info

# Importar el predictor
from app.ml.models.postulation_status_predictor import (
    predict_postulation_status,
    predict_candidates_for_offer,
    get_predictor_status,
    reload_model
)

logger = logging.getLogger(__name__)

@strawberry.type
class PostulationPrediction:
    """Resultado de predicción de estado de postulación"""
    prediction: str
    prediction_numeric: int
    confidence: float
    prob_aceptado: float
    prob_rechazado: float
    recommendation: str
    candidate_id: Optional[str] = None
    offer_id: Optional[str] = None
    prediction_timestamp: Optional[str] = None

@strawberry.type
class CandidateRanking:
    """Candidato rankeado por compatibilidad con oferta"""
    ranking: int
    candidate_id: str
    prediction: str
    confidence: float
    prob_aceptado: float
    prob_rechazado: float
    recommendation: str

@strawberry.type
class PredictorStatus:
    """Estado del predictor de postulaciones"""
    is_loaded: bool
    features_count: int
    model_algorithm: Optional[str] = None
    base_classifier: Optional[str] = None
    loaded_at: Optional[str] = None

@strawberry.input
class CandidateInput:
    """Input de datos del candidato"""
    id: str
    anios_experiencia: int
    nivel_educacion: str
    habilidades: str
    idiomas: str
    certificaciones: str
    puesto_actual: str

@strawberry.input
class OfferInput:
    """Input de datos de la oferta"""
    id: str
    titulo: str
    salario: float
    ubicacion: str
    requisitos: str
    empresa_id: str

class PostulationPredictionResolvers:
    """Resolvers para predicciones de postulaciones"""
    
    @staticmethod
    async def predict_single_postulation(
        candidate: CandidateInput,
        offer: OfferInput,
        info: Info
    ) -> PostulationPrediction:
        """
        Predice el estado de una postulación individual
        """
        try:
            logger.info(f"🔮 Prediciendo postulación: candidato {candidate.id} -> oferta {offer.id}")
            
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
            
            # Convertir resultado a tipo GraphQL
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
            
            logger.info(f"✅ Predicción completada: {result['prediction']} (confianza: {result['confidence']:.1%})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            # Retornar predicción de error
            return PostulationPrediction(
                prediction="ERROR",
                prediction_numeric=-1,
                confidence=0.0,
                prob_aceptado=0.0,
                prob_rechazado=1.0,
                recommendation=f"Error en predicción: {str(e)}",
                candidate_id=candidate.id,
                offer_id=offer.id
            )
    
    @staticmethod
    async def rank_candidates_for_offer(
        candidates: List[CandidateInput],
        offer: OfferInput,
        info: Info
    ) -> List[CandidateRanking]:
        """
        Rankea múltiples candidatos para una oferta específica
        """
        try:
            logger.info(f"🏆 Rankeando {len(candidates)} candidatos para oferta {offer.id}")
            
            # Convertir inputs a diccionarios
            candidates_data = []
            for candidate in candidates:
                candidate_data = {
                    'id': candidate.id,
                    'aniosExperiencia': candidate.anios_experiencia,
                    'nivelEducacion': candidate.nivel_educacion,
                    'habilidades': candidate.habilidades,
                    'idiomas': candidate.idiomas,
                    'certificaciones': candidate.certificaciones,
                    'puestoActual': candidate.puesto_actual
                }
                candidates_data.append(candidate_data)
            
            offer_data = {
                'id': offer.id,
                'titulo': offer.titulo,
                'salario': offer.salario,
                'ubicacion': offer.ubicacion,
                'requisitos': offer.requisitos,
                'empresaId': offer.empresa_id
            }
            
            # Realizar predicciones y ranking
            predictions = predict_candidates_for_offer(candidates_data, offer_data)
            
            # Convertir resultados a tipos GraphQL
            rankings = []
            for pred in predictions:
                ranking = CandidateRanking(
                    ranking=pred['ranking'],
                    candidate_id=pred['candidate_id'],
                    prediction=pred['prediction'],
                    confidence=pred['confidence'],
                    prob_aceptado=pred['probabilities']['ACEPTADO'],
                    prob_rechazado=pred['probabilities']['RECHAZADO'],
                    recommendation=pred['recommendation']
                )
                rankings.append(ranking)
            
            logger.info(f"✅ Ranking completado: {len(rankings)} candidatos rankeados")
            
            return rankings
            
        except Exception as e:
            logger.error(f"❌ Error en ranking: {e}")
            return []
    
    @staticmethod
    async def get_predictor_status(info: Info) -> PredictorStatus:
        """
        Obtiene el estado actual del predictor
        """
        try:
            status = get_predictor_status()
            
            return PredictorStatus(
                is_loaded=status['is_loaded'],
                features_count=status['features_count'],
                model_algorithm=status['model_algorithm'],
                base_classifier=status['base_classifier'],
                loaded_at=status.get('model_info', {}).get('loaded_at')
            )
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado: {e}")
            return PredictorStatus(
                is_loaded=False,
                features_count=0
            )
    
    @staticmethod
    async def reload_predictor_model(info: Info) -> bool:
        """
        Recarga el modelo predictor
        """
        try:
            logger.info("🔄 Recargando modelo predictor...")
            reload_model()
            logger.info("✅ Modelo recargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recargando modelo: {e}")
            return False

# Resolver functions para usar en el schema principal
async def predict_postulation_status_resolver(
    candidate: CandidateInput,
    offer: OfferInput,
    info: Info
) -> PostulationPrediction:
    """Resolver para predicción individual"""
    return await PostulationPredictionResolvers.predict_single_postulation(candidate, offer, info)

async def rank_candidates_for_offer_resolver(
    candidates: List[CandidateInput],
    offer: OfferInput,
    info: Info
) -> List[CandidateRanking]:
    """Resolver para ranking de candidatos"""
    return await PostulationPredictionResolvers.rank_candidates_for_offer(candidates, offer, info)

async def get_predictor_status_resolver(info: Info) -> PredictorStatus:
    """Resolver para estado del predictor"""
    return await PostulationPredictionResolvers.get_predictor_status(info)

async def reload_predictor_model_resolver(info: Info) -> bool:
    """Resolver para recargar modelo"""
    return await PostulationPredictionResolvers.reload_predictor_model(info)