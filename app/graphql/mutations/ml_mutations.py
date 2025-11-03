"""
Mutaciones de GraphQL para operaciones de Machine Learning
"""
import strawberry
from typing import Optional
import logging

from app.graphql.types.ml_types import (
    ModelTrainingResult, TrainingConfigInput
)
from app.graphql.resolvers.ml_resolvers import train_model

logger = logging.getLogger(__name__)


@strawberry.type
class MLMutation:
    """Mutaciones relacionadas con Machine Learning"""
    
    @strawberry.mutation
    async def train_compatibility_model(
        self, 
        config: Optional[TrainingConfigInput] = None
    ) -> ModelTrainingResult:
        """
        Entrena un nuevo modelo de compatibilidad candidato-oferta
        
        Args:
            config: Configuración opcional para el entrenamiento
            
        Returns:
            Resultado del entrenamiento con métricas y estado
        """
        logger.info("Iniciando entrenamiento de modelo desde mutación GraphQL")
        return await train_model(config)
    
    @strawberry.mutation
    async def retrain_model(self) -> ModelTrainingResult:
        """
        Re-entrena el modelo con los datos más recientes
        
        Returns:
            Resultado del re-entrenamiento
        """
        logger.info("Re-entrenando modelo con datos actuales")
        return await train_model()