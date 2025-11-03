"""
Script para entrenar el modelo de compatibilidad candidato-oferta
"""
import asyncio
import sys
import os
import logging
from pathlib import Path

# Agregar el directorio app al path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.training.model_trainer import train_compatibility_model
from app.config.settings import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Función principal para entrenar el modelo"""
    
    logger.info("=== INICIANDO SCRIPT DE ENTRENAMIENTO ===")
    logger.info(f"Directorio de modelos: {settings.ml_models_path}")
    
    try:
        # Crear directorio de modelos si no existe
        os.makedirs(settings.ml_models_path, exist_ok=True)
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento del modelo de compatibilidad...")
        result = await asyncio.to_thread(train_compatibility_model)
        
        # Mostrar resultados
        logger.info("=== ENTRENAMIENTO COMPLETADO ===")
        logger.info(f"Mejor modelo: {result.get('best_model', 'N/A')}")
        
        best_metrics = result.get('best_metrics', {})
        if best_metrics:
            logger.info("Métricas del mejor modelo:")
            for metric, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        
        data_summary = result.get('data_summary', {})
        if data_summary:
            logger.info("Resumen de datos:")
            logger.info(f"  Total registros: {data_summary.get('total_rows', 'N/A')}")
            logger.info(f"  Features: {data_summary.get('total_columns', 'N/A')}")
            target_dist = data_summary.get('target_distribution', {})
            if target_dist:
                logger.info(f"  Distribución objetivo: {target_dist}")
        
        logger.info(f"Modelo guardado en: {result.get('model_path', 'N/A')}")
        logger.info(f"Preprocessor guardado en: {result.get('preprocessor_path', 'N/A')}")
        
        logger.info("¡Entrenamiento exitoso!")
        return 0
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        logger.exception("Detalles del error:")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)