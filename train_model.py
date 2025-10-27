"""
Script principal para entrenar el modelo de predicción de contratación
"""
import os
import sys
import logging
from datetime import datetime

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.preprocessing import preprocess_data
from ml.features.feature_engineering import FeatureEngineer
from ml.models.trainer import train_hiring_prediction_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Función principal para entrenar el modelo"""
    logger.info("=== Iniciando entrenamiento del modelo de predicción de contratación ===")
    
    # Configuración de paths
    data_path = "postulaciones_sinteticas_500.csv"
    model_output_path = "trained_models/hiring_prediction_model.pkl"
    transformers_output_path = "trained_models/feature_transformers.pkl"
    
    try:
        # 1. Cargar y preprocessar datos
        logger.info("Paso 1: Cargando y preprocessando datos...")
        df_processed, summary = preprocess_data(data_path)
        
        logger.info(f"Datos cargados: {summary['total_rows']} filas, {summary['total_columns']} columnas")
        logger.info(f"Distribución del target: {summary['target_distribution']}")
        
        # 2. Crear feature engineer
        logger.info("Paso 2: Creando features...")
        feature_engineer = FeatureEngineer()
        
        # 3. Entrenar modelo
        logger.info("Paso 3: Entrenando modelos...")
        trainer = train_hiring_prediction_model(df_processed, feature_engineer)
        
        # 4. Mostrar resultados
        logger.info("=== Resultados del entrenamiento ===")
        logger.info(f"Mejor modelo: {trainer.best_model_name}")
        logger.info(f"ROC AUC Score: {trainer.best_score:.4f}")
        
        # Mostrar métricas de todos los modelos
        for model_name, results in trainer.evaluation_results.items():
            eval_results = results['evaluation']
            logger.info(f"{model_name}:")
            logger.info(f"  - ROC AUC: {eval_results['roc_auc']:.4f}")
            logger.info(f"  - PR AUC: {eval_results['pr_auc']:.4f}")
            logger.info(f"  - F1 Score: {eval_results['f1_score']:.4f}")
            logger.info(f"  - Precision: {eval_results['precision']:.4f}")
            logger.info(f"  - Recall: {eval_results['recall']:.4f}")
        
        # 5. Guardar modelo y transformadores
        logger.info("Paso 4: Guardando modelo y artefactos...")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        
        # Guardar modelo completo
        trainer.save_model(model_output_path, include_feature_engineer=True)
        
        # Guardar transformadores por separado
        feature_engineer.save_transformers(transformers_output_path)
        
        # 6. Generar visualizaciones
        try:
            logger.info("Paso 5: Generando visualizaciones...")
            plot_path = "trained_models/model_evaluation_plots.png"
            trainer.plot_evaluation_metrics(save_path=plot_path)
        except Exception as e:
            logger.warning(f"No se pudieron generar gráficos: {e}")
        
        # 7. Mostrar importancia de features
        try:
            logger.info("Paso 6: Analizando importancia de features...")
            feature_names = feature_engineer.get_feature_importance_names()
            importance_df = trainer.get_feature_importance(feature_names)
            
            logger.info("Top 10 features más importantes:")
            for i, row in importance_df.head(10).iterrows():
                logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
                
        except Exception as e:
            logger.warning(f"No se pudo analizar importancia de features: {e}")
        
        # 8. Ejemplo de predicción
        logger.info("Paso 7: Probando predicción de ejemplo...")
        
        # Usar una muestra del dataset para probar
        sample_data = df_processed.iloc[0].to_dict()
        
        # Remover el target para la predicción
        sample_data.pop('contactado', None)
        
        # Crear predictor y probar
        from ml.models.predictor import HiringPredictor
        
        predictor = HiringPredictor(model_output_path)
        result = predictor.predict_single(sample_data)
        
        logger.info("Ejemplo de predicción:")
        logger.info(f"  - Probabilidad: {result['probability']:.3f}")
        logger.info(f"  - Predicción: {'Contactar' if result['prediction'] == 1 else 'No contactar'}")
        logger.info(f"  - Confianza: {result['confidence_level']}")
        logger.info(f"  - Recomendación: {result['recommendation']}")
        
        logger.info("=== Entrenamiento completado exitosamente ===")
        logger.info(f"Modelo guardado en: {model_output_path}")
        logger.info(f"Transformadores guardados en: {transformers_output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Entrenamiento completado exitosamente!")
        print("El modelo está listo para ser usado en el microservicio.")
    else:
        print("\n❌ Error durante el entrenamiento.")
        sys.exit(1)