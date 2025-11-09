"""
Script de entrenamiento paso a paso para el modelo semi-supervisado de postulaciones
"""
import asyncio
import logging
import os
import pandas as pd
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_semi_supervised.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Función principal de entrenamiento paso a paso"""
    
    print("=" * 60)
    print("ENTRENAMIENTO SEMI-SUPERVISADO DE POSTULACIONES")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # PASO 1: Verificar conexiones
        print("PASO 1: Verificando conexiones a bases de datos...")
        from app.config.connection import db, mongodb
        
        # Conectar a PostgreSQL
        postgres_connected = await db.connect()
        if postgres_connected:
            print("✅ PostgreSQL conectado exitosamente")
        else:
            print("❌ Error conectando a PostgreSQL")
            return
        
        # Conectar a MongoDB
        mongodb_connected = await mongodb.connect()
        if mongodb_connected:
            print("✅ MongoDB conectado exitosamente")
        else:
            print("❌ Error conectando a MongoDB")
            return
        
        # PASO 2: Extraer y analizar datos
        print("\nPASO 2: Extrayendo datos desde PostgreSQL...")
        from app.ml.data.postgres_extractor import postgres_extractor
        
        # Obtener estadísticas de estados
        stats = await postgres_extractor.get_estado_distribution()
        if stats is not None and not stats.empty:
            print("Distribución actual de estados:")
            for _, row in stats.iterrows():
                print(f"  - {row['estado']}: {row['cantidad']} registros")
        
        # Extraer dataset completo
        print("\nExtrayendo dataset completo...")
        complete_df = await postgres_extractor.extract_complete_dataset()
        
        if complete_df.empty:
            print("❌ No se pudieron extraer datos")
            return
        
        print(f"✅ Dataset extraído: {len(complete_df)} registros")
        print(f"Columnas disponibles: {len(complete_df.columns)}")
        
        # PASO 3: Preparar preprocesador
        print("\nPASO 3: Preparando preprocesador...")
        from app.ml.preprocessing.postulaciones_preprocessor import postulaciones_preprocessor
        
        # Separar datos para semi-supervisado
        labeled_data, unlabeled_data, complete_data = postulaciones_preprocessor.prepare_for_semi_supervised(complete_df)
        
        print(f"✅ Datos etiquetados: {len(labeled_data)}")
        print(f"✅ Datos sin etiquetar: {len(unlabeled_data)}")
        
        if not labeled_data.empty:
            print("Distribución de estados etiquetados:")
            for estado, count in labeled_data['estado'].value_counts().items():
                print(f"  - {estado}: {count}")
        
        if not unlabeled_data.empty:
            print("Estados sin etiquetar a predecir:")
            for estado, count in unlabeled_data['estado'].value_counts().items():
                print(f"  - {estado}: {count}")
        
        # PASO 4: Preprocesar características
        print("\nPASO 4: Preprocesando características...")
        processed_features = postulaciones_preprocessor.preprocess_features(
            complete_data, 
            fit_transformers=True
        )
        
        feature_count = processed_features.shape[1] - 4  # Excluyendo columnas objetivo
        print(f"✅ Características procesadas: {feature_count}")
        print(f"Forma de la matriz: {processed_features.shape}")
        
        # PASO 5: Entrenar modelos semi-supervisados
        print("\nPASO 5: Entrenando modelos semi-supervisados...")
        from app.ml.training.postulaciones_semi_supervised_trainer import postulaciones_trainer
        
        # Configurar datos en el trainer
        postulaciones_trainer.labeled_data = labeled_data
        postulaciones_trainer.unlabeled_data = unlabeled_data
        postulaciones_trainer.complete_data = complete_data
        postulaciones_trainer.processed_features = processed_features
        
        # Entrenar modelos
        model_results = postulaciones_trainer.train_semi_supervised_models()
        
        print("✅ Modelos entrenados:")
        for model_name, result in model_results.items():
            metrics = result['metrics']
            print(f"  - {model_name}:")
            print(f"    * Accuracy: {metrics['accuracy']:.4f}")
            print(f"    * F1-score (weighted): {metrics['f1_weighted']:.4f}")
            print(f"    * Precision (weighted): {metrics['precision_weighted']:.4f}")
            print(f"    * Recall (weighted): {metrics['recall_weighted']:.4f}")
        
        # PASO 6: Guardar predicciones
        print("\nPASO 6: Guardando predicciones en MongoDB...")
        await postulaciones_trainer.save_predictions_to_mongo()
        
        if hasattr(postulaciones_trainer, 'unlabeled_predictions') and not postulaciones_trainer.unlabeled_predictions.empty:
            predictions_df = postulaciones_trainer.unlabeled_predictions
            print(f"✅ Predicciones guardadas: {len(predictions_df)} registros")
            print("Distribución de predicciones:")
            for estado, count in predictions_df['estado_predicho'].value_counts().items():
                print(f"  - {estado}: {count}")
        else:
            print("⚠️  No hay predicciones para mostrar (todos los datos estaban etiquetados)")
            # Crear un DataFrame vacío para el resumen
            predictions_df = pd.DataFrame()
        
        # PASO 7: Guardar modelos
        print("\nPASO 7: Guardando modelos entrenados...")
        
        # Crear directorio para modelos
        models_dir = os.path.join("trained_models", "postulaciones")
        os.makedirs(models_dir, exist_ok=True)
        
        # Guardar modelo principal
        model_path = os.path.join(models_dir, "semi_supervised_model.pkl")
        postulaciones_trainer.save_model(model_path)
        
        # Guardar preprocesador
        preprocessor_path = os.path.join(models_dir, "semi_supervised_preprocessor.pkl")
        postulaciones_preprocessor.save_preprocessor(preprocessor_path)
        
        print(f"✅ Modelo guardado en: {model_path}")
        print(f"✅ Preprocesador guardado en: {preprocessor_path}")
        
        # PASO 8: Generar resumen de entrenamiento
        print("\nPASO 8: Generando resumen de entrenamiento...")
        
        training_summary = {
            'fecha_entrenamiento': datetime.now().isoformat(),
            'mejor_modelo': postulaciones_trainer.best_model_name,
            'metricas_mejor_modelo': postulaciones_trainer.model_metrics,
            'datos_resumen': {
                'total_registros': len(complete_df),
                'registros_etiquetados': len(labeled_data),
                'registros_sin_etiquetar': len(unlabeled_data),
                'caracteristicas': feature_count
            },
            'predicciones_realizadas': len(predictions_df) if not predictions_df.empty else 0,
            'rutas_archivos': {
                'modelo': model_path,
                'preprocesador': preprocessor_path
            }
        }
        
        # Guardar resumen
        import json
        summary_path = os.path.join(models_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resumen guardado en: {summary_path}")
        
        # PASO 9: Mostrar resumen final
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"Mejor modelo: {postulaciones_trainer.best_model_name}")
        print(f"Accuracy: {postulaciones_trainer.model_metrics['accuracy']:.4f}")
        print(f"F1-score (weighted): {postulaciones_trainer.model_metrics['f1_weighted']:.4f}")
        print(f"Predicciones realizadas: {training_summary['predicciones_realizadas']}")
        print(f"Archivos generados:")
        print(f"  - Modelo: {model_path}")
        print(f"  - Preprocesador: {preprocessor_path}")
        print(f"  - Resumen: {summary_path}")
        print("=" * 60)
        
        return training_summary
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        print(f"\n❌ Error durante el entrenamiento: {e}")
        return None
    
    finally:
        # Cerrar conexiones
        try:
            await db.disconnect()
            await mongodb.disconnect()
            print("\n✅ Conexiones cerradas")
        except:
            pass


if __name__ == "__main__":
    # Ejecutar entrenamiento
    result = asyncio.run(main())
    
    if result:
        print("\n🎉 Entrenamiento semi-supervisado completado exitosamente!")
    else:
        print("\n💥 El entrenamiento falló. Revise los logs para más detalles.")