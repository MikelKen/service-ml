#!/usr/bin/env python3
"""
Script de entrenamiento paso a paso para modelo semi-supervisado de postulaciones

Este script entrena un modelo semi-supervisado que puede predecir el estado de postulaciones
utilizando tanto datos etiquetados como no etiquetados.

Uso:
    python train_semi_supervised_step_by_step.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.training.semi_supervised_trainer import semi_supervised_trainer
from app.ml.data.postgres_extractor import postgres_extractor
from app.config.connection import init_database, close_database

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/semi_supervised_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def paso_1_verificar_conexiones():
    """Paso 1: Verificar conexiones a bases de datos"""
    logger.info("="*60)
    logger.info("PASO 1: Verificando conexiones a bases de datos")
    logger.info("="*60)
    
    try:
        # Inicializar conexiones
        success = await init_database()
        
        if success:
            logger.info("✅ Conexiones establecidas exitosamente")
            
            # Verificar datos en PostgreSQL
            stats = await postgres_extractor.get_table_stats()
            logger.info(f"📊 Estadísticas de datos: {stats}")
            
            return True
        else:
            logger.error("❌ Error estableciendo conexiones")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en verificación de conexiones: {str(e)}")
        return False

async def paso_2_analizar_datos():
    """Paso 2: Analizar y extraer datos de PostgreSQL"""
    logger.info("="*60)
    logger.info("PASO 2: Analizando datos de PostgreSQL")
    logger.info("="*60)
    
    try:
        # Extraer distribución de estados
        distribucion = await postgres_extractor.extract_estado_distribution()
        logger.info(f"📈 Distribución de estados: {distribucion}")
        
        # Extraer datos completos
        logger.info("📤 Extrayendo datos completos de postulaciones...")
        df = await postgres_extractor.extract_postulaciones_with_features()
        
        if df.empty:
            logger.error("❌ No se encontraron datos para extraer")
            return None
        
        logger.info(f"✅ Datos extraídos exitosamente: {df.shape}")
        logger.info(f"📝 Columnas disponibles: {list(df.columns)}")
        
        # Analizar calidad de datos
        logger.info("🔍 Analizando calidad de datos:")
        logger.info(f"  - Valores nulos por columna:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.info(f"    {col}: {null_count} ({(null_count/len(df))*100:.1f}%)")
        
        # Estadísticas específicas para semi-supervisado
        labeled_count = df['estado'].notna().sum()
        unlabeled_count = len(df) - labeled_count
        
        logger.info(f"📊 Datos para semi-supervisado:")
        logger.info(f"  - Muestras etiquetadas: {labeled_count}")
        logger.info(f"  - Muestras no etiquetadas: {unlabeled_count}")
        logger.info(f"  - Porcentaje etiquetado: {(labeled_count/len(df))*100:.1f}%")
        
        if labeled_count < 5:
            logger.warning("⚠️  Advertencia: Muy pocas muestras etiquetadas para entrenamiento robusto")
        
        if unlabeled_count == 0:
            logger.warning("⚠️  Advertencia: No hay datos no etiquetados para aprendizaje semi-supervisado")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error extrayendo datos: {str(e)}")
        return None

async def paso_3_preprocesar_datos(df):
    """Paso 3: Preprocesar datos para modelo semi-supervisado"""
    logger.info("="*60)
    logger.info("PASO 3: Preprocesando datos")
    logger.info("="*60)
    
    try:
        logger.info("🔧 Iniciando preprocesamiento...")
        
        # Inicializar preprocesador
        preprocessor = semi_supervised_trainer.preprocessor
        
        # Mostrar características que se van a procesar
        logger.info(f"📝 Características de texto: {preprocessor.text_features}")
        logger.info(f"📝 Características categóricas: {preprocessor.categorical_features}")
        logger.info(f"📝 Características numéricas: {preprocessor.numerical_features}")
        
        # Ejecutar preprocesamiento
        X_labeled, y_labeled, X_unlabeled = preprocessor.fit_transform(df, 'estado')
        
        logger.info(f"✅ Preprocesamiento completado:")
        logger.info(f"  - Datos etiquetados: {X_labeled.shape}")
        logger.info(f"  - Etiquetas: {len(y_labeled)} clases únicas: {len(np.unique(y_labeled)) if len(y_labeled) > 0 else 0}")
        logger.info(f"  - Datos no etiquetados: {X_unlabeled.shape}")
        logger.info(f"  - Total de características: {X_labeled.shape[1] if len(X_labeled) > 0 else X_unlabeled.shape[1] if len(X_unlabeled) > 0 else 0}")
        
        # Mapeo de estados
        if preprocessor.estado_mapping:
            logger.info(f"🏷️  Mapeo de estados: {preprocessor.estado_mapping}")
        
        # Guardar preprocesador
        preprocessor_path = "trained_models/semi_supervised/preprocessor.pkl"
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        preprocessor.save(preprocessor_path)
        logger.info(f"💾 Preprocesador guardado en: {preprocessor_path}")
        
        return X_labeled, y_labeled, X_unlabeled
        
    except Exception as e:
        logger.error(f"❌ Error en preprocesamiento: {str(e)}")
        return None, None, None

async def paso_4_entrenar_modelos(X_labeled, y_labeled, X_unlabeled):
    """Paso 4: Entrenar diferentes tipos de modelos semi-supervisados"""
    logger.info("="*60)
    logger.info("PASO 4: Entrenando modelos semi-supervisados")
    logger.info("="*60)
    
    try:
        # Verificar que tenemos datos para entrenar
        if len(X_labeled) == 0:
            logger.error("❌ No hay datos etiquetados para entrenar")
            return None
        
        logger.info(f"🤖 Iniciando entrenamiento con:")
        logger.info(f"  - Muestras etiquetadas: {len(X_labeled)}")
        logger.info(f"  - Muestras no etiquetadas: {len(X_unlabeled)}")
        logger.info(f"  - Características: {X_labeled.shape[1]}")
        
        # Entrenar todos los modelos
        training_summary = await semi_supervised_trainer.train_all_models(save_to_mongo=True)
        
        logger.info("✅ Entrenamiento completado para todos los modelos")
        
        # Mostrar resultados
        logger.info("📊 Resumen de entrenamiento:")
        for model_type, results in training_summary['training_results'].items():
            if 'error' not in results:
                logger.info(f"  {model_type}:")
                logger.info(f"    - Precisión entrenamiento: {results.get('train_accuracy', 'N/A'):.4f}")
                logger.info(f"    - Precisión validación: {results.get('val_accuracy', 'N/A')}")
                logger.info(f"    - Predicciones no etiquetadas: {results.get('total_predictions', 'N/A')}")
                logger.info(f"    - Confianza promedio: {results.get('prediction_confidence', {}).get('mean', 'N/A')}")
            else:
                logger.error(f"  {model_type}: ERROR - {results['error']}")
        
        # Mejor modelo
        best_model = training_summary['best_model']
        if best_model['model_type']:
            logger.info(f"🏆 Mejor modelo: {best_model['model_type']} (precisión: {best_model['score']:.4f})")
        
        return training_summary
        
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento: {str(e)}")
        return None

async def paso_5_evaluar_predicciones(training_summary):
    """Paso 5: Evaluar predicciones en datos no etiquetados"""
    logger.info("="*60)
    logger.info("PASO 5: Evaluando predicciones")
    logger.info("="*60)
    
    try:
        if not training_summary or not training_summary['best_model']['model_type']:
            logger.error("❌ No hay modelo entrenado para evaluar")
            return False
        
        best_model_type = training_summary['best_model']['model_type']
        logger.info(f"🔍 Evaluando modelo: {best_model_type}")
        
        # Obtener resultados del mejor modelo
        best_results = training_summary['training_results'][best_model_type]
        
        logger.info("📈 Métricas de evaluación:")
        logger.info(f"  - Muestras predichas: {best_results.get('total_predictions', 0)}")
        logger.info(f"  - Distribución de predicciones: {best_results.get('unlabeled_predictions_distribution', {})}")
        
        if 'prediction_confidence' in best_results:
            conf_stats = best_results['prediction_confidence']
            logger.info(f"  - Confianza promedio: {conf_stats.get('mean', 0):.3f}")
            logger.info(f"  - Confianza mínima: {conf_stats.get('min', 0):.3f}")
            logger.info(f"  - Confianza máxima: {conf_stats.get('max', 0):.3f}")
        
        # Verificar archivos generados
        files_generated = training_summary.get('files_generated', {})
        logger.info("📁 Archivos generados:")
        logger.info(f"  - Preprocesador: {files_generated.get('preprocessor', 'N/A')}")
        
        if 'models' in files_generated:
            for model_type, path in files_generated['models'].items():
                logger.info(f"  - Modelo {model_type}: {path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error evaluando predicciones: {str(e)}")
        return False

async def paso_6_generar_predicciones_finales():
    """Paso 6: Generar predicciones para datos sin estado"""
    logger.info("="*60)
    logger.info("PASO 6: Generando predicciones finales")
    logger.info("="*60)
    
    try:
        # Extraer datos sin estado
        df_sin_estado = await postgres_extractor.extract_missing_estado_postulaciones()
        
        if df_sin_estado.empty:
            logger.info("ℹ️  No hay postulaciones sin estado para predecir")
            return True
        
        logger.info(f"📤 Encontradas {len(df_sin_estado)} postulaciones sin estado")
        
        # Cargar el mejor modelo (si existe)
        try:
            model_files = [f for f in os.listdir("trained_models/semi_supervised") if f.endswith("_model.pkl")]
            if not model_files:
                logger.warning("⚠️  No se encontraron modelos entrenados")
                return False
            
            # Usar el primer modelo disponible
            model_file = model_files[0]
            model_type = model_file.replace("_model.pkl", "")
            
            logger.info(f"🤖 Cargando modelo: {model_type}")
            model = semi_supervised_trainer.load_trained_model(model_type)
            
            # Cargar preprocesador
            preprocessor_path = "trained_models/semi_supervised/preprocessor.pkl"
            semi_supervised_trainer.preprocessor.load(preprocessor_path)
            
            # Procesar datos
            X_predict = semi_supervised_trainer.preprocessor.transform(df_sin_estado)
            
            # Realizar predicciones
            predictions = model.predict(X_predict)
            confidence_scores = model.get_prediction_confidence(X_predict)
            
            # Convertir a etiquetas
            label_encoder = semi_supervised_trainer.preprocessor.label_encoder
            predicted_labels = label_encoder.inverse_transform(predictions)
            
            # Crear resultado
            df_sin_estado['predicted_estado'] = predicted_labels
            df_sin_estado['prediction_confidence'] = confidence_scores
            df_sin_estado['prediction_timestamp'] = datetime.now().isoformat()
            
            # Guardar resultados
            output_path = f"trained_models/semi_supervised/final_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_sin_estado.to_csv(output_path, index=False)
            
            logger.info(f"✅ Predicciones guardadas en: {output_path}")
            logger.info(f"📊 Distribución de predicciones: {dict(zip(*np.unique(predicted_labels, return_counts=True)))}")
            
            # Estadísticas de confianza
            high_confidence = np.sum(confidence_scores > 0.8)
            low_confidence = np.sum(confidence_scores < 0.6)
            
            logger.info(f"📈 Estadísticas de confianza:")
            logger.info(f"  - Alta confianza (>0.8): {high_confidence} ({(high_confidence/len(confidence_scores))*100:.1f}%)")
            logger.info(f"  - Baja confianza (<0.6): {low_confidence} ({(low_confidence/len(confidence_scores))*100:.1f}%)")
            logger.info(f"  - Confianza promedio: {np.mean(confidence_scores):.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generando predicciones: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error en paso de predicciones finales: {str(e)}")
        return False

async def main():
    """Función principal que ejecuta todo el proceso paso a paso"""
    start_time = datetime.now()
    
    logger.info("🚀 INICIANDO ENTRENAMIENTO SEMI-SUPERVISADO PASO A PASO")
    logger.info(f"⏰ Hora de inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    try:
        # Crear directorio de logs si no existe
        os.makedirs("logs", exist_ok=True)
        
        # Paso 1: Verificar conexiones
        if not await paso_1_verificar_conexiones():
            logger.error("❌ Falló la verificación de conexiones. Abortando...")
            return
        
        # Paso 2: Analizar datos
        df = await paso_2_analizar_datos()
        if df is None:
            logger.error("❌ Falló la extracción de datos. Abortando...")
            return
        
        # Paso 3: Preprocesar datos
        X_labeled, y_labeled, X_unlabeled = await paso_3_preprocesar_datos(df)
        if X_labeled is None:
            logger.error("❌ Falló el preprocesamiento. Abortando...")
            return
        
        # Paso 4: Entrenar modelos
        training_summary = await paso_4_entrenar_modelos(X_labeled, y_labeled, X_unlabeled)
        if training_summary is None:
            logger.error("❌ Falló el entrenamiento. Abortando...")
            return
        
        # Paso 5: Evaluar predicciones
        if not await paso_5_evaluar_predicciones(training_summary):
            logger.error("❌ Falló la evaluación de predicciones")
        
        # Paso 6: Generar predicciones finales
        if not await paso_6_generar_predicciones_finales():
            logger.error("❌ Falló la generación de predicciones finales")
        
        # Resumen final
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*80)
        logger.info("✅ ENTRENAMIENTO SEMI-SUPERVISADO COMPLETADO")
        logger.info(f"⏰ Hora de finalización: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⌛ Duración total: {duration}")
        logger.info("="*80)
        
        # Mostrar archivos generados
        logger.info("📁 Archivos generados en trained_models/semi_supervised/:")
        semi_supervised_dir = "trained_models/semi_supervised"
        if os.path.exists(semi_supervised_dir):
            for file in os.listdir(semi_supervised_dir):
                logger.info(f"  - {file}")
        
    except Exception as e:
        logger.error(f"❌ Error en proceso principal: {str(e)}")
    finally:
        # Cerrar conexiones
        await close_database()
        logger.info("🔌 Conexiones cerradas")

if __name__ == "__main__":
    # Importar numpy aquí para evitar problemas de importación
    import numpy as np
    
    # Ejecutar el proceso principal
    asyncio.run(main())