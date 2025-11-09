import asyncio
import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from app.ml.data.postgres_extractor import postgres_extractor
from app.ml.preprocessing.semi_supervised_preprocessor import SemiSupervisedPreprocessor
from app.ml.models.semi_supervised_model import SemiSupervisedPostulacionModel
from app.config.connection import mongodb, init_database

logger = logging.getLogger(__name__)

class SemiSupervisedTrainer:
    """Entrenador para modelo semi-supervisado de postulaciones"""
    
    def __init__(self, models_path: str = "trained_models"):
        self.models_path = models_path
        self.preprocessor = SemiSupervisedPreprocessor()
        self.models = {}
        self.training_history = []
        
        # Crear directorio de modelos si no existe
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(f"{self.models_path}/semi_supervised", exist_ok=True)
    
    async def train_all_models(self, save_to_mongo: bool = True) -> Dict[str, Any]:
        """
        Entrena todos los tipos de modelos semi-supervisados
        
        Args:
            save_to_mongo: Si guardar los resultados en MongoDB
            
        Returns:
            Resumen del entrenamiento de todos los modelos
        """
        try:
            logger.info("Iniciando entrenamiento de modelos semi-supervisados")
            
            # Inicializar conexiones a base de datos
            await init_database()
            
            # Extraer datos
            logger.info("Extrayendo datos de PostgreSQL...")
            df = await postgres_extractor.extract_postulaciones_with_features()
            
            if df.empty:
                raise ValueError("No se encontraron datos para entrenar")
            
            # Verificar distribución de datos
            await self._analyze_data_distribution(df)
            
            # Preprocesar datos
            logger.info("Preprocesando datos...")
            X_labeled, y_labeled, X_unlabeled = self.preprocessor.fit_transform(df, 'estado')
            
            # Guardar preprocesador
            preprocessor_path = f"{self.models_path}/semi_supervised/preprocessor.pkl"
            self.preprocessor.save(preprocessor_path)
            
            # Tipos de modelos a entrenar
            model_types = ['label_propagation', 'label_spreading', 'self_training']
            training_results = {}
            
            for model_type in model_types:
                logger.info(f"\nEntrenando modelo: {model_type}")
                
                try:
                    # Crear y entrenar modelo
                    model = SemiSupervisedPostulacionModel(model_type=model_type)
                    metrics = model.train(X_labeled, y_labeled, X_unlabeled, validation_split=0.2)
                    
                    # Evaluar predicciones en datos no etiquetados
                    unlabeled_eval = model.evaluate_unlabeled_predictions(X_unlabeled)
                    metrics.update(unlabeled_eval)
                    
                    # Guardar modelo
                    model_path = f"{self.models_path}/semi_supervised/{model_type}_model.pkl"
                    model.save(model_path)
                    
                    self.models[model_type] = model
                    training_results[model_type] = metrics
                    
                    logger.info(f"Modelo {model_type} entrenado exitosamente")
                    
                except Exception as e:
                    logger.error(f"Error entrenando modelo {model_type}: {str(e)}")
                    training_results[model_type] = {'error': str(e)}
            
            # Comparar modelos
            best_model = self._compare_models(training_results)
            
            # Crear resumen completo
            training_summary = {
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'total_samples': len(df),
                    'labeled_samples': len(X_labeled),
                    'unlabeled_samples': len(X_unlabeled),
                    'features_count': X_labeled.shape[1] if len(X_labeled) > 0 else 0,
                    'classes': self.preprocessor.estado_mapping
                },
                'models_trained': list(training_results.keys()),
                'training_results': training_results,
                'best_model': best_model,
                'files_generated': {
                    'preprocessor': preprocessor_path,
                    'models': {model_type: f"{self.models_path}/semi_supervised/{model_type}_model.pkl" 
                              for model_type in model_types if model_type in self.models}
                }
            }
            
            # Guardar resumen
            summary_path = f"{self.models_path}/semi_supervised/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            # Guardar en MongoDB si se especifica
            if save_to_mongo:
                await self._save_results_to_mongo(training_summary, df)
            
            # Generar predicciones para datos no etiquetados
            if best_model['model_type'] in self.models:
                await self._predict_unlabeled_data(df, best_model['model_type'])
            
            self.training_history.append(training_summary)
            
            logger.info("Entrenamiento de modelos semi-supervisados completado")
            return training_summary
            
        except Exception as e:
            logger.error(f"Error en entrenamiento de modelos: {str(e)}")
            raise e
    
    async def _analyze_data_distribution(self, df: pd.DataFrame):
        """Analiza la distribución de datos"""
        try:
            # Estadísticas básicas
            total_samples = len(df)
            labeled_samples = df['estado'].notna().sum()
            unlabeled_samples = total_samples - labeled_samples
            
            logger.info(f"Análisis de datos:")
            logger.info(f"  Total de muestras: {total_samples}")
            logger.info(f"  Muestras etiquetadas: {labeled_samples}")
            logger.info(f"  Muestras no etiquetadas: {unlabeled_samples}")
            logger.info(f"  Porcentaje etiquetado: {(labeled_samples/total_samples)*100:.2f}%")
            
            # Distribución de estados
            if labeled_samples > 0:
                estado_dist = df['estado'].value_counts()
                logger.info(f"  Distribución de estados: {estado_dist.to_dict()}")
            
            # Verificar si hay suficientes datos para semi-supervisado
            if labeled_samples < 5:
                logger.warning("⚠️  Muy pocas muestras etiquetadas para entrenamiento semi-supervisado")
            
            if unlabeled_samples == 0:
                logger.warning("⚠️  No hay muestras no etiquetadas para el aprendizaje semi-supervisado")
            
        except Exception as e:
            logger.error(f"Error analizando distribución de datos: {str(e)}")
    
    def _compare_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compara los resultados de entrenamiento de diferentes modelos"""
        try:
            best_model = {'model_type': None, 'score': -1, 'metric': 'val_accuracy'}
            
            for model_type, results in training_results.items():
                if 'error' in results:
                    continue
                
                # Priorizar precisión de validación, luego de entrenamiento
                score = results.get('val_accuracy', results.get('train_accuracy', 0))
                
                if score > best_model['score']:
                    best_model.update({
                        'model_type': model_type,
                        'score': score,
                        'results': results
                    })
            
            if best_model['model_type']:
                logger.info(f"Mejor modelo: {best_model['model_type']} con {best_model['metric']}: {best_model['score']:.4f}")
            else:
                logger.warning("No se pudo determinar el mejor modelo")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error comparando modelos: {str(e)}")
            return {'model_type': None, 'score': -1}
    
    async def _save_results_to_mongo(self, training_summary: Dict[str, Any], original_df: pd.DataFrame):
        """Guarda los resultados del entrenamiento en MongoDB"""
        try:
            logger.info("Guardando resultados en MongoDB...")
            
            db = mongodb.get_database()
            
            # Colección para resúmenes de entrenamiento
            training_collection = db['ml_training_summaries']
            await training_collection.insert_one(training_summary)
            
            # Colección para datos procesados
            processed_data_collection = db['processed_postulaciones_data']
            
            # Guardar datos etiquetados y no etiquetados por separado
            labeled_data = original_df[original_df['estado'].notna()].to_dict('records')
            unlabeled_data = original_df[original_df['estado'].isna()].to_dict('records')
            
            if labeled_data:
                await processed_data_collection.insert_many([
                    {'type': 'labeled', 'data': item, 'timestamp': datetime.now().isoformat()}
                    for item in labeled_data
                ])
            
            if unlabeled_data:
                await processed_data_collection.insert_many([
                    {'type': 'unlabeled', 'data': item, 'timestamp': datetime.now().isoformat()}
                    for item in unlabeled_data
                ])
            
            # Colección para métricas de modelos
            metrics_collection = db['ml_model_metrics']
            for model_type, results in training_summary['training_results'].items():
                if 'error' not in results:
                    await metrics_collection.insert_one({
                        'model_type': model_type,
                        'training_timestamp': training_summary['timestamp'],
                        'metrics': results,
                        'is_semi_supervised': True
                    })
            
            logger.info("Resultados guardados en MongoDB exitosamente")
            
        except Exception as e:
            logger.error(f"Error guardando en MongoDB: {str(e)}")
    
    async def _predict_unlabeled_data(self, df: pd.DataFrame, best_model_type: str):
        """Genera predicciones para datos no etiquetados usando el mejor modelo"""
        try:
            logger.info(f"Generando predicciones con modelo {best_model_type}")
            
            # Obtener datos no etiquetados
            unlabeled_mask = df['estado'].isna()
            unlabeled_df = df[unlabeled_mask].copy()
            
            if len(unlabeled_df) == 0:
                logger.info("No hay datos no etiquetados para predecir")
                return
            
            # Obtener modelo entrenado
            model = self.models[best_model_type]
            
            # Procesar datos
            X_unlabeled = self.preprocessor.transform(unlabeled_df)
            
            # Realizar predicciones
            predictions = model.predict(X_unlabeled)
            confidence_scores = model.get_prediction_confidence(X_unlabeled)
            
            # Convertir predicciones numéricas a etiquetas de texto
            label_encoder = self.preprocessor.label_encoder
            predicted_labels = label_encoder.inverse_transform(predictions)
            
            # Crear DataFrame con predicciones
            predictions_df = unlabeled_df.copy()
            predictions_df['predicted_estado'] = predicted_labels
            predictions_df['prediction_confidence'] = confidence_scores
            predictions_df['prediction_timestamp'] = datetime.now().isoformat()
            predictions_df['model_used'] = best_model_type
            
            # Guardar predicciones
            predictions_path = f"{self.models_path}/semi_supervised/unlabeled_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions_df.to_csv(predictions_path, index=False)
            
            # Guardar en MongoDB
            db = mongodb.get_database()
            predictions_collection = db['ml_predictions']
            
            predictions_data = predictions_df.to_dict('records')
            await predictions_collection.insert_many(predictions_data)
            
            # Estadísticas de predicciones
            pred_stats = {
                'total_predictions': len(predicted_labels),
                'unique_predictions': len(np.unique(predicted_labels)),
                'prediction_distribution': dict(zip(*np.unique(predicted_labels, return_counts=True))),
                'high_confidence_predictions': int(np.sum(confidence_scores > 0.8)),
                'low_confidence_predictions': int(np.sum(confidence_scores < 0.6)),
                'mean_confidence': float(np.mean(confidence_scores))
            }
            
            logger.info(f"Predicciones generadas: {pred_stats}")
            logger.info(f"Archivo guardado en: {predictions_path}")
            
        except Exception as e:
            logger.error(f"Error generando predicciones: {str(e)}")
    
    async def retrain_with_new_data(self, model_type: str = None) -> Dict[str, Any]:
        """Re-entrena un modelo específico con nuevos datos"""
        try:
            if model_type is None:
                # Usar el mejor modelo del último entrenamiento
                if self.training_history:
                    model_type = self.training_history[-1]['best_model']['model_type']
                else:
                    model_type = 'label_propagation'  # Default
            
            logger.info(f"Re-entrenando modelo {model_type}")
            
            # Extraer datos actualizados
            df = await postgres_extractor.extract_postulaciones_with_features()
            
            # Preprocesar
            X_labeled, y_labeled, X_unlabeled = self.preprocessor.fit_transform(df, 'estado')
            
            # Re-entrenar modelo
            model = SemiSupervisedPostulacionModel(model_type=model_type)
            metrics = model.train(X_labeled, y_labeled, X_unlabeled)
            
            # Guardar modelo actualizado
            model_path = f"{self.models_path}/semi_supervised/{model_type}_model_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model.save(model_path)
            
            self.models[model_type] = model
            
            logger.info(f"Re-entrenamiento completado para {model_type}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error en re-entrenamiento: {str(e)}")
            raise e
    
    def load_trained_model(self, model_type: str, model_path: str = None) -> SemiSupervisedPostulacionModel:
        """Carga un modelo entrenado"""
        try:
            if model_path is None:
                model_path = f"{self.models_path}/semi_supervised/{model_type}_model.pkl"
            
            model = SemiSupervisedPostulacionModel(model_type=model_type)
            model.load(model_path)
            
            self.models[model_type] = model
            logger.info(f"Modelo {model_type} cargado desde {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise e
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Obtiene el historial de entrenamientos"""
        return self.training_history

# Instancia global del entrenador
semi_supervised_trainer = SemiSupervisedTrainer()