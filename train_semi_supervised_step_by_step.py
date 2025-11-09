#!/usr/bin/env python3
"""
🎯 ENTRENAMIENTO DE MODELO SEMI-SUPERVISADO
Script para entrenar y evaluar modelos semi-supervisados paso a paso
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import json

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.preprocessing.semi_supervised_preprocessor import SemiSupervisedPreprocessor
from app.ml.models.semi_supervised_model import SemiSupervisedClassifier
from app.config.mongodb_connection import get_mongodb_sync, mongodb_connection

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemiSupervisedTrainer:
    """Entrenador de modelos semi-supervisados"""
    
    def __init__(self):
        self.preprocessor = SemiSupervisedPreprocessor()
        self.models = {}
        self.results = {}
    
    async def train_all_algorithms(self):
        """Entrenar todos los algoritmos semi-supervisados disponibles"""
        logger.info("🎯 ENTRENAMIENTO DE MODELOS SEMI-SUPERVISADOS")
        logger.info("=" * 60)
        
        # Algoritmos a entrenar
        algorithms = [
            'label_propagation',
            'label_spreading',
            'self_training_rf',
            'self_training_lr',
            'self_training_gb'
        ]
        
        # 1. Preparar datos
        logger.info("📊 Preparando datos...")
        X_labeled, y_labeled, X_unlabeled, summary = self.preprocessor.fit_transform(
            save_path="trained_models/semi_supervised_preprocessor.pkl"
        )
        
        logger.info(f"✅ Datos preparados:")
        logger.info(f"  🏷️ Etiquetados: {len(X_labeled)}")
        logger.info(f"  ❓ No etiquetados: {len(X_unlabeled)}")
        logger.info(f"  🔧 Features: {X_labeled.shape[1]}")
        
        # 2. Entrenar cada algoritmo
        for algorithm in algorithms:
            logger.info(f"\n🤖 Entrenando: {algorithm}")
            logger.info("-" * 40)
            
            try:
                # Crear modelo
                classifier = SemiSupervisedClassifier(algorithm=algorithm)
                
                # Entrenar
                training_result = classifier.train(
                    X_labeled=X_labeled,
                    y_labeled=y_labeled,
                    X_unlabeled=X_unlabeled,
                    validation_split=0.2
                )
                
                # Guardar modelo
                model_path = f"trained_models/semi_supervised/{algorithm}_model.pkl"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                classifier.save_model(model_path)
                
                # Guardar resultados
                self.models[algorithm] = classifier
                self.results[algorithm] = training_result
                
                # Mostrar métricas
                metrics = training_result['metrics']
                logger.info(f"📊 Métricas de {algorithm}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")
                
                # Guardar registro en MongoDB
                await self._save_training_record(algorithm, training_result, model_path)
                
                logger.info(f"✅ {algorithm} entrenado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {algorithm}: {e}")
                self.results[algorithm] = {'error': str(e)}
        
        # 3. Comparar modelos
        await self._compare_models()
        
        # 4. Generar reporte
        await self._generate_training_report()
    
    async def _save_training_record(self, algorithm: str, training_result: dict, model_path: str):
        """Guardar registro de entrenamiento en MongoDB"""
        try:
            mongodb_connection.connect_sync()
            db = get_mongodb_sync()
            collection = db['ml_model_tracking']
            
            # Desactivar modelos anteriores del mismo algoritmo
            collection.update_many(
                {'algorithm': algorithm, 'is_active': True},
                {'$set': {'is_active': False, 'updated_at': datetime.now()}}
            )
            
            # Crear nuevo registro
            record = {
                'model_id': f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_name': f"SemiSupervised_{algorithm}",
                'model_type': 'semi_supervised',
                'algorithm': algorithm,
                'version': '1.0',
                'is_active': True,
                
                'training_config': {
                    'labeled_samples': training_result.get('labeled_samples', 0),
                    'unlabeled_samples': training_result.get('unlabeled_samples', 0),
                    'features_used': [],
                    'hyperparameters': {}
                },
                
                'metrics': training_result.get('metrics', {}),
                
                'dataset_info': {
                    'total_samples': training_result.get('labeled_samples', 0) + training_result.get('unlabeled_samples', 0),
                    'labeled_ratio': training_result.get('labeled_samples', 0) / max(training_result.get('labeled_samples', 0) + training_result.get('unlabeled_samples', 0), 1),
                    'class_distribution': {},
                    'feature_importance': {}
                },
                
                'trained_at': datetime.fromisoformat(training_result.get('timestamp', datetime.now().isoformat())),
                'created_at': datetime.now(),
                'model_path': model_path,
                'preprocessor_path': "trained_models/semi_supervised_preprocessor.pkl"
            }
            
            collection.insert_one(record)
            logger.info(f"💾 Registro guardado en MongoDB para {algorithm}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando registro de {algorithm}: {e}")
        finally:
            mongodb_connection.disconnect_sync()
    
    async def _compare_models(self):
        """Comparar rendimiento de todos los modelos"""
        logger.info("\n📊 COMPARACIÓN DE MODELOS")
        logger.info("=" * 50)
        
        if not self.results:
            logger.warning("No hay resultados para comparar")
            return
        
        # Extraer métricas para comparación
        comparison_data = []
        
        for algorithm, result in self.results.items():
            if 'error' in result:
                continue
            
            metrics = result.get('metrics', {})
            comparison_data.append({
                'algorithm': algorithm,
                'train_accuracy': metrics.get('train_accuracy', 0),
                'train_f1': metrics.get('train_f1', 0),
                'val_accuracy': metrics.get('val_accuracy', 0),
                'val_f1': metrics.get('val_f1', 0),
                'training_time': result.get('training_time_seconds', 0)
            })
        
        if not comparison_data:
            logger.warning("No hay datos válidos para comparar")
            return
        
        # Ordenar por F1-score de validación (o entrenamiento si no hay validación)
        comparison_data.sort(key=lambda x: x.get('val_f1', x.get('train_f1', 0)), reverse=True)
        
        logger.info("🏆 RANKING DE MODELOS (por F1-Score):")
        for i, data in enumerate(comparison_data, 1):
            f1_score = data.get('val_f1', data.get('train_f1', 0))
            accuracy = data.get('val_accuracy', data.get('train_accuracy', 0))
            time_mins = data['training_time'] / 60
            
            logger.info(f"  {i}. {data['algorithm']}")
            logger.info(f"     F1-Score: {f1_score:.4f}")
            logger.info(f"     Accuracy: {accuracy:.4f}")
            logger.info(f"     Tiempo: {time_mins:.1f} min")
        
        # Mejor modelo
        best_model = comparison_data[0]
        logger.info(f"\n🥇 MEJOR MODELO: {best_model['algorithm']}")
        logger.info(f"   F1-Score: {best_model.get('val_f1', best_model.get('train_f1', 0)):.4f}")
        
        return best_model
    
    async def _generate_training_report(self):
        """Generar reporte completo de entrenamiento"""
        logger.info("\n📋 GENERANDO REPORTE DE ENTRENAMIENTO...")
        
        # Crear estructura del reporte
        report = {
            'timestamp': datetime.now().isoformat(),
            'preprocessing_summary': {
                'total_features': self.preprocessor.feature_names.__len__() if hasattr(self.preprocessor, 'feature_names') else 0,
                'is_fitted': self.preprocessor.is_fitted
            },
            'models_trained': [],
            'comparison': {},
            'recommendations': []
        }
        
        # Añadir resultados de cada modelo
        for algorithm, result in self.results.items():
            if 'error' in result:
                model_report = {
                    'algorithm': algorithm,
                    'success': False,
                    'error': result['error']
                }
            else:
                model_report = {
                    'algorithm': algorithm,
                    'success': True,
                    'metrics': result.get('metrics', {}),
                    'training_time_seconds': result.get('training_time_seconds', 0),
                    'labeled_samples': result.get('labeled_samples', 0),
                    'unlabeled_samples': result.get('unlabeled_samples', 0),
                    'pseudo_label_stats': result.get('pseudo_label_stats', {})
                }
            
            report['models_trained'].append(model_report)
        
        # Añadir comparación
        successful_models = [m for m in report['models_trained'] if m['success']]
        if successful_models:
            best_model = max(successful_models, 
                           key=lambda x: x['metrics'].get('val_f1', x['metrics'].get('train_f1', 0)))
            
            report['comparison'] = {
                'best_model': best_model['algorithm'],
                'best_f1_score': best_model['metrics'].get('val_f1', best_model['metrics'].get('train_f1', 0)),
                'models_count': len(successful_models)
            }
            
            # Recomendaciones
            report['recommendations'] = self._generate_recommendations(successful_models)
        
        # Guardar reporte
        report_path = f"training_reports/semi_supervised_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('training_reports', exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Reporte guardado en: {report_path}")
        
        # Mostrar resumen
        logger.info("\n📊 RESUMEN DEL ENTRENAMIENTO:")
        logger.info(f"  ✅ Modelos exitosos: {len(successful_models)}/{len(self.results)}")
        
        if 'best_model' in report['comparison']:
            logger.info(f"  🏆 Mejor modelo: {report['comparison']['best_model']}")
            logger.info(f"  📈 Mejor F1-Score: {report['comparison']['best_f1_score']:.4f}")
        
        if report['recommendations']:
            logger.info("  💡 Recomendaciones:")
            for rec in report['recommendations']:
                logger.info(f"    - {rec}")
    
    def _generate_recommendations(self, successful_models: list) -> list:
        """Generar recomendaciones basadas en los resultados"""
        recommendations = []
        
        if not successful_models:
            return ["No se pudieron entrenar modelos exitosamente"]
        
        # Analizar métricas
        f1_scores = [m['metrics'].get('val_f1', m['metrics'].get('train_f1', 0)) for m in successful_models]
        avg_f1 = sum(f1_scores) / len(f1_scores)
        
        if avg_f1 < 0.6:
            recommendations.append("F1-Score promedio bajo - considerar más datos etiquetados")
        
        if avg_f1 > 0.8:
            recommendations.append("Excelente rendimiento - modelos listos para producción")
        
        # Analizar datos
        labeled_counts = [m.get('labeled_samples', 0) for m in successful_models]
        avg_labeled = sum(labeled_counts) / len(labeled_counts) if labeled_counts else 0
        
        if avg_labeled < 100:
            recommendations.append("Pocos datos etiquetados - etiquetar más muestras mejorará el rendimiento")
        
        # Analizar tiempos de entrenamiento
        training_times = [m.get('training_time_seconds', 0) for m in successful_models]
        max_time = max(training_times) if training_times else 0
        
        if max_time > 300:  # 5 minutos
            recommendations.append("Algunos modelos tardan mucho - considerar optimización de hiperparámetros")
        
        # Recomendaciones específicas por algoritmo
        algorithms = [m['algorithm'] for m in successful_models]
        
        if 'label_propagation' in algorithms and 'self_training_rf' in algorithms:
            recommendations.append("Considerar ensemble de Label Propagation y Self-Training RF")
        
        return recommendations


async def main():
    """Función principal"""
    trainer = SemiSupervisedTrainer()
    
    try:
        await trainer.train_all_algorithms()
        logger.info("🎉 ENTRENAMIENTO COMPLETADO")
        
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())