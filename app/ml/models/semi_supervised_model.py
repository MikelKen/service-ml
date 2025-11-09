#!/usr/bin/env python3
"""
🤖 MODELO DE MACHINE LEARNING SEMI-SUPERVISADO
Implementa algoritmos semi-supervisados para clasificación de postulaciones
Incluye Label Propagation, Self-Training y Co-Training
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import sys
import pickle
import json

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib

from app.ml.preprocessing.semi_supervised_preprocessor import semi_supervised_preprocessor
from app.config.mongodb_connection import get_mongodb_sync, mongodb_connection

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiSupervisedClassifier:
    """Clasificador semi-supervisado para postulaciones"""
    
    def __init__(self, algorithm: str = 'label_propagation'):
        self.algorithm = algorithm
        self.model = None
        self.base_classifier = None
        self.calibrated_model = None
        self.is_trained = False
        
        # Métricas del modelo
        self.metrics = {}
        self.training_history = []
        
        # Configuraciones de algoritmos
        self.algorithm_configs = {
            'label_propagation': {
                'kernel': 'knn',
                'n_neighbors': 7,
                'gamma': 20,
                'max_iter': 1000,
                'tol': 1e-3
            },
            'label_spreading': {
                'kernel': 'knn',
                'n_neighbors': 7,
                'gamma': 20,
                'max_iter': 1000,
                'alpha': 0.2,
                'tol': 1e-3
            },
            'self_training_rf': {
                'base_classifier': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'threshold': 0.75,
                'max_iter': 10,
                'verbose': True
            },
            'self_training_lr': {
                'base_classifier': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'threshold': 0.7,
                'max_iter': 10,
                'verbose': True
            },
            'self_training_gb': {
                'base_classifier': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'threshold': 0.8,
                'max_iter': 10,
                'verbose': True
            }
        }
        
        # Metadatos del modelo
        self.model_info = {
            'algorithm': algorithm,
            'version': '1.0',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'training_config': {},
            'performance_metrics': {},
            'pseudo_label_stats': {}
        }
    
    def _create_model(self) -> Union[LabelPropagation, LabelSpreading, SelfTrainingClassifier]:
        """Crear instancia del modelo según el algoritmo seleccionado"""
        config = self.algorithm_configs.get(self.algorithm, {})
        
        if self.algorithm == 'label_propagation':
            model = LabelPropagation(
                kernel=config['kernel'],
                n_neighbors=config['n_neighbors'],
                gamma=config['gamma'],
                max_iter=config['max_iter'],
                tol=config['tol']
            )
            
        elif self.algorithm == 'label_spreading':
            model = LabelSpreading(
                kernel=config['kernel'],
                n_neighbors=config['n_neighbors'],
                gamma=config['gamma'],
                max_iter=config['max_iter'],
                alpha=config['alpha'],
                tol=config['tol']
            )
            
        elif self.algorithm.startswith('self_training'):
            self.base_classifier = config['base_classifier']
            model = SelfTrainingClassifier(
                base_estimator=self.base_classifier,
                threshold=config['threshold'],
                max_iter=config['max_iter'],
                verbose=config['verbose']
            )
            
        else:
            raise ValueError(f"Algoritmo no soportado: {self.algorithm}")
        
        self.model_info['training_config'] = config
        return model
    
    def prepare_semi_supervised_data(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
                                   X_unlabeled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preparar datos para entrenamiento semi-supervisado"""
        logger.info("🔧 Preparando datos para entrenamiento semi-supervisado...")
        
        # Combinar datos etiquetados y no etiquetados
        X_combined = np.vstack([X_labeled, X_unlabeled])
        
        # Crear array de etiquetas con -1 para datos no etiquetados
        y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])
        
        logger.info(f"📊 Datos combinados: {X_combined.shape}")
        logger.info(f"📊 Etiquetas conocidas: {len(y_labeled)}")
        logger.info(f"📊 Etiquetas desconocidas: {len(X_unlabeled)}")
        
        return X_combined, y_combined
    
    def train(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
             X_unlabeled: np.ndarray, validation_split: float = 0.2) -> Dict[str, Any]:
        """Entrenar el modelo semi-supervisado"""
        logger.info(f"🚀 Iniciando entrenamiento con algoritmo: {self.algorithm}")
        
        # Validar datos mínimos
        if len(X_labeled) < 10:
            raise ValueError("Se requieren al menos 10 muestras etiquetadas para entrenar")
        
        # Crear modelo
        self.model = self._create_model()
        
        # Preparar datos
        X_combined, y_combined = self.prepare_semi_supervised_data(X_labeled, y_labeled, X_unlabeled)
        
        # División para validación
        if validation_split > 0:
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(X_labeled))
            train_idx, val_idx = train_test_split(
                indices, test_size=validation_split, stratify=y_labeled, random_state=42
            )
            
            # Crear conjuntos de entrenamiento y validación
            X_train_labeled = X_labeled[train_idx]
            y_train_labeled = y_labeled[train_idx]
            X_val_labeled = X_labeled[val_idx]
            y_val_labeled = y_labeled[val_idx]
            
            # Recombinar con datos no etiquetados
            X_train_combined = np.vstack([X_train_labeled, X_unlabeled])
            y_train_combined = np.concatenate([y_train_labeled, np.full(len(X_unlabeled), -1)])
        else:
            X_train_combined = X_combined
            y_train_combined = y_combined
            X_val_labeled = None
            y_val_labeled = None
        
        # Entrenar modelo
        training_start = datetime.now()
        logger.info("⏳ Entrenando modelo...")
        
        try:
            self.model.fit(X_train_combined, y_train_combined)
            self.is_trained = True
            
            training_time = (datetime.now() - training_start).total_seconds()
            logger.info(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
            
            # Calcular métricas
            metrics = self._calculate_metrics(X_labeled, y_labeled, X_val_labeled, y_val_labeled)
            
            # Análisis de pseudo-etiquetas
            pseudo_label_stats = self._analyze_pseudo_labels(X_unlabeled)
            
            # Calibrar modelo para obtener probabilidades confiables
            if hasattr(self.model, 'predict_proba') and X_val_labeled is not None:
                self.calibrated_model = CalibratedClassifierCV(self.model, cv=3)
                self.calibrated_model.fit(X_train_combined, y_train_combined)
            
            # Guardar historial de entrenamiento
            training_record = {
                'timestamp': training_start.isoformat(),
                'algorithm': self.algorithm,
                'training_time_seconds': training_time,
                'labeled_samples': len(X_labeled),
                'unlabeled_samples': len(X_unlabeled),
                'metrics': metrics,
                'pseudo_label_stats': pseudo_label_stats
            }
            
            self.training_history.append(training_record)
            self.metrics = metrics
            self.model_info['performance_metrics'] = metrics
            self.model_info['pseudo_label_stats'] = pseudo_label_stats
            
            logger.info("📊 MÉTRICAS DEL MODELO:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")
            
            return training_record
            
        except Exception as e:
            logger.error(f"❌ Error durante el entrenamiento: {e}")
            raise
    
    def _calculate_metrics(self, X_labeled: np.ndarray, y_labeled: np.ndarray,
                          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calcular métricas del modelo"""
        metrics = {}
        
        try:
            # Predicciones en datos de entrenamiento
            y_pred_train = self.model.predict(X_labeled)
            
            # Métricas básicas en entrenamiento
            metrics['train_accuracy'] = accuracy_score(y_labeled, y_pred_train)
            metrics['train_precision'] = precision_score(y_labeled, y_pred_train, average='weighted', zero_division=0)
            metrics['train_recall'] = recall_score(y_labeled, y_pred_train, average='weighted', zero_division=0)
            metrics['train_f1'] = f1_score(y_labeled, y_pred_train, average='weighted', zero_division=0)
            
            # Métricas en validación si están disponibles
            if X_val is not None and y_val is not None:
                y_pred_val = self.model.predict(X_val)
                
                metrics['val_accuracy'] = accuracy_score(y_val, y_pred_val)
                metrics['val_precision'] = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
                metrics['val_recall'] = recall_score(y_val, y_pred_val, average='weighted', zero_division=0)
                metrics['val_f1'] = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
                
                # ROC AUC si hay probabilidades
                if hasattr(self.model, 'predict_proba'):
                    try:
                        y_proba_val = self.model.predict_proba(X_val)
                        if y_proba_val.shape[1] == 2:  # Clasificación binaria
                            metrics['val_roc_auc'] = roc_auc_score(y_val, y_proba_val[:, 1])
                    except:
                        pass
            
            # Cross-validation en datos etiquetados
            if len(X_labeled) > 20:  # Solo si hay suficientes datos
                try:
                    cv_scores = cross_val_score(
                        self.base_classifier if self.base_classifier else self.model,
                        X_labeled, y_labeled, cv=min(5, len(X_labeled)//4), scoring='f1_weighted'
                    )
                    metrics['cv_f1_mean'] = cv_scores.mean()
                    metrics['cv_f1_std'] = cv_scores.std()
                except:
                    pass
            
        except Exception as e:
            logger.error(f"❌ Error calculando métricas: {e}")
        
        return metrics
    
    def _analyze_pseudo_labels(self, X_unlabeled: np.ndarray) -> Dict[str, Any]:
        """Analizar las pseudo-etiquetas generadas"""
        if not self.is_trained:
            return {}
        
        try:
            # Predicciones en datos no etiquetados
            pseudo_labels = self.model.predict(X_unlabeled)
            
            # Probabilidades si están disponibles
            pseudo_probas = None
            if hasattr(self.model, 'predict_proba'):
                pseudo_probas = self.model.predict_proba(X_unlabeled)
            elif hasattr(self.model, 'label_distributions_'):
                # Para Label Propagation/Spreading
                pseudo_probas = self.model.label_distributions_[-len(X_unlabeled):]
            
            stats = {
                'total_unlabeled': len(X_unlabeled),
                'pseudo_label_distribution': {
                    'positive': int(np.sum(pseudo_labels == 1)),
                    'negative': int(np.sum(pseudo_labels == 0))
                }
            }
            
            if pseudo_probas is not None:
                # Confianza de las pseudo-etiquetas
                max_probas = np.max(pseudo_probas, axis=1)
                stats['confidence_stats'] = {
                    'mean_confidence': float(np.mean(max_probas)),
                    'median_confidence': float(np.median(max_probas)),
                    'high_confidence_samples': int(np.sum(max_probas > 0.8)),
                    'low_confidence_samples': int(np.sum(max_probas < 0.6))
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error analizando pseudo-etiquetas: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realizar predicciones"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Obtener probabilidades de predicción"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Usar modelo calibrado si está disponible
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'label_distributions_'):
            # Para Label Propagation/Spreading
            return self.model.label_distributions_[:len(X)]
        else:
            # Fallback: convertir predicciones a probabilidades dummy
            predictions = self.model.predict(X)
            probas = np.zeros((len(predictions), 2))
            probas[predictions == 0, 0] = 0.9
            probas[predictions == 0, 1] = 0.1
            probas[predictions == 1, 0] = 0.1
            probas[predictions == 1, 1] = 0.9
            return probas
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Predicciones con niveles de confianza"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Calcular confianza
        max_probas = np.max(probabilities, axis=1)
        confidence_levels = []
        
        for prob in max_probas:
            if prob >= 0.8:
                confidence_levels.append('high')
            elif prob >= 0.6:
                confidence_levels.append('medium')
            else:
                confidence_levels.append('low')
        
        return predictions, max_probas, confidence_levels
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtener importancia de features"""
        if not self.is_trained:
            return {}
        
        # Solo disponible para algunos algoritmos
        if self.base_classifier and hasattr(self.base_classifier, 'feature_importances_'):
            importance = self.base_classifier.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importance))]
            return dict(zip(feature_names, importance))
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
            feature_names = [f'feature_{i}' for i in range(len(importance))]
            return dict(zip(feature_names, importance))
        else:
            return {}
    
    def save_model(self, filepath: str):
        """Guardar modelo completo"""
        logger.info(f"💾 Guardando modelo en {filepath}")
        
        model_data = {
            'algorithm': self.algorithm,
            'model': self.model,
            'base_classifier': self.base_classifier,
            'calibrated_model': self.calibrated_model,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'training_history': self.training_history,
            'model_info': self.model_info,
            'algorithm_configs': self.algorithm_configs
        }
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar con joblib para modelos de scikit-learn
        joblib.dump(model_data, filepath)
        
        # Guardar metadatos en JSON
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_info, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Modelo guardado exitosamente")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SemiSupervisedClassifier':
        """Cargar modelo desde archivo"""
        logger.info(f"📁 Cargando modelo desde {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Crear instancia
        instance = cls(algorithm=model_data['algorithm'])
        
        # Restaurar atributos
        for key, value in model_data.items():
            setattr(instance, key, value)
        
        logger.info(f"✅ Modelo cargado exitosamente")
        return instance
    
    def update_mongodb_predictions(self, application_ids: List[str], 
                                 predictions: np.ndarray, 
                                 probabilities: np.ndarray,
                                 confidence_levels: List[str]):
        """Actualizar predicciones en MongoDB"""
        logger.info("📝 Actualizando predicciones en MongoDB...")
        
        try:
            mongodb_connection.connect_sync()
            db = get_mongodb_sync()
            collection = db['ml_applications']
            
            for i, app_id in enumerate(application_ids):
                update_data = {
                    'ml_prediction': int(predictions[i]),
                    'ml_probability': float(probabilities[i].max()),
                    'ml_confidence': confidence_levels[i],
                    'updated_at': datetime.now(timezone.utc),
                    'model_algorithm': self.algorithm,
                    'model_version': self.model_info.get('version', '1.0')
                }
                
                collection.update_one(
                    {'application_id': app_id},
                    {'$set': update_data}
                )
            
            logger.info(f"✅ Actualizadas {len(application_ids)} predicciones en MongoDB")
            
        except Exception as e:
            logger.error(f"❌ Error actualizando MongoDB: {e}")
        finally:
            mongodb_connection.disconnect_sync()


# Instancia global del clasificador
semi_supervised_classifier = SemiSupervisedClassifier()


def train_semi_supervised_model(algorithm: str = 'label_propagation', 
                               save_path: str = None) -> Dict[str, Any]:
    """Función conveniente para entrenar modelo semi-supervisado"""
    
    # Crear preprocessor y obtener datos
    X_labeled, y_labeled, X_unlabeled, summary = semi_supervised_preprocessor.fit_transform()
    
    # Crear y entrenar modelo
    classifier = SemiSupervisedClassifier(algorithm=algorithm)
    training_record = classifier.train(X_labeled, y_labeled, X_unlabeled)
    
    # Guardar modelo si se especifica
    if save_path:
        classifier.save_model(save_path)
    
    # Actualizar instancia global
    global semi_supervised_classifier
    semi_supervised_classifier = classifier
    
    return training_record


if __name__ == "__main__":
    # Test del modelo semi-supervisado
    try:
        # Entrenar modelo con Label Propagation
        result = train_semi_supervised_model(
            algorithm='label_propagation',
            save_path='trained_models/semi_supervised/label_propagation_model.pkl'
        )
        
        print("🎉 Modelo semi-supervisado entrenado exitosamente")
        print(f"📊 Precisión: {result['metrics'].get('val_f1', 'N/A')}")
        
    except Exception as e:
        logger.error(f"❌ Error en test del modelo: {e}")