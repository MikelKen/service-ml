import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import logging
import pickle
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class SemiSupervisedPostulacionModel:
    """Modelo semi-supervisado para predicción de estados de postulaciones"""
    
    def __init__(self, model_type: str = 'label_propagation'):
        """
        Inicializa el modelo semi-supervisado
        
        Args:
            model_type: Tipo de modelo ('label_propagation', 'label_spreading', 'self_training')
        """
        self.model_type = model_type
        self.model = None
        self.supervisor_model = None  # Modelo supervisor para auto-entrenamiento
        self.is_trained = False
        self.classes_ = None
        self.training_metrics = {}
        self.feature_importance = {}
        
        # Configurar modelo base
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo según el tipo especificado"""
        if self.model_type == 'label_propagation':
            self.model = LabelPropagation(
                kernel='knn',
                n_neighbors=7,
                gamma=20,
                max_iter=1000,
                tol=1e-3
            )
        elif self.model_type == 'label_spreading':
            self.model = LabelSpreading(
                kernel='knn',
                n_neighbors=7,
                gamma=20,
                max_iter=1000,
                tol=1e-3,
                alpha=0.2
            )
        elif self.model_type == 'self_training':
            # Modelo base para auto-entrenamiento
            self.supervisor_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
    
    def train(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
              X_unlabeled: np.ndarray, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el modelo semi-supervisado
        
        Args:
            X_labeled: Características de datos etiquetados
            y_labeled: Etiquetas de datos etiquetados
            X_unlabeled: Características de datos no etiquetados
            validation_split: Proporción de datos para validación
            
        Returns:
            Métricas de entrenamiento
        """
        try:
            logger.info(f"Iniciando entrenamiento del modelo {self.model_type}")
            logger.info(f"Datos etiquetados: {X_labeled.shape}, No etiquetados: {X_unlabeled.shape}")
            
            if len(X_labeled) == 0:
                raise ValueError("No hay datos etiquetados para entrenar")
            
            # Si no hay datos no etiquetados, usar modelo supervisado
            if len(X_unlabeled) == 0:
                logger.warning("No hay datos no etiquetados. Usando entrenamiento supervisado.")
                return self._train_supervised(X_labeled, y_labeled, validation_split)
            
            # Dividir datos etiquetados para validación
            if validation_split > 0 and len(X_labeled) > 5:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_labeled, y_labeled, test_size=validation_split, 
                    random_state=42, stratify=y_labeled if len(np.unique(y_labeled)) > 1 else None
                )
            else:
                X_train, y_train = X_labeled, y_labeled
                X_val, y_val = None, None
            
            # Entrenar según el tipo de modelo
            if self.model_type in ['label_propagation', 'label_spreading']:
                metrics = self._train_graph_based(X_train, y_train, X_unlabeled, X_val, y_val)
            elif self.model_type == 'self_training':
                metrics = self._train_self_training(X_train, y_train, X_unlabeled, X_val, y_val)
            
            self.is_trained = True
            self.training_metrics = metrics
            
            logger.info("Entrenamiento completado exitosamente")
            return metrics
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise e
    
    def _train_supervised(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
                         validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena un modelo supervisado cuando no hay datos no etiquetados
        """
        logger.info("Entrenando modelo supervisado (fallback)")
        
        # Usar RandomForest como modelo supervisado
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Dividir datos para validación
        if validation_split > 0 and len(X_labeled) > 5:
            X_train, X_val, y_train, y_val = train_test_split(
                X_labeled, y_labeled, test_size=validation_split, 
                random_state=42, stratify=y_labeled if len(np.unique(y_labeled)) > 1 else None
            )
        else:
            X_train, y_train = X_labeled, y_labeled
            X_val, y_val = None, None
        
        # Entrenar modelo
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        # Métricas de entrenamiento
        metrics = {
            'model_type': f'{self.model_type}_supervised_fallback',
            'training_samples': len(X_train),
            'unlabeled_samples': 0,
            'total_samples': len(X_train),
            'classes': self.classes_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Precisión en entrenamiento
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        metrics['train_accuracy'] = train_accuracy
        
        # Validación si hay datos disponibles
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            metrics['val_accuracy'] = val_accuracy
            metrics['val_classification_report'] = classification_report(y_val, y_val_pred, output_dict=True)
        
        # Cross-validation
        if len(X_labeled) > 10:
            cv_scores = cross_val_score(self.model, X_labeled, y_labeled, cv=5)
            metrics['cross_val_accuracy'] = {
                'mean': float(np.mean(cv_scores)),
                'std': float(np.std(cv_scores)),
                'scores': cv_scores.tolist()
            }
        
        logger.info(f"Modelo supervisado entrenado. Precisión: {train_accuracy:.4f}")
        
        return metrics
    
    def _train_graph_based(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_unlabeled: np.ndarray, X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Entrena modelos basados en grafos (Label Propagation/Spreading)"""
        
        # Combinar datos etiquetados y no etiquetados
        X_combined = np.vstack([X_train, X_unlabeled])
        y_combined = np.hstack([y_train, [-1] * len(X_unlabeled)])  # -1 para no etiquetados
        
        logger.info(f"Datos combinados para entrenamiento: {X_combined.shape}")
        
        # Entrenar modelo
        self.model.fit(X_combined, y_combined)
        self.classes_ = self.model.classes_
        
        # Métricas de entrenamiento
        metrics = {
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'unlabeled_samples': len(X_unlabeled),
            'total_samples': len(X_combined),
            'classes': self.classes_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Predicciones en datos de entrenamiento
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        metrics['train_accuracy'] = train_accuracy
        
        # Validación si hay datos disponibles
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            metrics['val_accuracy'] = val_accuracy
            metrics['val_classification_report'] = classification_report(y_val, y_val_pred, output_dict=True)
        
        # Estadísticas de propagación de etiquetas
        unlabeled_predictions = self.model.predict(X_unlabeled)
        unique_predicted, counts = np.unique(unlabeled_predictions, return_counts=True)
        
        metrics['unlabeled_predictions_distribution'] = dict(zip(unique_predicted.tolist(), counts.tolist()))
        
        # Confianza de las predicciones
        if hasattr(self.model, 'label_distributions_'):
            label_distributions = self.model.label_distributions_[-len(X_unlabeled):]
            max_probs = np.max(label_distributions, axis=1)
            metrics['prediction_confidence'] = {
                'mean': float(np.mean(max_probs)),
                'std': float(np.std(max_probs)),
                'min': float(np.min(max_probs)),
                'max': float(np.max(max_probs))
            }
        
        logger.info(f"Métricas de entrenamiento: Precisión de entrenamiento: {train_accuracy:.4f}")
        
        return metrics
    
    def _train_self_training(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_unlabeled: np.ndarray, X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Entrena usando auto-entrenamiento iterativo"""
        
        from sklearn.semi_supervised import SelfTrainingClassifier
        
        # Crear modelo de auto-entrenamiento
        self.model = SelfTrainingClassifier(
            base_estimator=self.supervisor_model,
            threshold=0.75,  # Umbral de confianza
            criterion='threshold',
            max_iter=10,
            verbose=True
        )
        
        # Combinar datos para auto-entrenamiento
        X_combined = np.vstack([X_train, X_unlabeled])
        y_combined = np.hstack([y_train, [-1] * len(X_unlabeled)])
        
        logger.info(f"Iniciando auto-entrenamiento con {len(X_train)} muestras etiquetadas")
        
        # Entrenar
        self.model.fit(X_combined, y_combined)
        self.classes_ = self.model.classes_
        
        # Métricas
        metrics = {
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'unlabeled_samples': len(X_unlabeled),
            'total_samples': len(X_combined),
            'classes': self.classes_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Predicciones y métricas
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        metrics['train_accuracy'] = train_accuracy
        
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            metrics['val_accuracy'] = val_accuracy
            metrics['val_classification_report'] = classification_report(y_val, y_val_pred, output_dict=True)
        
        # Estadísticas de auto-entrenamiento
        if hasattr(self.model, 'labeled_iter_'):
            metrics['labeled_iterations'] = len(self.model.labeled_iter_)
            
        # Predicciones en datos no etiquetados
        unlabeled_predictions = self.model.predict(X_unlabeled)
        unique_predicted, counts = np.unique(unlabeled_predictions, return_counts=True)
        metrics['unlabeled_predictions_distribution'] = dict(zip(unique_predicted.tolist(), counts.tolist()))
        
        # Importancia de características (si está disponible)
        if hasattr(self.model.base_estimator_, 'feature_importances_'):
            self.feature_importance = self.model.base_estimator_.feature_importances_
            metrics['has_feature_importance'] = True
        
        logger.info(f"Auto-entrenamiento completado. Precisión: {train_accuracy:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones sobre nuevos datos"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones probabilísticas"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Para modelos que no tienen predict_proba, usar distribuciones de etiquetas
            if hasattr(self.model, 'label_distributions_'):
                # Necesitamos reentrenar con los nuevos datos para obtener distribuciones
                logger.warning("predict_proba no disponible directamente. Usando predicciones determinísticas.")
                predictions = self.model.predict(X)
                # Crear probabilidades "dummy" basadas en predicciones
                n_classes = len(self.classes_)
                proba = np.zeros((len(X), n_classes))
                for i, pred in enumerate(predictions):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    proba[i, class_idx] = 1.0
                return proba
            else:
                raise NotImplementedError("predict_proba no disponible para este modelo")
    
    def get_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Obtiene la confianza de las predicciones"""
        try:
            proba = self.predict_proba(X)
            return np.max(proba, axis=1)
        except:
            # Fallback: confianza uniforme
            return np.ones(len(X)) * 0.5
    
    def evaluate_unlabeled_predictions(self, X_unlabeled: np.ndarray) -> Dict[str, Any]:
        """Evalúa las predicciones en datos no etiquetados"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        predictions = self.predict(X_unlabeled)
        confidence_scores = self.get_prediction_confidence(X_unlabeled)
        
        # Estadísticas
        unique_preds, counts = np.unique(predictions, return_counts=True)
        
        evaluation = {
            'total_predictions': len(predictions),
            'predictions_distribution': dict(zip(unique_preds.tolist(), counts.tolist())),
            'confidence_stats': {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores)),
                'high_confidence_count': int(np.sum(confidence_scores > 0.8)),
                'low_confidence_count': int(np.sum(confidence_scores < 0.6))
            }
        }
        
        return evaluation
    
    def save(self, filepath: str, save_metrics: bool = True):
        """Guarda el modelo entrenado"""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'classes_': self.classes_,
                'feature_importance': self.feature_importance
            }
            
            # Guardar modelo
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Guardar métricas por separado
            if save_metrics and self.training_metrics:
                metrics_path = filepath.replace('.pkl', '_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(self.training_metrics, f, indent=2)
            
            logger.info(f"Modelo guardado en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            raise e
    
    def load(self, filepath: str):
        """Carga un modelo guardado"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            self.classes_ = model_data['classes_']
            self.feature_importance = model_data.get('feature_importance', {})
            
            # Cargar métricas si existen
            metrics_path = filepath.replace('.pkl', '_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.training_metrics = json.load(f)
            
            logger.info(f"Modelo cargado desde: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise e
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del entrenamiento"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'training_metrics': self.training_metrics,
            'has_feature_importance': len(self.feature_importance) > 0
        }