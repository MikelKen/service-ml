#!/usr/bin/env python3
"""
🎯 MODELO SEMI-SUPERVISADO PARA PREDICCIÓN DE ESTADO DE POSTULACIONES
Implementa algoritmos semi-supervisados para predecir si una postulación será ACEPTADA o RECHAZADA
"""

import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiSupervisedPredictor:
    """Modelo semi-supervisado para predecir estado de postulaciones"""
    
    def __init__(self, algorithm: str = 'label_propagation', base_classifier: str = 'random_forest'):
        """
        Inicializa el modelo semi-supervisado
        
        Args:
            algorithm: 'label_propagation', 'label_spreading', 'self_training'
            base_classifier: 'random_forest', 'svm', 'logistic_regression'
        """
        self.algorithm = algorithm
        self.base_classifier_type = base_classifier
        self.model = None
        self.base_classifier = None
        self.is_fitted = False
        self.feature_names = []
        self.metrics = {}
        self.label_mapping = {0: 'RECHAZADO', 1: 'ACEPTADO'}
        self.reverse_label_mapping = {'RECHAZADO': 0, 'ACEPTADO': 1}
        
        # Configurar clasificador base
        self._setup_base_classifier()
        
        # Configurar algoritmo semi-supervisado
        self._setup_semi_supervised_algorithm()
    
    def _setup_base_classifier(self):
        """Configura el clasificador base"""
        if self.base_classifier_type == 'random_forest':
            self.base_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif self.base_classifier_type == 'svm':
            self.base_classifier = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        elif self.base_classifier_type == 'logistic_regression':
            self.base_classifier = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Clasificador base no soportado: {self.base_classifier_type}")
    
    def _setup_semi_supervised_algorithm(self):
        """Configura el algoritmo semi-supervisado"""
        if self.algorithm == 'label_propagation':
            self.model = LabelPropagation(
                kernel='knn',
                n_neighbors=7,
                alpha=0.2,
                max_iter=1000,
                tol=1e-3
            )
        elif self.algorithm == 'label_spreading':
            self.model = LabelSpreading(
                kernel='knn',
                n_neighbors=7,
                alpha=0.2,
                max_iter=1000,
                tol=1e-3
            )
        elif self.algorithm == 'self_training':
            self.model = SelfTrainingClassifier(
                base_estimator=self.base_classifier,
                threshold=0.75,
                criterion='threshold',
                k_best=10,
                max_iter=10,
                verbose=True
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {self.algorithm}")
    
    def prepare_semi_supervised_data(self, X: np.ndarray, y: np.ndarray, 
                                   labeled_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento semi-supervisado
        
        Args:
            X: Features
            y: Labels (0 para RECHAZADO, 1 para ACEPTADO)
            labeled_ratio: Proporción de datos etiquetados
        
        Returns:
            X, y_semi donde y_semi contiene -1 para datos no etiquetados
        """
        logger.info(f"🔄 Preparando datos semi-supervisados (ratio etiquetado: {labeled_ratio})")
        
        # Crear copia de las etiquetas
        y_semi = y.copy()
        
        # Determinar cantidad de datos etiquetados
        n_samples = len(y)
        n_labeled = int(n_samples * labeled_ratio)
        
        # Seleccionar índices etiquetados de manera estratificada
        # Asegurar que tengamos ejemplos de ambas clases
        unique_classes = np.unique(y)
        labeled_indices = []
        
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            n_class_labeled = max(1, int(len(class_indices) * labeled_ratio))
            selected_indices = np.random.choice(class_indices, n_class_labeled, replace=False)
            labeled_indices.extend(selected_indices)
        
        # Completar hasta n_labeled si es necesario
        remaining_indices = [i for i in range(n_samples) if i not in labeled_indices]
        if len(labeled_indices) < n_labeled:
            additional_needed = n_labeled - len(labeled_indices)
            if additional_needed <= len(remaining_indices):
                additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
                labeled_indices.extend(additional_indices)
        
        # Marcar datos no etiquetados con -1
        unlabeled_indices = [i for i in range(n_samples) if i not in labeled_indices]
        y_semi[unlabeled_indices] = -1
        
        logger.info(f"📊 Datos etiquetados: {len(labeled_indices)} ({len(labeled_indices)/n_samples*100:.1f}%)")
        logger.info(f"📊 Datos no etiquetados: {len(unlabeled_indices)} ({len(unlabeled_indices)/n_samples*100:.1f}%)")
        
        # Mostrar distribución de clases etiquetadas
        labeled_classes = y[labeled_indices]
        class_distribution = {self.label_mapping[cls]: np.sum(labeled_classes == cls) 
                            for cls in unique_classes}
        logger.info(f"📊 Distribución etiquetada: {class_distribution}")
        
        return X, y_semi
    
    def fit(self, X: np.ndarray, y: np.ndarray, labeled_ratio: float = 0.3, 
            feature_names: Optional[List[str]] = None) -> 'SemiSupervisedPredictor':
        """
        Entrena el modelo semi-supervisado
        
        Args:
            X: Features normalizadas
            y: Labels (0 para RECHAZADO, 1 para ACEPTADO)
            labeled_ratio: Proporción de datos etiquetados
            feature_names: Nombres de las características
        """
        logger.info(f"🚀 ENTRENANDO MODELO SEMI-SUPERVISADO: {self.algorithm}")
        logger.info(f"📊 Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
        logger.info(f"🏷️ Clasificador base: {self.base_classifier_type}")
        
        # Guardar nombres de características
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Preparar datos semi-supervisados
        X_train, y_semi = self.prepare_semi_supervised_data(X, y, labeled_ratio)
        
        # Entrenar modelo
        logger.info("🔄 Entrenando modelo semi-supervisado...")
        
        try:
            self.model.fit(X_train, y_semi)
            self.is_fitted = True
            
            logger.info("✅ Entrenamiento completado")
            
            # Evaluar modelo con datos etiquetados
            self._evaluate_on_labeled_data(X, y, y_semi)
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clases para nuevas muestras"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Usar fit() primero.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice probabilidades para nuevas muestras"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Usar fit() primero.")
        
        # Verificar si el modelo soporta predict_proba
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Para modelos que no tienen predict_proba, usar decision_function si está disponible
            if hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(X)
                # Convertir a probabilidades usando sigmoid
                from scipy.special import expit
                probas = expit(decision_scores)
                # Crear matriz de probabilidades para ambas clases
                return np.column_stack([1 - probas, probas])
            else:
                raise AttributeError("El modelo no soporta predicción de probabilidades")
    
    def predict_single(self, features: Union[np.ndarray, List[float]]) -> Dict[str, Any]:
        """
        Predice el estado para una sola postulación
        
        Args:
            features: Características de la postulación
        
        Returns:
            Diccionario con predicción, probabilidades y confianza
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")
        
        # Convertir a array 2D si es necesario
        if isinstance(features, list):
            features = np.array(features)
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predicción
        prediction = self.model.predict(features)[0]
        predicted_label = self.label_mapping[prediction]
        
        # Probabilidades si están disponibles
        try:
            probabilities = self.predict_proba(features)[0]
            prob_rechazado = float(probabilities[0])
            prob_aceptado = float(probabilities[1])
            confidence = float(max(probabilities))
        except (AttributeError, IndexError):
            prob_rechazado = 1.0 if prediction == 0 else 0.0
            prob_aceptado = 1.0 if prediction == 1 else 0.0
            confidence = 1.0
        
        return {
            'prediction': predicted_label,
            'prediction_numeric': int(prediction),
            'probabilities': {
                'RECHAZADO': prob_rechazado,
                'ACEPTADO': prob_aceptado
            },
            'confidence': confidence,
            'recommendation': self._generate_recommendation(predicted_label, confidence)
        }
    
    def _evaluate_on_labeled_data(self, X: np.ndarray, y_true: np.ndarray, y_semi: np.ndarray):
        """Evalúa el modelo usando solo los datos etiquetados"""
        try:
            # Obtener índices de datos etiquetados
            labeled_mask = y_semi != -1
            X_labeled = X[labeled_mask]
            y_true_labeled = y_true[labeled_mask]
            
            if len(X_labeled) == 0:
                logger.warning("No hay datos etiquetados para evaluación")
                return
            
            # Predicciones en datos etiquetados
            y_pred = self.model.predict(X_labeled)
            
            # Calcular métricas
            accuracy = accuracy_score(y_true_labeled, y_pred)
            
            # AUC solo si hay ambas clases
            if len(np.unique(y_true_labeled)) > 1:
                try:
                    probas = self.predict_proba(X_labeled)
                    auc = roc_auc_score(y_true_labeled, probas[:, 1])
                except:
                    auc = None
            else:
                auc = None
            
            # Guardar métricas
            self.metrics = {
                'accuracy': float(accuracy),
                'auc': float(auc) if auc is not None else None,
                'n_labeled_samples': len(X_labeled),
                'class_distribution': {
                    self.label_mapping[cls]: int(np.sum(y_true_labeled == cls))
                    for cls in np.unique(y_true_labeled)
                }
            }
            
            # Log de resultados
            logger.info("📊 EVALUACIÓN EN DATOS ETIQUETADOS:")
            logger.info(f"  🎯 Accuracy: {accuracy:.3f}")
            if auc:
                logger.info(f"  📈 AUC: {auc:.3f}")
            logger.info(f"  📊 Muestras evaluadas: {len(X_labeled)}")
            
            # Reporte detallado
            report = classification_report(y_true_labeled, y_pred, 
                                         target_names=['RECHAZADO', 'ACEPTADO'],
                                         output_dict=True, zero_division=0)
            
            logger.info("📊 Reporte de clasificación:")
            for class_name in ['RECHAZADO', 'ACEPTADO']:
                if class_name.lower() in report:
                    metrics = report[class_name.lower()]
                    logger.info(f"  {class_name}: Precision={metrics['precision']:.3f}, "
                              f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error en evaluación: {e}")
            self.metrics = {'error': str(e)}
    
    def _generate_recommendation(self, prediction: str, confidence: float) -> str:
        """Genera recomendación basada en predicción y confianza"""
        if confidence >= 0.8:
            if prediction == 'ACEPTADO':
                return "Candidato altamente recomendado para continuar en el proceso"
            else:
                return "Candidato con baja probabilidad de éxito en el proceso"
        elif confidence >= 0.6:
            if prediction == 'ACEPTADO':
                return "Candidato con potencial, considerar entrevista adicional"
            else:
                return "Candidato requiere evaluación más detallada"
        else:
            return "Predicción incierta, se recomienda evaluación manual"
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del modelo"""
        if not self.is_fitted:
            return {"error": "Modelo no entrenado"}
        
        summary = {
            'algorithm': self.algorithm,
            'base_classifier': self.base_classifier_type,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names),
            'metrics': self.metrics,
            'label_mapping': self.label_mapping
        }
        
        # Información específica del algoritmo
        if hasattr(self.model, 'n_iter_'):
            summary['training_iterations'] = int(self.model.n_iter_)
        
        if hasattr(self.model, 'label_distributions_'):
            summary['label_distributions'] = self.model.label_distributions_.tolist()
        
        return summary
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Obtiene importancia de características si está disponible"""
        if not self.is_fitted:
            return None
        
        # Para self-training con Random Forest
        if (self.algorithm == 'self_training' and 
            hasattr(self.model.base_estimator_, 'feature_importances_')):
            
            importances = self.model.base_estimator_.feature_importances_
            return {
                name: float(importance) 
                for name, importance in zip(self.feature_names, importances)
            }
        
        return None
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        model_data = {
            'algorithm': self.algorithm,
            'base_classifier_type': self.base_classifier_type,
            'model': self.model,
            'base_classifier': self.base_classifier,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'label_mapping': self.label_mapping,
            'reverse_label_mapping': self.reverse_label_mapping,
            'training_date': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Modelo guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Carga un modelo entrenado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            algorithm=model_data['algorithm'],
            base_classifier=model_data['base_classifier_type']
        )
        
        model.model = model_data['model']
        model.base_classifier = model_data['base_classifier']
        model.is_fitted = model_data['is_fitted']
        model.feature_names = model_data['feature_names']
        model.metrics = model_data['metrics']
        model.label_mapping = model_data['label_mapping']
        model.reverse_label_mapping = model_data['reverse_label_mapping']
        
        logger.info(f"📂 Modelo cargado desde: {filepath}")
        logger.info(f"🗓️ Entrenado el: {model_data.get('training_date', 'fecha desconocida')}")
        
        return model

if __name__ == "__main__":
    # Ejemplo de uso
    print("🎯 SemiSupervisedPredictor creado")
    print("✅ Algoritmos disponibles: label_propagation, label_spreading, self_training")
    print("🚀 Listo para entrenar modelo semi-supervisado de postulaciones")