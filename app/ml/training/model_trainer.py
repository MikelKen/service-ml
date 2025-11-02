"""
Módulo de entrenamiento de modelos para predicción de contratación
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
import joblib
import logging
from typing import Dict, Tuple, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Clase para entrenar y evaluar modelos de ML"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        self.feature_engineer = None
        self.evaluation_results = {}
    
    def initialize_models(self) -> Dict[str, Any]:
        """Inicializa modelos a entrenar"""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                max_depth=10
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight='balanced',
                verbosity=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        }
        
        return models
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, 
                          test_size: float = 0.2) -> Dict[str, Any]:
        """Entrena y evalúa múltiples modelos"""
        logger.info("Iniciando entrenamiento y evaluación de modelos...")
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
        logger.info(f"Datos de prueba: {X_test.shape[0]} muestras")
        
        # Inicializar modelos
        models = self.initialize_models()
        results = {}
        
        # Entrenar cada modelo
        for name, model in models.items():
            logger.info(f"Entrenando {name}...")
            
            try:
                # Entrenar modelo
                model.fit(X_train, y_train)
                
                # Evaluar modelo
                eval_results = self.evaluate_model(model, X_test, y_test, name)
                results[name] = {
                    'model': model,
                    'evaluation': eval_results
                }
                
                # Actualizar mejor modelo
                current_score = eval_results['roc_auc']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_model = model
                    self.best_model_name = name
                
                logger.info(f"{name} - ROC AUC: {current_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.models = results
        self.evaluation_results = results
        
        logger.info(f"Mejor modelo: {self.best_model_name} con ROC AUC: {self.best_score:.4f}")
        
        return results
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, Any]:
        """Evalúa un modelo individual"""
        try:
            # Predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métricas principales
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            return {
                'roc_auc': roc_auc,
                'average_precision': avg_precision,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'precision': class_report['1']['precision'],
                'recall': class_report['1']['recall'],
                'f1_score': class_report['1']['f1-score']
            }
            
        except Exception as e:
            logger.error(f"Error evaluando modelo {model_name}: {str(e)}")
            return {'error': str(e)}
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv: int = 5) -> Dict[str, Any]:
        """Realiza validación cruzada detallada"""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scores = {
            'roc_auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Entrenar en fold
            model.fit(X_train_cv, y_train_cv)
            
            # Predecir en validación
            y_pred = model.predict(X_val_cv)
            y_pred_proba = model.predict_proba(X_val_cv)[:, 1]
            
            # Calcular métricas
            scores['roc_auc'].append(roc_auc_score(y_val_cv, y_pred_proba))
            
            # Para precision, recall, f1 usar classification_report
            report = classification_report(y_val_cv, y_pred, output_dict=True)
            scores['precision'].append(report['1']['precision'])
            scores['recall'].append(report['1']['recall'])
            scores['f1'].append(report['1']['f1-score'])
        
        # Calcular estadísticas
        cv_results = {}
        for metric, values in scores.items():
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
            cv_results[f'{metric}_scores'] = values
        
        return cv_results
    
    def save_best_model(self, save_path: str, feature_engineer=None, 
                       include_evaluation: bool = True):
        """Guarda el mejor modelo y artefactos"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Preparar artefactos para guardar
        artifacts = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        # Incluir feature engineer si se proporciona
        if feature_engineer:
            artifacts['feature_engineer'] = feature_engineer
            self.feature_engineer = feature_engineer
        
        # Incluir resultados de evaluación
        if include_evaluation and self.evaluation_results:
            artifacts['evaluation_results'] = self.evaluation_results
        
        # Guardar
        joblib.dump(artifacts, save_path)
        logger.info(f"Modelo guardado en: {save_path}")
        
        return save_path
    
    def get_feature_importance(self, model=None) -> Dict[str, float]:
        """Obtiene importancia de features"""
        if model is None:
            model = self.best_model
        
        if model is None:
            return {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Modelos con feature_importances_ (RF, XGBoost, LightGBM)
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Modelos lineales
                importances = abs(model.coef_[0])
            else:
                return {}
            
            # Crear nombres de features si no están disponibles
            if self.feature_engineer and hasattr(self.feature_engineer, 'feature_names_'):
                feature_names = self.feature_engineer.feature_names_
            else:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Combinar nombres y importancias
            feature_importance = dict(zip(feature_names, importances))
            
            # Ordenar por importancia
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error obteniendo feature importance: {str(e)}")
            return {}
    
    def plot_evaluation_results(self, save_path: str = None):
        """Genera gráficos de evaluación"""
        if not self.evaluation_results:
            logger.warning("No hay resultados de evaluación para graficar")
            return
        
        # Configurar subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Evaluación de Modelos', fontsize=16)
        
        # Preparar datos para gráficos
        model_names = []
        roc_aucs = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for name, results in self.evaluation_results.items():
            if 'error' not in results:
                model_names.append(name)
                roc_aucs.append(results['evaluation']['roc_auc'])
                precisions.append(results['evaluation']['precision'])
                recalls.append(results['evaluation']['recall'])
                f1_scores.append(results['evaluation']['f1_score'])
        
        # ROC AUC comparison
        axes[0, 0].bar(model_names, roc_aucs)
        axes[0, 0].set_title('ROC AUC Comparison')
        axes[0, 0].set_ylabel('ROC AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        axes[0, 1].bar(model_names, precisions)
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        axes[1, 0].bar(model_names, recalls)
        axes[1, 0].set_title('Recall Comparison')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        axes[1, 1].bar(model_names, f1_scores)
        axes[1, 1].set_title('F1 Score Comparison')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del entrenamiento"""
        if not self.evaluation_results:
            return {'status': 'No training completed'}
        
        summary = {
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'models_trained': len(self.evaluation_results),
            'models_details': {}
        }
        
        for name, results in self.evaluation_results.items():
            if 'error' not in results:
                eval_data = results['evaluation']
                summary['models_details'][name] = {
                    'roc_auc': eval_data['roc_auc'],
                    'precision': eval_data['precision'],
                    'recall': eval_data['recall'],
                    'f1_score': eval_data['f1_score']
                }
            else:
                summary['models_details'][name] = {'error': results['error']}
        
        return summary