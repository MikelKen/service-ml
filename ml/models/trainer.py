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
                if eval_results['roc_auc'] > self.best_score:
                    self.best_score = eval_results['roc_auc']
                    self.best_model = model
                    self.best_model_name = name
                
                logger.info(f"{name} - ROC AUC: {eval_results['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {str(e)}")
                continue
        
        self.models = results
        self.evaluation_results = results
        
        logger.info(f"Mejor modelo: {self.best_model_name} (ROC AUC: {self.best_score:.4f})")
        
        return results
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """Evalúa un modelo individual"""
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'accuracy': class_report['accuracy']
        }
        
        return results
    
    def cross_validate_best_model(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Realiza validación cruzada del mejor modelo"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado. Ejecute train_and_evaluate primero.")
        
        logger.info(f"Validación cruzada de {self.best_model_name}...")
        
        # Configurar validación cruzada estratificada
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Scores de validación cruzada
        cv_scores = cross_val_score(
            self.best_model, X, y, cv=cv_strategy, scoring='roc_auc'
        )
        
        cv_results = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'model_name': self.best_model_name
        }
        
        logger.info(f"CV ROC AUC: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']*2:.4f})")
        
        return cv_results
    
    def calibrate_best_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Calibra el mejor modelo para obtener mejores probabilidades"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado. Ejecute train_and_evaluate primero.")
        
        logger.info(f"Calibrando {self.best_model_name}...")
        
        # Crear modelo calibrado
        calibrated_model = CalibratedClassifierCV(
            self.best_model, method='isotonic', cv=3
        )
        
        # Entrenar modelo calibrado
        calibrated_model.fit(X, y)
        
        # Actualizar mejor modelo con versión calibrada
        self.best_model = calibrated_model
        
        logger.info("Calibración completada")
        
        return calibrated_model
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """Obtiene importancia de features del mejor modelo"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado.")
        
        # Obtener importancia según el tipo de modelo
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_[0])
        else:
            logger.warning("El modelo no tiene atributo de importancia de features")
            return pd.DataFrame()
        
        # Crear DataFrame de importancia
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_evaluation_metrics(self, save_path: str = None):
        """Crea visualizaciones de métricas de evaluación"""
        if not self.evaluation_results:
            logger.warning("No hay resultados de evaluación para plotear")
            return
        
        # Configurar subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Evaluación de Modelos de ML', fontsize=16)
        
        # Preparar datos para plotting
        model_names = list(self.evaluation_results.keys())
        metrics = ['roc_auc', 'pr_auc', 'f1_score', 'accuracy']
        
        # 1. Comparación de ROC AUC
        roc_scores = [self.evaluation_results[name]['evaluation']['roc_auc'] for name in model_names]
        axes[0, 0].bar(model_names, roc_scores)
        axes[0, 0].set_title('ROC AUC Comparison')
        axes[0, 0].set_ylabel('ROC AUC Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Comparación de PR AUC
        pr_scores = [self.evaluation_results[name]['evaluation']['pr_auc'] for name in model_names]
        axes[0, 1].bar(model_names, pr_scores)
        axes[0, 1].set_title('Precision-Recall AUC Comparison')
        axes[0, 1].set_ylabel('PR AUC Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Comparación de F1 Score
        f1_scores = [self.evaluation_results[name]['evaluation']['f1_score'] for name in model_names]
        axes[1, 0].bar(model_names, f1_scores)
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Heatmap de todas las métricas
        metrics_data = []
        for name in model_names:
            row = [self.evaluation_results[name]['evaluation'][metric] for metric in metrics]
            metrics_data.append(row)
        
        sns.heatmap(
            metrics_data, 
            xticklabels=metrics, 
            yticklabels=model_names,
            annot=True, 
            fmt='.3f',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('All Metrics Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráficos guardados en: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str, include_feature_engineer: bool = True):
        """Guarda el mejor modelo y artefactos"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Preparar artefactos para guardar
        artifacts = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'evaluation_results': self.evaluation_results,
            'training_timestamp': datetime.now().isoformat()
        }
        
        if include_feature_engineer and self.feature_engineer:
            artifacts['feature_engineer'] = self.feature_engineer
        
        # Guardar modelo
        joblib.dump(artifacts, filepath)
        logger.info(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga modelo y artefactos guardados"""
        artifacts = joblib.load(filepath)
        
        self.best_model = artifacts['model']
        self.best_model_name = artifacts['model_name']
        self.best_score = artifacts['best_score']
        self.evaluation_results = artifacts.get('evaluation_results', {})
        
        if 'feature_engineer' in artifacts:
            self.feature_engineer = artifacts['feature_engineer']
        
        logger.info(f"Modelo cargado: {self.best_model_name} (ROC AUC: {self.best_score:.4f})")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones con el mejor modelo"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado. Cargue un modelo primero.")
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities


def train_hiring_prediction_model(df: pd.DataFrame, feature_engineer: Any) -> ModelTrainer:
    """Función principal para entrenar modelo de predicción de contratación"""
    logger.info("Iniciando pipeline de entrenamiento...")
    
    # Crear features
    X = feature_engineer.fit_transform(df)
    y = df['contactado'].values
    
    # Inicializar trainer
    trainer = ModelTrainer()
    trainer.feature_engineer = feature_engineer
    
    # Entrenar modelos
    results = trainer.train_and_evaluate(X, y)
    
    # Validación cruzada
    cv_results = trainer.cross_validate_best_model(X, y)
    
    # Calibrar modelo
    trainer.calibrate_best_model(X, y)
    
    logger.info("Entrenamiento completado exitosamente")
    
    return trainer


if __name__ == "__main__":
    # Test del entrenamiento
    from ml.data.preprocessing import preprocess_data
    from ml.features.feature_engineering import FeatureEngineer
    
    csv_path = "../../postulaciones_sinteticas_500.csv"
    df, _ = preprocess_data(csv_path)
    
    engineer = FeatureEngineer()
    trainer = train_hiring_prediction_model(df, engineer)
    
    print(f"Mejor modelo: {trainer.best_model_name}")
    print(f"ROC AUC: {trainer.best_score:.4f}")