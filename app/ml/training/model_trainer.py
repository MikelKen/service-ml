"""
Módulo de entrenamiento de modelos ML para predicción de compatibilidad candidato-oferta
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
import os
from datetime import datetime
import json

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from app.ml.data.data_extractor import data_extractor
from app.ml.preprocessing.mongo_preprocessor import mongo_preprocessor
from app.config.settings import settings

logger = logging.getLogger(__name__)


class CompatibilityModelTrainer:
    """Clase para entrenar modelos de predicción de compatibilidad candidato-oferta"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=settings.ml_random_state,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=settings.ml_random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=settings.ml_random_state,
                class_weight='balanced',
                max_iter=1000
            ),
            'svm': SVC(
                probability=True, 
                random_state=settings.ml_random_state,
                class_weight='balanced'
            )
        }
        
        self.best_model = None
        self.best_model_name = None
        self.model_metrics = {}
        self.feature_importance = {}
        self.training_history = []
    
    def extract_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Extrae datos de MongoDB y los prepara para entrenamiento"""
        logger.info("Extrayendo datos desde MongoDB...")
        
        # Extraer datos de entrenamiento
        training_data = data_extractor.create_training_dataset(
            positive_samples_ratio=0.3,
            negative_samples_multiplier=2
        )
        
        if training_data.empty:
            raise ValueError("No se pudieron obtener datos de entrenamiento")
        
        logger.info(f"Datos extraídos: {len(training_data)} registros")
        
        # Preprocessar datos
        logger.info("Preprocessando datos...")
        processed_data = mongo_preprocessor.preprocess_data(training_data, fit_transformers=True)
        
        # Obtener resumen
        summary = mongo_preprocessor.get_feature_summary(processed_data)
        
        return processed_data, summary
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separa features y target variable"""
        
        # Excluir columnas que no son features
        exclude_columns = ['target', 'candidate_id', 'offer_id', 'created_at']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns]
        y = df['target']
        
        logger.info(f"Features preparadas: {X.shape[1]} columnas, {X.shape[0]} filas")
        logger.info(f"Distribución del target: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Entrena múltiples modelos y selecciona el mejor"""
        
        logger.info("Iniciando entrenamiento de modelos...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=settings.ml_test_size,
            random_state=settings.ml_random_state,
            stratify=y
        )
        
        logger.info(f"Datos de entrenamiento: {X_train.shape}")
        logger.info(f"Datos de prueba: {X_test.shape}")
        
        # Entrenar cada modelo
        model_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Entrenando modelo: {model_name}")
            
            try:
                # Entrenar modelo
                model.fit(X_train, y_train)
                
                # Realizar predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calcular métricas
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Validación cruzada
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=settings.ml_cross_validation_folds,
                    scoring='roc_auc'
                )
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                # Feature importance (si disponible)
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                    # Ordenar por importancia
                    feature_importance = dict(sorted(feature_importance.items(), 
                                                   key=lambda x: x[1], reverse=True))
                    self.feature_importance[model_name] = feature_importance
                
                model_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': self.feature_importance.get(model_name, {})
                }
                
                logger.info(f"{model_name} - ROC AUC: {metrics['roc_auc']:.4f} (+/- {metrics['cv_std']:.4f})")
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {e}")
                continue
        
        # Seleccionar mejor modelo
        self._select_best_model(model_results)
        
        return model_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calcula métricas de evaluación del modelo"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def _select_best_model(self, model_results: Dict[str, Any]):
        """Selecciona el mejor modelo basado en métricas"""
        
        best_score = 0
        best_name = None
        
        for model_name, result in model_results.items():
            # Usar ROC AUC como métrica principal
            score = result['metrics'].get('roc_auc', 0)
            
            if score > best_score:
                best_score = score
                best_name = model_name
        
        if best_name:
            self.best_model = model_results[best_name]['model']
            self.best_model_name = best_name
            self.model_metrics = model_results[best_name]['metrics']
            
            logger.info(f"Mejor modelo seleccionado: {best_name} (ROC AUC: {best_score:.4f})")
        else:
            logger.warning("No se pudo seleccionar un mejor modelo")
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, model_name: str = None) -> Dict[str, Any]:
        """Realiza ajuste de hiperparámetros para el modelo especificado"""
        
        if not settings.ml_enable_hyperparameter_tuning:
            logger.info("Ajuste de hiperparámetros deshabilitado")
            return {}
        
        if model_name is None:
            model_name = self.best_model_name or 'random_forest'
        
        logger.info(f"Realizando ajuste de hiperparámetros para: {model_name}")
        
        # Parámetros por modelo
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No hay parámetros definidos para {model_name}")
            return {}
        
        # Realizar búsqueda en grilla
        base_model = self.models[model_name]
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=3,  # Reducido para velocidad
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        try:
            grid_search.fit(X, y)
            
            # Actualizar mejor modelo
            self.best_model = grid_search.best_estimator_
            self.best_model_name = model_name
            
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Mejores parámetros: {grid_search.best_params_}")
            logger.info(f"Mejor score: {grid_search.best_score_:.4f}")
            
            return tuning_results
            
        except Exception as e:
            logger.error(f"Error en ajuste de hiperparámetros: {e}")
            return {}
    
    def save_model(self, filepath: str, include_preprocessor: bool = True):
        """Guarda el modelo entrenado y metadatos"""
        
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Preparar datos del modelo
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'metrics': self.model_metrics,
            'feature_importance': self.feature_importance.get(self.best_model_name, {}),
            'training_date': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        # Incluir preprocessor si se especifica
        if include_preprocessor and mongo_preprocessor.is_fitted:
            preprocessor_data = {
                'tfidf_skills': mongo_preprocessor.tfidf_skills,
                'tfidf_requirements': mongo_preprocessor.tfidf_requirements,
                'scaler': mongo_preprocessor.scaler,
                'label_encoders': mongo_preprocessor.label_encoders
            }
            model_data['preprocessor'] = preprocessor_data
        
        # Guardar modelo
        joblib.dump(model_data, filepath)
        
        # Guardar métricas en JSON para fácil lectura
        metrics_filepath = filepath.replace('.pkl', '_metrics.json')
        with open(metrics_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': self.best_model_name,
                'metrics': self.model_metrics,
                'feature_importance': dict(list(self.feature_importance.get(self.best_model_name, {}).items())[:10]),
                'training_date': model_data['training_date']
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo guardado en: {filepath}")
        logger.info(f"Métricas guardadas en: {metrics_filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo entrenado"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {filepath}")
        
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.model_metrics = model_data['metrics']
            self.feature_importance = {self.best_model_name: model_data.get('feature_importance', {})}
            
            # Cargar preprocessor si está incluido
            if 'preprocessor' in model_data:
                preprocessor_data = model_data['preprocessor']
                mongo_preprocessor.tfidf_skills = preprocessor_data['tfidf_skills']
                mongo_preprocessor.tfidf_requirements = preprocessor_data['tfidf_requirements']
                mongo_preprocessor.scaler = preprocessor_data['scaler']
                mongo_preprocessor.label_encoders = preprocessor_data['label_encoders']
                mongo_preprocessor.is_fitted = True
            
            logger.info(f"Modelo cargado desde: {filepath}")
            logger.info(f"Modelo: {self.best_model_name}")
            logger.info(f"ROC AUC: {self.model_metrics.get('roc_auc', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def train_full_pipeline(self) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de entrenamiento"""
        
        logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO ===")
        
        try:
            # 1. Extraer y preparar datos
            df, data_summary = self.extract_and_prepare_data()
            
            # 2. Preparar features y target
            X, y = self.prepare_features_and_target(df)
            
            # 3. Entrenar modelos
            model_results = self.train_models(X, y)
            
            # 4. Ajuste de hiperparámetros (opcional)
            tuning_results = self.hyperparameter_tuning(X, y)
            
            # 5. Guardar modelo
            model_path = os.path.join(settings.ml_models_path, "compatibility_model.pkl")
            self.save_model(model_path)
            
            # 6. Guardar preprocessor por separado
            preprocessor_path = os.path.join(settings.ml_models_path, "compatibility_preprocessor.pkl")
            mongo_preprocessor.save_preprocessor(preprocessor_path)
            
            # Resumen del entrenamiento
            training_summary = {
                'data_summary': data_summary,
                'model_results': {name: result['metrics'] for name, result in model_results.items()},
                'best_model': self.best_model_name,
                'best_metrics': self.model_metrics,
                'tuning_results': tuning_results,
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'training_date': datetime.now().isoformat()
            }
            
            # Agregar a historial
            self.training_history.append(training_summary)
            
            logger.info("=== ENTRENAMIENTO COMPLETADO EXITOSAMENTE ===")
            return training_summary
            
        except Exception as e:
            logger.error(f"Error en pipeline de entrenamiento: {e}")
            raise


# Instancia global del trainer
compatibility_trainer = CompatibilityModelTrainer()


def train_compatibility_model() -> Dict[str, Any]:
    """Función conveniente para entrenar el modelo de compatibilidad"""
    return compatibility_trainer.train_full_pipeline()


def load_compatibility_model(filepath: str = None):
    """Función conveniente para cargar el modelo de compatibilidad"""
    if filepath is None:
        filepath = os.path.join(settings.ml_models_path, "compatibility_model.pkl")
    
    compatibility_trainer.load_model(filepath)


if __name__ == "__main__":
    # Test del entrenamiento
    try:
        results = train_compatibility_model()
        print("Entrenamiento completado exitosamente")
        print(f"Mejor modelo: {results['best_model']}")
        print(f"ROC AUC: {results['best_metrics'].get('roc_auc', 'N/A')}")
    except Exception as e:
        print(f"Error en entrenamiento: {e}")