"""
Entrenador de modelo semi-supervisado para predicción de estados de postulaciones
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
import os
from datetime import datetime
import json
import asyncio

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Custom imports
from app.ml.data.postgres_extractor import postgres_extractor
from app.ml.preprocessing.postulaciones_preprocessor import postulaciones_preprocessor
from app.config.settings import settings

logger = logging.getLogger(__name__)


class PostulacionesSemiSupervisedTrainer:
    """Entrenador semi-supervisado para estados de postulaciones"""
    
    def __init__(self):
        self.models = {
            'label_propagation': LabelPropagation(
                kernel='knn',
                n_neighbors=7,
                max_iter=1000,
                tol=1e-3
            ),
            'label_spreading': LabelSpreading(
                kernel='knn',
                n_neighbors=7,
                alpha=0.2,
                max_iter=1000,
                tol=1e-3
            ),
            'self_training_rf': None,  # Se configurará dinámicamente
            'self_training_lr': None   # Se configurará dinámicamente
        }
        
        self.best_model = None
        self.best_model_name = None
        self.model_metrics = {}
        self.training_history = []
        self.feature_importance = {}
        
        # Datos de entrenamiento
        self.labeled_data = None
        self.unlabeled_data = None
        self.complete_data = None
        self.processed_features = None
        
        # Estado del entrenamiento
        self.is_trained = False
    
    async def extract_and_prepare_data(self) -> Dict[str, Any]:
        """Extrae datos de PostgreSQL y los prepara para entrenamiento semi-supervisado"""
        logger.info("=== INICIANDO EXTRACCIÓN DE DATOS ===")
        
        try:
            # 1. Extraer datos desde PostgreSQL y guardar en MongoDB
            logger.info("Extrayendo datos desde PostgreSQL...")
            complete_df = await postgres_extractor.extract_complete_dataset()
            
            if complete_df.empty:
                raise ValueError("No se pudieron extraer datos de PostgreSQL")
            
            # 2. Guardar en MongoDB
            logger.info("Guardando datos en MongoDB...")
            await postgres_extractor.save_to_mongo(complete_df, "postulaciones_completas")
            
            # 3. Preparar datos para semi-supervisado
            logger.info("Preparando datos para aprendizaje semi-supervisado...")
            labeled_data, unlabeled_data, complete_data = postulaciones_preprocessor.prepare_for_semi_supervised(complete_df)
            
            # 4. Preprocesar características
            logger.info("Preprocesando características...")
            processed_features = postulaciones_preprocessor.preprocess_features(
                complete_data, 
                fit_transformers=True
            )
            
            # Guardar datos
            self.labeled_data = labeled_data
            self.unlabeled_data = unlabeled_data
            self.complete_data = complete_data
            self.processed_features = processed_features
            
            # Resumen de datos
            summary = {
                'total_records': len(complete_df),
                'labeled_records': len(labeled_data),
                'unlabeled_records': len(unlabeled_data),
                'features_count': processed_features.shape[1] - 4,  # Excluyendo columnas objetivo
                'feature_names': postulaciones_preprocessor.feature_names,
                'estado_distribution': labeled_data['estado'].value_counts().to_dict(),
                'unlabeled_estados': unlabeled_data['estado'].value_counts().to_dict() if not unlabeled_data.empty else {}
            }
            
            logger.info(f"Datos preparados exitosamente:")
            logger.info(f"  - Total de registros: {summary['total_records']}")
            logger.info(f"  - Registros etiquetados: {summary['labeled_records']}")
            logger.info(f"  - Registros sin etiquetar: {summary['unlabeled_records']}")
            logger.info(f"  - Características: {summary['features_count']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error en preparación de datos: {e}")
            raise
    
    def prepare_semi_supervised_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara matrices X e y para entrenamiento semi-supervisado"""
        if self.processed_features is None:
            raise ValueError("Los datos no han sido procesados. Ejecute extract_and_prepare_data() primero.")
        
        # Obtener características (excluir columnas objetivo)
        feature_columns = [col for col in self.processed_features.columns 
                          if col not in ['estado', 'estado_encoded', 'success_target', 'postulacion_id']]
        
        X = self.processed_features[feature_columns].values
        
        # Preparar y (estados encodificados)
        y = self.processed_features['estado_encoded'].values
        
        # Convertir estados sin etiquetar a -1 (convención scikit-learn)
        y_semi = y.copy()
        y_semi[y_semi == -999] = -1  # Sin etiquetar
        
        logger.info(f"Datos semi-supervisados preparados:")
        logger.info(f"  - Forma de X: {X.shape}")
        logger.info(f"  - Etiquetados: {np.sum(y_semi != -1)}")
        logger.info(f"  - Sin etiquetar: {np.sum(y_semi == -1)}")
        logger.info(f"  - Distribución de etiquetas: {np.unique(y_semi[y_semi != -1], return_counts=True)}")
        
        return X, y_semi
    
    def train_semi_supervised_models(self) -> Dict[str, Any]:
        """Entrena modelos semi-supervisados"""
        logger.info("=== INICIANDO ENTRENAMIENTO SEMI-SUPERVISADO ===")
        
        # Preparar datos
        X, y_semi = self.prepare_semi_supervised_data()
        
        # Separar datos etiquetados para validación
        labeled_mask = y_semi != -1
        X_labeled = X[labeled_mask]
        y_labeled = y_semi[labeled_mask]
        
        # División train/test solo con datos etiquetados
        X_train_labeled, X_test_labeled, y_train_labeled, y_test_labeled = train_test_split(
            X_labeled, y_labeled,
            test_size=0.3,
            random_state=settings.ml_random_state,
            stratify=y_labeled
        )
        
        # Crear conjunto de entrenamiento combinando etiquetados de entrenamiento + sin etiquetar
        unlabeled_mask = y_semi == -1
        X_unlabeled = X[unlabeled_mask]
        
        # Combinar para entrenamiento semi-supervisado
        X_semi_train = np.vstack([X_train_labeled, X_unlabeled])
        y_semi_train = np.hstack([y_train_labeled, np.full(len(X_unlabeled), -1)])
        
        logger.info(f"Conjunto de entrenamiento semi-supervisado:")
        logger.info(f"  - Datos etiquetados para entrenamiento: {len(X_train_labeled)}")
        logger.info(f"  - Datos sin etiquetar: {len(X_unlabeled)}")
        logger.info(f"  - Total para entrenamiento: {len(X_semi_train)}")
        logger.info(f"  - Datos etiquetados para prueba: {len(X_test_labeled)}")
        
        model_results = {}
        
        # 1. Label Propagation
        logger.info("Entrenando Label Propagation...")
        try:
            lp_model = self.models['label_propagation']
            lp_model.fit(X_semi_train, y_semi_train)
            
            # Evaluar en datos de prueba etiquetados
            y_pred_lp = lp_model.predict(X_test_labeled)
            metrics_lp = self._calculate_metrics(y_test_labeled, y_pred_lp)
            
            model_results['label_propagation'] = {
                'model': lp_model,
                'metrics': metrics_lp,
                'predictions_unlabeled': lp_model.predict(X_unlabeled)
            }
            
            logger.info(f"Label Propagation - Accuracy: {metrics_lp['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error entrenando Label Propagation: {e}")
        
        # 2. Label Spreading
        logger.info("Entrenando Label Spreading...")
        try:
            ls_model = self.models['label_spreading']
            ls_model.fit(X_semi_train, y_semi_train)
            
            y_pred_ls = ls_model.predict(X_test_labeled)
            metrics_ls = self._calculate_metrics(y_test_labeled, y_pred_ls)
            
            model_results['label_spreading'] = {
                'model': ls_model,
                'metrics': metrics_ls,
                'predictions_unlabeled': ls_model.predict(X_unlabeled)
            }
            
            logger.info(f"Label Spreading - Accuracy: {metrics_ls['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error entrenando Label Spreading: {e}")
        
        # 3. Self-Training con Random Forest
        logger.info("Entrenando Self-Training con Random Forest...")
        try:
            from sklearn.semi_supervised import SelfTrainingClassifier
            
            base_rf = RandomForestClassifier(
                n_estimators=100,
                random_state=settings.ml_random_state,
                class_weight='balanced'
            )
            
            self_training_rf = SelfTrainingClassifier(
                base_estimator=base_rf,  # Cambiado de base_classifier a base_estimator
                threshold=0.8,
                criterion='threshold',
                max_iter=10
            )
            
            self_training_rf.fit(X_semi_train, y_semi_train)
            
            y_pred_strf = self_training_rf.predict(X_test_labeled)
            metrics_strf = self._calculate_metrics(y_test_labeled, y_pred_strf)
            
            # Feature importance del Random Forest base
            if hasattr(self_training_rf.base_estimator_, 'feature_importances_'):
                feature_importance = dict(zip(
                    postulaciones_preprocessor.feature_names,
                    self_training_rf.base_estimator_.feature_importances_
                ))
                self.feature_importance['self_training_rf'] = dict(sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                ))
            
            model_results['self_training_rf'] = {
                'model': self_training_rf,
                'metrics': metrics_strf,
                'predictions_unlabeled': self_training_rf.predict(X_unlabeled),
                'feature_importance': self.feature_importance.get('self_training_rf', {})
            }
            
            logger.info(f"Self-Training RF - Accuracy: {metrics_strf['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error entrenando Self-Training RF: {e}")
        
        # 4. Self-Training con Logistic Regression
        logger.info("Entrenando Self-Training con Logistic Regression...")
        try:
            from sklearn.semi_supervised import SelfTrainingClassifier
            
            base_lr = LogisticRegression(
                random_state=settings.ml_random_state,
                class_weight='balanced',
                max_iter=1000
            )
            
            self_training_lr = SelfTrainingClassifier(
                base_estimator=base_lr,  # Cambiado de base_classifier a base_estimator
                threshold=0.8,
                criterion='threshold',
                max_iter=10
            )
            
            self_training_lr.fit(X_semi_train, y_semi_train)
            
            y_pred_stlr = self_training_lr.predict(X_test_labeled)
            metrics_stlr = self._calculate_metrics(y_test_labeled, y_pred_stlr)
            
            model_results['self_training_lr'] = {
                'model': self_training_lr,
                'metrics': metrics_stlr,
                'predictions_unlabeled': self_training_lr.predict(X_unlabeled)
            }
            
            logger.info(f"Self-Training LR - Accuracy: {metrics_stlr['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error entrenando Self-Training LR: {e}")
        
        # Seleccionar mejor modelo
        self._select_best_model(model_results)
        
        # Actualizar predicciones en datos sin etiquetar
        self._update_unlabeled_predictions(model_results)
        
        return model_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de evaluación"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Reporte de clasificación
        try:
            cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = cr
        except:
            pass
        
        return metrics
    
    def _select_best_model(self, model_results: Dict[str, Any]):
        """Selecciona el mejor modelo basado en F1-score weighted"""
        best_score = 0
        best_name = None
        
        for model_name, result in model_results.items():
            # Usar F1-score weighted como métrica principal
            score = result['metrics'].get('f1_weighted', 0)
            
            if score > best_score:
                best_score = score
                best_name = model_name
        
        if best_name:
            self.best_model = model_results[best_name]['model']
            self.best_model_name = best_name
            self.model_metrics = model_results[best_name]['metrics']
            
            logger.info(f"Mejor modelo seleccionado: {best_name}")
            logger.info(f"F1-score weighted: {best_score:.4f}")
            logger.info(f"Accuracy: {self.model_metrics['accuracy']:.4f}")
        else:
            logger.warning("No se pudo seleccionar un mejor modelo")
    
    def _update_unlabeled_predictions(self, model_results: Dict[str, Any]):
        """Actualiza las predicciones para datos sin etiquetar"""
        if not self.best_model_name or self.best_model_name not in model_results:
            logger.warning("No hay mejor modelo para hacer predicciones")
            return
        
        best_result = model_results[self.best_model_name]
        predictions = best_result['predictions_unlabeled']
        
        # Mapeo inverso de estados
        inverse_mapping = {v: k for k, v in postulaciones_preprocessor.estado_mapping.items()}
        
        # Obtener índices de datos sin etiquetar
        unlabeled_mask = self.processed_features['estado_encoded'] == -999
        
        if unlabeled_mask.sum() == 0:
            logger.warning("No hay datos sin etiquetar para predecir")
            self.unlabeled_predictions = pd.DataFrame()
            return
        
        unlabeled_indices = self.processed_features[unlabeled_mask].index
        
        if len(predictions) != len(unlabeled_indices):
            logger.warning(f"Mismatch en cantidad de predicciones: {len(predictions)} vs {len(unlabeled_indices)}")
            # Tomar el mínimo para evitar errores
            min_len = min(len(predictions), len(unlabeled_indices))
            predictions = predictions[:min_len]
            unlabeled_indices = unlabeled_indices[:min_len]
        
        # Crear DataFrame con predicciones
        predictions_df = pd.DataFrame({
            'postulacion_id': self.processed_features.loc[unlabeled_indices, 'postulacion_id'].values,
            'estado_original': self.processed_features.loc[unlabeled_indices, 'estado'].values,
            'estado_predicho_encoded': predictions,
            'estado_predicho': [inverse_mapping.get(pred, 'unknown') for pred in predictions]
        })
        
        logger.info(f"Predicciones realizadas para {len(predictions_df)} registros sin etiquetar")
        logger.info(f"Distribución de predicciones: {predictions_df['estado_predicho'].value_counts().to_dict()}")
        
        # Guardar predicciones
        self.unlabeled_predictions = predictions_df
    
    async def save_predictions_to_mongo(self):
        """Guarda las predicciones en MongoDB"""
        if not hasattr(self, 'unlabeled_predictions') or self.unlabeled_predictions.empty:
            logger.warning("No hay predicciones para guardar")
            return
        
        try:
            from app.config.connection import mongodb
            
            # Conectar a MongoDB si no está conectado
            if mongodb.database is None:
                await mongodb.connect()
            
            collection = mongodb.get_collection("postulaciones_predictions")
            
            # Agregar metadata
            predictions_with_meta = self.unlabeled_predictions.copy()
            predictions_with_meta['model_used'] = self.best_model_name
            predictions_with_meta['prediction_date'] = datetime.now().isoformat()
            predictions_with_meta['model_accuracy'] = self.model_metrics.get('accuracy', 0)
            
            # Convertir a registros
            records = predictions_with_meta.to_dict('records')
            
            # Limpiar colección existente
            await collection.delete_many({})
            
            # Insertar predicciones
            if records:
                await collection.insert_many(records)
                logger.info(f"Guardadas {len(records)} predicciones en MongoDB")
            
        except Exception as e:
            logger.error(f"Error guardando predicciones en MongoDB: {e}")
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado y todos los componentes"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Preparar datos del modelo
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'metrics': self.model_metrics,
            'feature_importance': self.feature_importance.get(self.best_model_name, {}),
            'preprocessor': {
                'column_transformer': postulaciones_preprocessor.column_transformer,
                'tfidf_vectorizers': postulaciones_preprocessor.tfidf_vectorizers,
                'feature_names': postulaciones_preprocessor.feature_names,
                'estado_mapping': postulaciones_preprocessor.estado_mapping,
                'labeled_states': postulaciones_preprocessor.labeled_states,
                'unlabeled_states': postulaciones_preprocessor.unlabeled_states
            },
            'training_date': datetime.now().isoformat(),
            'version': '1.0.0',
            'model_type': 'semi_supervised'
        }
        
        # Agregar predicciones si existen
        if hasattr(self, 'unlabeled_predictions'):
            model_data['unlabeled_predictions'] = self.unlabeled_predictions.to_dict('records')
        
        # Guardar modelo
        joblib.dump(model_data, filepath)
        
        # Guardar métricas en JSON
        metrics_filepath = filepath.replace('.pkl', '_metrics.json')
        metrics_summary = {
            'model_name': self.best_model_name,
            'model_type': 'semi_supervised',
            'metrics': self.model_metrics,
            'feature_importance': dict(list(self.feature_importance.get(self.best_model_name, {}).items())[:10]),
            'training_date': model_data['training_date'],
            'data_summary': {
                'labeled_count': len(self.labeled_data) if self.labeled_data is not None else 0,
                'unlabeled_count': len(self.unlabeled_data) if self.unlabeled_data is not None else 0,
                'features_count': len(postulaciones_preprocessor.feature_names)
            }
        }
        
        with open(metrics_filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo semi-supervisado guardado en: {filepath}")
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
            
            # Cargar preprocesador
            if 'preprocessor' in model_data:
                preprocessor_data = model_data['preprocessor']
                postulaciones_preprocessor.column_transformer = preprocessor_data['column_transformer']
                postulaciones_preprocessor.tfidf_vectorizers = preprocessor_data['tfidf_vectorizers']
                postulaciones_preprocessor.feature_names = preprocessor_data['feature_names']
                postulaciones_preprocessor.estado_mapping = preprocessor_data['estado_mapping']
                postulaciones_preprocessor.labeled_states = preprocessor_data['labeled_states']
                postulaciones_preprocessor.unlabeled_states = preprocessor_data['unlabeled_states']
                postulaciones_preprocessor.is_fitted = True
            
            # Cargar predicciones si existen
            if 'unlabeled_predictions' in model_data:
                self.unlabeled_predictions = pd.DataFrame(model_data['unlabeled_predictions'])
            
            self.is_trained = True
            
            logger.info(f"Modelo semi-supervisado cargado desde: {filepath}")
            logger.info(f"Modelo: {self.best_model_name}")
            logger.info(f"Accuracy: {self.model_metrics.get('accuracy', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    async def train_full_pipeline(self) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de entrenamiento semi-supervisado"""
        logger.info("=== INICIANDO PIPELINE SEMI-SUPERVISADO ===")
        
        try:
            # 1. Extraer y preparar datos
            data_summary = await self.extract_and_prepare_data()
            
            # 2. Entrenar modelos semi-supervisados
            model_results = self.train_semi_supervised_models()
            
            # 3. Guardar predicciones en MongoDB
            await self.save_predictions_to_mongo()
            
            # 4. Guardar modelo
            model_path = os.path.join(settings.ml_models_path, "postulaciones", "semi_supervised_model.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.save_model(model_path)
            
            # 5. Guardar preprocessor por separado
            preprocessor_path = os.path.join(settings.ml_models_path, "postulaciones", "semi_supervised_preprocessor.pkl")
            postulaciones_preprocessor.save_preprocessor(preprocessor_path)
            
            # Resumen del entrenamiento
            training_summary = {
                'data_summary': data_summary,
                'model_results': {name: result['metrics'] for name, result in model_results.items()},
                'best_model': self.best_model_name,
                'best_metrics': self.model_metrics,
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'predictions_count': len(self.unlabeled_predictions) if hasattr(self, 'unlabeled_predictions') else 0,
                'training_date': datetime.now().isoformat(),
                'model_type': 'semi_supervised'
            }
            
            # Agregar a historial
            self.training_history.append(training_summary)
            
            self.is_trained = True
            
            logger.info("=== ENTRENAMIENTO SEMI-SUPERVISADO COMPLETADO ===")
            return training_summary
            
        except Exception as e:
            logger.error(f"Error en pipeline semi-supervisado: {e}")
            raise
    
    def predict_estado(self, postulacion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predice el estado de una postulación individual"""
        if not self.is_trained or self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        try:
            # Convertir a DataFrame
            df = pd.DataFrame([postulacion_data])
            
            # Preprocesar
            processed_df = postulaciones_preprocessor.preprocess_features(df, fit_transformers=False)
            
            # Obtener características
            feature_columns = [col for col in processed_df.columns 
                             if col not in ['estado', 'estado_encoded', 'success_target', 'postulacion_id']]
            
            X = processed_df[feature_columns].values
            
            # Predecir
            prediction = self.best_model.predict(X)[0]
            
            # Obtener probabilidades si está disponible
            probabilities = None
            if hasattr(self.best_model, 'predict_proba'):
                try:
                    proba = self.best_model.predict_proba(X)[0]
                    probabilities = dict(zip(self.best_model.classes_, proba))
                except:
                    pass
            
            # Mapeo inverso
            inverse_mapping = {v: k for k, v in postulaciones_preprocessor.estado_mapping.items()}
            estado_predicho = inverse_mapping.get(prediction, 'unknown')
            
            result = {
                'estado_predicho': estado_predicho,
                'estado_encoded': int(prediction),
                'model_used': self.best_model_name,
                'prediction_date': datetime.now().isoformat(),
                'probabilities': probabilities
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise


# Instancia global
postulaciones_trainer = PostulacionesSemiSupervisedTrainer()


async def train_semi_supervised_model() -> Dict[str, Any]:
    """Función conveniente para entrenar el modelo semi-supervisado"""
    return await postulaciones_trainer.train_full_pipeline()


def load_semi_supervised_model(filepath: str = None):
    """Función conveniente para cargar el modelo semi-supervisado"""
    if filepath is None:
        filepath = os.path.join(settings.ml_models_path, "postulaciones", "semi_supervised_model.pkl")
    
    postulaciones_trainer.load_model(filepath)


if __name__ == "__main__":
    async def main():
        try:
            logger.info("Iniciando entrenamiento de modelo semi-supervisado")
            results = await train_semi_supervised_model()
            
            print("=== ENTRENAMIENTO COMPLETADO ===")
            print(f"Mejor modelo: {results['best_model']}")
            print(f"Accuracy: {results['best_metrics'].get('accuracy', 'N/A'):.4f}")
            print(f"F1-score weighted: {results['best_metrics'].get('f1_weighted', 'N/A'):.4f}")
            print(f"Predicciones realizadas: {results['predictions_count']}")
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            print(f"Error: {e}")
    
    asyncio.run(main())