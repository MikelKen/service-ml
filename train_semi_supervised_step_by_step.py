#!/usr/bin/env python3
"""
🎯 ENTRENAMIENTO PASO A PASO DEL MODELO SEMI-SUPERVISADO
Script para entrenar modelo de predicción de estado de postulaciones
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any

# Agregar el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importar módulos locales
from app.ml.preprocessing.postulation_preprocessor import PostulationPreprocessor
from app.ml.models.semi_supervised_predictor import SemiSupervisedPredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SemiSupervisedTrainer:
    """Entrenador completo para el modelo semi-supervisado"""
    
    def __init__(self):
        self.preprocessor = PostulationPreprocessor()
        self.models = {}
        self.training_results = {}
        
        # Configuraciones de modelos a probar
        self.model_configs = [
            {'algorithm': 'label_propagation', 'base_classifier': 'random_forest'},
            {'algorithm': 'label_spreading', 'base_classifier': 'random_forest'},
            {'algorithm': 'self_training', 'base_classifier': 'random_forest'},
            {'algorithm': 'self_training', 'base_classifier': 'svm'},
            {'algorithm': 'self_training', 'base_classifier': 'logistic_regression'}
        ]
        
        # Ratios de datos etiquetados a probar
        self.labeled_ratios = [0.1, 0.2, 0.3, 0.4]
    
    def step_1_generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        PASO 1: Generar datos sintéticos para entrenamiento
        """
        logger.info("🔄 PASO 1: Generando datos sintéticos para entrenamiento")
        
        np.random.seed(42)
        
        # Generar datos de candidatos
        candidates_data = []
        job_titles = [
            'Desarrollador Python Junior', 'Senior React Developer', 'Data Scientist',
            'DevOps Engineer', 'Frontend Developer', 'Backend Developer',
            'Full Stack Developer', 'QA Engineer', 'Project Manager',
            'UI/UX Designer', 'Database Administrator', 'Software Architect'
        ]
        
        education_levels = [
            'Técnico en Sistemas', 'Licenciatura en Informática', 
            'Ingeniería en Sistemas', 'Maestría en Computer Science',
            'Ingeniería en Software', 'Licenciatura en Matemáticas'
        ]
        
        locations = ['Santa Cruz', 'La Paz', 'Cochabamba', 'Tarija', 'Sucre']
        
        for i in range(n_samples):
            # Características del candidato
            years_experience = np.random.randint(0, 15)
            education_level = np.random.choice(education_levels)
            
            # Skills técnicas
            tech_skills = np.random.choice([
                'Python, Django, SQL, Git',
                'JavaScript, React, Node.js, MongoDB',
                'Java, Spring Boot, MySQL, Docker',
                'Python, Machine Learning, TensorFlow, SQL',
                'Angular, TypeScript, .NET, SQL Server',
                'Vue.js, Laravel, PHP, PostgreSQL',
                'React Native, iOS, Android, Firebase'
            ])
            
            # Idiomas
            languages = np.random.choice([
                'Español (nativo), Inglés (intermedio)',
                'Español (nativo), Inglés (avanzado)',
                'Español (nativo), Inglés (básico)',
                'Español (nativo), Inglés (avanzado), Portugués (básico)'
            ])
            
            # Certificaciones
            has_certs = np.random.random() > 0.6
            certifications = np.random.choice([
                'AWS Certified Developer, Scrum Master',
                'Google Cloud Professional, Azure Fundamentals',
                'MongoDB Certified Developer',
                'Oracle Certified Java Programmer',
                ''
            ]) if has_certs else ''
            
            current_position = np.random.choice([
                'Junior Developer', 'Senior Developer', 'Lead Developer',
                'Software Engineer', 'Data Analyst', 'QA Engineer'
            ])
            
            # Características de la oferta
            job_title = np.random.choice(job_titles)
            salary = np.random.randint(5000, 25000)
            location = np.random.choice(locations)
            
            # Requisitos de la oferta
            requirements = np.random.choice([
                'Experiencia en Python, Django, conocimientos en SQL, Git',
                'React, JavaScript, experiencia en desarrollo frontend',
                'Java, Spring Framework, bases de datos relacionales',
                'Machine Learning, Python, TensorFlow, estadística',
                'Angular, TypeScript, desarrollo web, APIs REST',
                'PHP, Laravel, MySQL, desarrollo backend'
            ])
            
            # GENERAR TARGET BASADO EN LÓGICA DE NEGOCIO
            # Factores que influyen en aceptación:
            match_score = 0
            
            # 1. Matching de experiencia con seniority del puesto
            if 'Senior' in job_title and years_experience >= 5:
                match_score += 0.3
            elif 'Junior' in job_title and years_experience <= 3:
                match_score += 0.3
            elif 'Lead' in job_title and years_experience >= 7:
                match_score += 0.3
            
            # 2. Matching de tecnologías
            candidate_skills_lower = tech_skills.lower()
            requirements_lower = requirements.lower()
            
            tech_matches = 0
            key_techs = ['python', 'java', 'javascript', 'react', 'angular', 'sql']
            for tech in key_techs:
                if tech in requirements_lower and tech in candidate_skills_lower:
                    tech_matches += 1
            
            if tech_matches >= 2:
                match_score += 0.25
            elif tech_matches >= 1:
                match_score += 0.15
            
            # 3. Nivel de inglés
            if 'avanzado' in languages:
                match_score += 0.15
            elif 'intermedio' in languages:
                match_score += 0.1
            
            # 4. Certificaciones
            if certifications:
                match_score += 0.1
            
            # 5. Educación apropiada
            if any(edu in education_level.lower() for edu in ['ingeniería', 'maestría']):
                match_score += 0.1
            
            # 6. Agregar algo de ruido aleatorio
            match_score += np.random.normal(0, 0.1)
            
            # Determinar target (umbral ajustable)
            target = 1 if match_score > 0.5 else 0
            
            # Crear registro
            record = {
                'candidate_id': f'candidate_{i}',
                'years_experience': years_experience,
                'education_level': education_level,
                'skills': tech_skills,
                'languages': languages,
                'certifications': certifications,
                'current_position': current_position,
                'offer_id': f'offer_{i}',
                'job_title': job_title,
                'salary': salary,
                'location': location,
                'requirements': requirements,
                'company_id': f'company_{i % 100}',
                'target': target,
                'match_score': match_score
            }
            
            candidates_data.append(record)
        
        df = pd.DataFrame(candidates_data)
        
        # Mostrar estadísticas
        logger.info(f"✅ Datos generados: {len(df)} registros")
        logger.info(f"📊 Distribución del target:")
        logger.info(f"  - ACEPTADO (1): {(df['target'] == 1).sum()} ({(df['target'] == 1).mean()*100:.1f}%)")
        logger.info(f"  - RECHAZADO (0): {(df['target'] == 0).sum()} ({(df['target'] == 0).mean()*100:.1f}%)")
        
        return df
    
    def step_2_preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        PASO 2: Preprocessar los datos
        """
        logger.info("🔄 PASO 2: Preprocessando los datos")
        
        # Extraer target
        y = df['target'].values
        
        # Remover target y columnas auxiliares del DataFrame
        X_df = df.drop(['target', 'match_score', 'candidate_id', 'offer_id', 'company_id'], axis=1)
        
        # Preprocessar
        X = self.preprocessor.fit_transform(X_df)
        feature_names = self.preprocessor.get_feature_names()
        
        logger.info(f"✅ Datos preprocessados:")
        logger.info(f"  - Características: {X.shape[1]}")
        logger.info(f"  - Muestras: {X.shape[0]}")
        
        return X, y, feature_names
    
    def step_3_train_models(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str]) -> Dict[str, Any]:
        """
        PASO 3: Entrenar múltiples modelos semi-supervisados
        """
        logger.info("🔄 PASO 3: Entrenando modelos semi-supervisados")
        
        results = {}
        
        for config in self.model_configs:
            for labeled_ratio in self.labeled_ratios:
                model_name = f"{config['algorithm']}_{config['base_classifier']}_ratio_{labeled_ratio}"
                
                logger.info(f"🚀 Entrenando: {model_name}")
                
                try:
                    # Crear y entrenar modelo
                    model = SemiSupervisedPredictor(
                        algorithm=config['algorithm'],
                        base_classifier=config['base_classifier']
                    )
                    
                    model.fit(X, y, labeled_ratio=labeled_ratio, feature_names=feature_names)
                    
                    # Guardar modelo y resultados
                    self.models[model_name] = model
                    results[model_name] = {
                        'model': model,
                        'config': config,
                        'labeled_ratio': labeled_ratio,
                        'metrics': model.metrics,
                        'summary': model.get_model_summary()
                    }
                    
                    logger.info(f"✅ {model_name} entrenado exitosamente")
                    
                except Exception as e:
                    logger.error(f"❌ Error entrenando {model_name}: {e}")
                    continue
        
        logger.info(f"✅ Entrenamiento completado: {len(results)} modelos")
        return results
    
    def step_4_evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        PASO 4: Evaluar y comparar modelos
        """
        logger.info("🔄 PASO 4: Evaluando modelos")
        
        # Dividir datos para evaluación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Predicciones
                y_pred = model.predict(X_test)
                
                # Métricas
                accuracy = (y_pred == y_test).mean()
                
                # Reporte de clasificación
                report = classification_report(
                    y_test, y_pred, 
                    target_names=['RECHAZADO', 'ACEPTADO'],
                    output_dict=True,
                    zero_division=0
                )
                
                evaluation_results[model_name] = {
                    'accuracy': accuracy,
                    'classification_report': report,
                    'test_samples': len(y_test),
                    'model_summary': model.get_model_summary()
                }
                
                logger.info(f"📊 {model_name}: Accuracy = {accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Error evaluando {model_name}: {e}")
                continue
        
        return evaluation_results
    
    def step_5_select_best_model(self, evaluation_results: Dict[str, Any]) -> str:
        """
        PASO 5: Seleccionar el mejor modelo
        """
        logger.info("🔄 PASO 5: Seleccionando mejor modelo")
        
        best_model_name = None
        best_score = -1
        
        # Criterio: accuracy ponderado por robustez
        for model_name, results in evaluation_results.items():
            accuracy = results['accuracy']
            
            # Considerar también el balanced accuracy
            report = results['classification_report']
            if 'macro avg' in report:
                f1_macro = report['macro avg']['f1-score']
                score = 0.7 * accuracy + 0.3 * f1_macro
            else:
                score = accuracy
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        logger.info(f"🏆 Mejor modelo: {best_model_name} (Score: {best_score:.3f})")
        
        return best_model_name
    
    def step_6_save_best_model(self, best_model_name: str):
        """
        PASO 6: Guardar el mejor modelo
        """
        logger.info("🔄 PASO 6: Guardando mejor modelo")
        
        # Crear directorios
        model_dir = os.path.join(current_dir, 'trained_models', 'semi_supervised')
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar modelo
        best_model = self.models[best_model_name]
        model_path = os.path.join(model_dir, f'semi_supervised_model_{timestamp}.pkl')
        best_model.save_model(model_path)
        
        # Guardar preprocessor
        preprocessor_path = os.path.join(model_dir, f'preprocessor_{timestamp}.pkl')
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        # Guardar resumen de entrenamiento
        summary = {
            'best_model': best_model_name,
            'training_date': timestamp,
            'model_config': best_model.get_model_summary(),
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'feature_names': self.preprocessor.get_feature_names()
        }
        
        summary_path = os.path.join(model_dir, f'training_summary_{timestamp}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Modelo guardado en: {model_path}")
        logger.info(f"💾 Preprocessor guardado en: {preprocessor_path}")
        logger.info(f"💾 Resumen guardado en: {summary_path}")
        
        return model_path, preprocessor_path, summary_path
    
    def run_complete_training(self, n_samples: int = 1000):
        """
        Ejecuta el entrenamiento completo paso a paso
        """
        logger.info("🚀 INICIANDO ENTRENAMIENTO COMPLETO DEL MODELO SEMI-SUPERVISADO")
        logger.info("=" * 80)
        
        try:
            # PASO 1: Generar datos
            df = self.step_1_generate_synthetic_data(n_samples)
            
            # PASO 2: Preprocessar
            X, y, feature_names = self.step_2_preprocess_data(df)
            
            # PASO 3: Entrenar modelos
            training_results = self.step_3_train_models(X, y, feature_names)
            
            # PASO 4: Evaluar modelos
            evaluation_results = self.step_4_evaluate_models(X, y)
            
            # PASO 5: Seleccionar mejor modelo
            best_model_name = self.step_5_select_best_model(evaluation_results)
            
            # PASO 6: Guardar mejor modelo
            model_path, preprocessor_path, summary_path = self.step_6_save_best_model(best_model_name)
            
            logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            
            return {
                'success': True,
                'best_model': best_model_name,
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'summary_path': summary_path,
                'evaluation_results': evaluation_results
            }
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Función principal"""
    print("🎯 ENTRENADOR DE MODELO SEMI-SUPERVISADO PARA POSTULACIONES")
    print("=" * 80)
    
    trainer = SemiSupervisedTrainer()
    
    # Ejecutar entrenamiento
    results = trainer.run_complete_training(n_samples=1500)
    
    if results['success']:
        print(f"✅ Entrenamiento exitoso!")
        print(f"🏆 Mejor modelo: {results['best_model']}")
        print(f"💾 Modelo guardado: {results['model_path']}")
    else:
        print(f"❌ Error en entrenamiento: {results['error']}")

if __name__ == "__main__":
    main()