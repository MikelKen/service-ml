"""
Entrenador simple para el modelo de contratación con datos realistas
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Carga y prepara datos realistas"""
    # Generar datos realistas si no existen
    if not os.path.exists('datos_entrenamiento_realista.csv'):
        logger.info("Generando datos realistas...")
        os.system("python generate_realistic_data.py")
    
    # Cargar datos
    df = pd.read_csv('datos_entrenamiento_realista.csv')
    logger.info(f"Datos cargados: {df.shape[0]} registros")
    
    return df

def create_features(df):
    """Crea features mejoradas para el modelo"""
    features = pd.DataFrame()
    
    # Features numéricas básicas
    features['años_experiencia'] = df['años_experiencia']
    features['salario'] = df['salario']
    
    # Feature de experiencia optimizada (penalizar extremos)
    features['exp_optimal'] = df['años_experiencia'].apply(
        lambda x: 1.0 if 3 <= x <= 12 else 0.8 if 1 <= x <= 2 or 13 <= x <= 15 else 0.5
    )
    
    # Educación numérica
    education_map = {'técnico': 1, 'licenciatura': 2, 'maestría': 3, 'doctorado': 4}
    features['nivel_educacion_num'] = df['nivel_educacion'].map(education_map)
    
    # Calcular skill match mejorado
    def calculate_skill_match(row):
        skills = str(row['habilidades']).lower().split(', ')
        reqs = str(row['requisitos']).lower().split(', ')
        
        skills_set = set([s.strip() for s in skills if s.strip()])
        reqs_set = set([r.strip() for r in reqs if r.strip()])
        
        if not reqs_set:
            return 0
        
        intersection = skills_set.intersection(reqs_set)
        return len(intersection) / len(reqs_set)
    
    features['skill_match'] = df.apply(calculate_skill_match, axis=1)
    
    # Features temporales
    df['fecha_postulacion'] = pd.to_datetime(df['fecha_postulacion'])
    df['fecha_publicacion'] = pd.to_datetime(df['fecha_publicacion'])
    features['dias_desde_publicacion'] = (df['fecha_postulacion'] - df['fecha_publicacion']).dt.days
    
    # Certificaciones
    features['tiene_certificaciones'] = (df['certificaciones'] != 'sin certificacion').astype(int)
    
    # Número de habilidades
    features['num_habilidades'] = df['habilidades'].apply(
        lambda x: len([s.strip() for s in str(x).split(',') if s.strip()])
    )
    
    # Salario vs experiencia (detectar candidatos muy caros o muy baratos)
    features['salario_por_exp'] = df['salario'] / (df['años_experiencia'] + 1)  # +1 para evitar división por 0
    
    # Feature de salario normalizado por puesto
    salary_by_job = df.groupby('titulo')['salario'].median()
    features['salario_relativo'] = df.apply(
        lambda row: row['salario'] / salary_by_job[row['titulo']], axis=1
    )
    
    # Mes de postulación (seasonality)
    features['mes'] = df['fecha_postulacion'].dt.month
    
    # Días de la semana
    features['dia_semana'] = df['fecha_postulacion'].dt.dayofweek
    
    logger.info(f"Features creadas: {list(features.columns)}")
    return features

def train_model():
    """Entrena el modelo con datos realistas"""
    print("🚀 Iniciando entrenamiento con datos realistas...")
    
    # Cargar datos
    df = load_and_prepare_data()
    
    # Crear features
    X = create_features(df)
    y = df['contactado']
    
    logger.info(f"Distribución del target: {y.value_counts().to_dict()}")
    logger.info(f"Tasa de contratación: {y.mean():.2%}")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo (RandomForest es más robusto para este tipo de datos)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Para manejar desbalance
    )
    
    logger.info("Entrenando modelo...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.3f}")
    
    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    print("\n📈 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Importancia de features:")
    print(importance.head(10))
    
    # Crear directorio para modelos
    os.makedirs('trained_models', exist_ok=True)
    
    # Guardar modelo y artefactos
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'model_name': 'RandomForestClassifier',
        'accuracy': accuracy,
        'feature_importance': importance.to_dict('records')
    }
    
    model_path = 'trained_models/simple_hiring_model.pkl'
    joblib.dump(model_artifacts, model_path)
    
    print(f"\n✅ Modelo entrenado y guardado en: {model_path}")
    print(f"📊 Accuracy final: {accuracy:.1%}")
    
    # Probar con ejemplos realistas
    print("\n🧪 Probando con ejemplos:")
    test_examples = [
        # Candidato ideal
        pd.DataFrame([{
            'años_experiencia': 5, 'salario': 18000, 'exp_optimal': 1.0,
            'nivel_educacion_num': 3, 'skill_match': 0.8, 'dias_desde_publicacion': 3,
            'tiene_certificaciones': 1, 'num_habilidades': 5, 'salario_por_exp': 3600,
            'salario_relativo': 1.0, 'mes': 6, 'dia_semana': 2
        }]),
        # Candidato sobrecalificado/caro
        pd.DataFrame([{
            'años_experiencia': 15, 'salario': 35000, 'exp_optimal': 0.5,
            'nivel_educacion_num': 4, 'skill_match': 0.9, 'dias_desde_publicacion': 1,
            'tiene_certificaciones': 1, 'num_habilidades': 8, 'salario_por_exp': 2333,
            'salario_relativo': 1.8, 'mes': 6, 'dia_semana': 2
        }]),
        # Candidato junior sin experiencia
        pd.DataFrame([{
            'años_experiencia': 0, 'salario': 8000, 'exp_optimal': 0.5,
            'nivel_educacion_num': 2, 'skill_match': 0.3, 'dias_desde_publicacion': 5,
            'tiene_certificaciones': 0, 'num_habilidades': 3, 'salario_por_exp': 8000,
            'salario_relativo': 0.7, 'mes': 6, 'dia_semana': 2
        }])
    ]
    
    for i, example in enumerate(test_examples):
        example_scaled = scaler.transform(example)
        prob = model.predict_proba(example_scaled)[0, 1]
        pred = model.predict(example_scaled)[0]
        print(f"Ejemplo {i+1}: Probabilidad = {prob:.1%}, Predicción = {pred}")
    
    return True

if __name__ == "__main__":
    train_model()