"""
Script de entrenamiento simplificado para el modelo de predicción de contratación
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_prepare_data(csv_path: str):
    """Carga y prepara los datos de manera simple"""
    logger.info(f"Cargando datos desde: {csv_path}")
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Crear variable objetivo simple
    target_mapping = {
        'aceptado': 1,
        'entrevista': 1,
        'contratado': 1,
        'rechazado': 0,
        'en revisión': 0,
        'pendiente': 0,
        'descartado': 0
    }
    
    df['contactado'] = df['estado'].str.lower().map(target_mapping).fillna(0)
    logger.info(f"Distribución del target: {df['contactado'].value_counts().to_dict()}")
    
    # Limpiar y convertir datos básicos
    df['años_experiencia'] = pd.to_numeric(df['años_experiencia'], errors='coerce').fillna(0)
    df['salario'] = pd.to_numeric(df['salario'], errors='coerce').fillna(0)
    
    # Fechas básicas
    df['fecha_postulacion'] = pd.to_datetime(df['fecha_postulacion'], errors='coerce')
    df['fecha_publicacion'] = pd.to_datetime(df['fecha_publicacion'], errors='coerce')
    
    # Feature temporal simple
    df['dias_desde_publicacion'] = (df['fecha_postulacion'] - df['fecha_publicacion']).dt.days.fillna(0)
    
    return df

def create_simple_features(df):
    """Crear features simples sin dependencias complejas"""
    logger.info("Creando features simples...")
    
    features = pd.DataFrame()
    
    # Features numéricas básicas
    features['años_experiencia'] = df['años_experiencia']
    features['salario'] = df['salario']
    features['dias_desde_publicacion'] = df['dias_desde_publicacion']
    
    # Feature de coincidencia de habilidades simple
    def simple_skill_match(row):
        if pd.isna(row['habilidades']) or pd.isna(row['requisitos']):
            return 0
        
        skills = set([s.strip().lower() for s in str(row['habilidades']).split(',') if s.strip()])
        reqs = set([r.strip().lower() for r in str(row['requisitos']).split(',') if r.strip()])
        
        if not skills or not reqs:
            return 0
        
        intersection = skills.intersection(reqs)
        return len(intersection) / len(reqs)
    
    features['skill_match'] = df.apply(simple_skill_match, axis=1)
    
    # Features categóricas simples (encoding ordinal)
    education_map = {'técnico': 1, 'licenciatura': 2, 'maestría': 3, 'doctorado': 4}
    features['nivel_educacion_num'] = df['nivel_educacion'].str.lower().map(education_map).fillna(1)
    
    # Feature de experiencia vs salario
    features['salario_por_exp'] = df.apply(lambda row: row['salario'] / max(row['años_experiencia'], 1), axis=1)
    
    # Features binarias simples
    features['tiene_certificaciones'] = df['certificaciones'].apply(
        lambda x: 1 if pd.notna(x) and str(x).lower() not in ['', 'sin certificacion', 'ninguna'] else 0
    )
    
    # Feature de número de habilidades
    features['num_habilidades'] = df['habilidades'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )
    
    # Feature temporal
    features['mes'] = df['fecha_postulacion'].dt.month.fillna(6)
    
    logger.info(f"Features creadas: {features.shape[1]} columnas")
    
    return features

def train_simple_model(X, y):
    """Entrenar modelo simple usando solo scikit-learn básico"""
    logger.info("Entrenando modelo simple...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
        logger.info(f"Datos de prueba: {X_test.shape[0]} muestras")
        
        # Escalar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelos simples
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            logger.info(f"Entrenando {name}...")
            
            # Entrenar
            model.fit(X_train_scaled, y_train)
            
            # Predecir
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        logger.info(f"Mejor modelo: {best_name} (Accuracy: {best_score:.4f})")
        
        # Guardar modelo simple
        model_artifacts = {
            'model': best_model,
            'scaler': scaler,
            'model_name': best_name,
            'accuracy': best_score,
            'feature_names': X.columns.tolist(),
            'training_date': datetime.now().isoformat()
        }
        
        return model_artifacts
        
    except ImportError as e:
        logger.error(f"Error importando sklearn: {e}")
        logger.info("Por favor instale scikit-learn: pip install scikit-learn")
        return None
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        return None

def save_simple_model(artifacts, output_path):
    """Guardar modelo simple"""
    try:
        import joblib
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar
        joblib.dump(artifacts, output_path)
        logger.info(f"Modelo guardado en: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error guardando modelo: {e}")
        return False

def main():
    """Función principal simplificada"""
    logger.info("=== Entrenamiento Simplificado del Modelo ===")
    
    # Verificar que existe el archivo de datos
    data_path = "postulaciones_sinteticas_500.csv"
    if not os.path.exists(data_path):
        logger.error(f"Archivo de datos no encontrado: {data_path}")
        return False
    
    try:
        # 1. Cargar y preparar datos
        df = load_and_prepare_data(data_path)
        
        # 2. Crear features simples
        X = create_simple_features(df)
        y = df['contactado']
        
        # 3. Entrenar modelo
        artifacts = train_simple_model(X, y)
        
        if artifacts is None:
            logger.error("Error en el entrenamiento")
            return False
        
        # 4. Guardar modelo
        output_path = "trained_models/simple_hiring_model.pkl"
        if save_simple_model(artifacts, output_path):
            logger.info("✅ Entrenamiento completado exitosamente!")
            
            # Mostrar resumen
            logger.info(f"Modelo: {artifacts['model_name']}")
            logger.info(f"Accuracy: {artifacts['accuracy']:.4f}")
            logger.info(f"Features utilizadas: {len(artifacts['feature_names'])}")
            logger.info(f"Archivo guardado: {output_path}")
            
            return True
        else:
            return False
        
    except Exception as e:
        logger.error(f"Error general: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print("📁 Modelo guardado en: trained_models/simple_hiring_model.pkl")
        print("🚀 Ahora puede probar predicciones")
    else:
        print("\n❌ Error durante el entrenamiento")
        print("💡 Revise los logs arriba para diagnosticar el problema")
        sys.exit(1)