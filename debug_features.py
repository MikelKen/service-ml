"""
Debug para verificar las features que se están generando
"""
import pandas as pd
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.preprocessing.mongo_preprocessor import mongo_preprocessor
from app.ml.data.data_extractor import data_extractor

def debug_features():
    """Debugger las features generadas"""
    
    print("🔍 DEBUGGING FEATURES GENERADAS")
    print("=" * 50)
    
    # 1. Generar un pequeño dataset de entrenamiento
    print("📊 Generando dataset de entrenamiento...")
    training_df = data_extractor.create_training_dataset()
    
    if training_df.empty:
        print("❌ No se pudo generar dataset de entrenamiento")
        return
    
    print(f"✅ Dataset: {training_df.shape}")
    print(f"📋 Columnas originales: {list(training_df.columns)}")
    
    # 2. Preprocessar datos de entrenamiento
    print("\n🔧 Preprocessando datos de entrenamiento...")
    
    # Reset preprocessor
    mongo_preprocessor.is_fitted = False
    mongo_preprocessor._initialize_transformers()  # Inicializar transformers
    
    df_processed_train = mongo_preprocessor.preprocess_data(training_df.head(10), fit_transformers=True)
    
    print(f"✅ Datos procesados training: {df_processed_train.shape}")
    print(f"📋 Columnas después de preprocessing: {len(df_processed_train.columns)}")
    
    # Identificar las nuevas features
    new_features = [col for col in df_processed_train.columns if col.startswith(('is_junior', 'junior_', 'modern_', 'skills_to_experience'))]
    print(f"🆕 Features mejoradas detectadas: {new_features}")
    
    # 3. Crear datos de predicción
    print("\n🎯 Creando datos de predicción...")
    
    prediction_data = {
        'candidate_id': 'test_junior_candidate',
        'offer_id': 'test_junior_offer',
        'years_experience': 1,
        'education_level': 'Ingeniería de Sistemas',
        'skills': 'Python, Django, PostgreSQL, Git, HTML, CSS, JavaScript',
        'languages': 'Español (Nativo), Inglés (Intermedio)',
        'certifications': 'Python for Everybody Specialization, Web Development Bootcamp',
        'current_position': 'Desarrollador Junior en StartupTech',
        'job_title': 'Desarrollador Backend Junior',
        'salary': 4500.00,
        'location': 'Santa Cruz de la Sierra',
        'requirements': 'Recién graduado en Ingeniería de Sistemas, conocimientos en Python',
        'company_id': 'startup_tech_123',
        'created_at': '2025-11-02T18:30:00'
    }
    
    df_prediction = pd.DataFrame([prediction_data])
    
    print(f"📋 Datos de predicción originales: {df_prediction.shape}")
    print(f"🔍 Columnas: {list(df_prediction.columns)}")
    
    # 4. Preprocessar datos de predicción
    print("\n🔧 Preprocessando datos de predicción...")
    
    df_processed_pred = mongo_preprocessor.preprocess_data(df_prediction, fit_transformers=False)
    
    print(f"✅ Datos procesados predicción: {df_processed_pred.shape}")
    print(f"📋 Columnas después de preprocessing: {len(df_processed_pred.columns)}")
    
    # 5. Comparar features
    print("\n🔍 COMPARACIÓN DE FEATURES:")
    
    features_train = set(df_processed_train.columns)
    features_pred = set(df_processed_pred.columns)
    
    print(f"🏋️ Features en training: {len(features_train)}")
    print(f"🎯 Features en predicción: {len(features_pred)}")
    
    missing_in_pred = features_train - features_pred
    extra_in_pred = features_pred - features_train
    
    if missing_in_pred:
        print(f"\n❌ Features faltantes en predicción ({len(missing_in_pred)}):")
        for i, feature in enumerate(sorted(missing_in_pred)[:10]):  # Solo mostrar las primeras 10
            print(f"   {i+1}. {feature}")
        if len(missing_in_pred) > 10:
            print(f"   ... y {len(missing_in_pred) - 10} más")
    
    if extra_in_pred:
        print(f"\n➕ Features extra en predicción ({len(extra_in_pred)}):")
        for i, feature in enumerate(sorted(extra_in_pred)[:10]):
            print(f"   {i+1}. {feature}")
        if len(extra_in_pred) > 10:
            print(f"   ... y {len(extra_in_pred) - 10} más")
    
    # 6. Verificar features específicas mejoradas
    print("\n🆕 VERIFICACIÓN DE FEATURES MEJORADAS:")
    
    improved_features = ['is_junior', 'junior_education_boost', 'skills_to_experience_ratio', 
                        'junior_cert_boost', 'salary_expectation_realistic', 'modern_tech_score', 
                        'junior_modern_boost']
    
    for feature in improved_features:
        in_train = feature in df_processed_train.columns
        in_pred = feature in df_processed_pred.columns
        train_val = df_processed_train[feature].iloc[0] if in_train else 'N/A'
        pred_val = df_processed_pred[feature].iloc[0] if in_pred else 'N/A'
        
        status = "✅" if in_train and in_pred else "❌"
        print(f"   {status} {feature}: Train={train_val}, Pred={pred_val}")
    
    # 7. Guardar columnas para referencia
    with open('debug_features_train.txt', 'w') as f:
        f.write('\n'.join(sorted(df_processed_train.columns)))
    
    with open('debug_features_pred.txt', 'w') as f:
        f.write('\n'.join(sorted(df_processed_pred.columns)))
    
    print(f"\n💾 Features guardadas en archivos debug_features_*.txt")

if __name__ == "__main__":
    debug_features()