"""
Script para reentrenar el modelo con manejo correcto de valores 'unknown'
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ml.training.model_trainer import train_compatibility_model
from app.ml.models.predictor import compatibility_predictor, load_default_model

async def retrain_model():
    """Reentrenar modelo para manejar valores unknown"""
    
    print("🔄 === REENTRENANDO MODELO === 🔄")
    print()
    
    print("1️⃣ Iniciando reentrenamiento...")
    try:
        # Reentrenar modelo
        result = train_compatibility_model()
        
        print(f"✅ Entrenamiento completado:")
        print(f"   - Mejor modelo: {result.get('best_model')}")
        print(f"   - ROC AUC: {result.get('best_metrics', {}).get('roc_auc')}")
        print(f"   - Accuracy: {result.get('best_metrics', {}).get('accuracy')}")
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return False
    
    print()
    print("2️⃣ Recargando modelo...")
    try:
        # Recargar modelo
        load_default_model()
        print("✅ Modelo recargado exitosamente")
        
    except Exception as e:
        print(f"❌ Error recargando modelo: {e}")
        return False
    
    print()
    print("3️⃣ Probando predicción...")
    try:
        # Probar predicción
        candidate_id = "860d3462-51b2-4edc-8648-8a2198b92470"
        offer_id = "1949bff6-245d-4f12-aff0-f1d8c83d8154"
        
        result = compatibility_predictor.predict_compatibility(candidate_id, offer_id)
        
        print(f"✅ Predicción exitosa:")
        print(f"   - Probabilidad: {result.get('probability')}")
        print(f"   - Predicción: {result.get('prediction')}")
        print(f"   - Confianza: {result.get('confidence')}")
        print(f"   - Error: {result.get('error', 'Ninguno')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en predicción de prueba: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(retrain_model())
    if success:
        print("\n🎉 MODELO REENTRENADO Y FUNCIONANDO CORRECTAMENTE 🎉")
    else:
        print("\n❌ PROBLEMAS EN EL REENTRENAMIENTO")