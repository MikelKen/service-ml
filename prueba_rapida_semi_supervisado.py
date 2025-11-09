#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que el entrenamiento semi-supervisado funcione
"""

import asyncio
import sys
from pathlib import Path

# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent))

async def test_semi_supervised_training():
    """Prueba rápida del entrenamiento semi-supervisado"""
    
    print("🧪 PRUEBA RÁPIDA DE ENTRENAMIENTO SEMI-SUPERVISADO")
    print("=" * 60)
    
    try:
        # Importar componentes necesarios
        from app.ml.data.postgres_extractor import PostgresExtractor
        from app.ml.preprocessing.semi_supervised_preprocessor import SemiSupervisedPreprocessor
        from app.ml.models.semi_supervised_model import SemiSupervisedPostulacionModel
        
        print("✅ Importaciones exitosas")
        
        # Paso 1: Verificar datos
        print("\n📊 Verificando datos...")
        extractor = PostgresExtractor()
        
        # Verificar distribución de estados
        distribution = await extractor.extract_estado_distribution()
        total_labeled = sum(distribution.values())
        
        # Verificar datos no etiquetados
        unlabeled_data = await extractor.extract_missing_estado_postulaciones()
        total_unlabeled = len(unlabeled_data)
        
        print(f"  📋 Datos etiquetados: {total_labeled}")
        print(f"  📋 Datos no etiquetados: {total_unlabeled}")
        
        if total_unlabeled == 0:
            print("  ⚠️  No hay datos no etiquetados")
            print("  💡 Ejecuta: python crear_datos_semi_supervisados.py create")
            return False
        
        # Paso 2: Extraer muestra pequeña para prueba
        print("\\n🔄 Extrayendo muestra de datos...")
        sample_data = await extractor.extract_postulaciones_with_features(limit=1000)
        print(f"  📦 Muestra extraída: {len(sample_data)} registros")
        
        # Paso 3: Preprocesar
        print("\\n🔧 Preprocesando datos...")
        preprocessor = SemiSupervisedPreprocessor()
        X, y, X_unlabeled, label_encoder = preprocessor.fit_transform(sample_data)
        
        print(f"  ✅ Datos etiquetados: {X.shape}")
        print(f"  ✅ Datos no etiquetados: {X_unlabeled.shape}")
        print(f"  ✅ Clases encontradas: {len(label_encoder.classes_)}")
        
        # Paso 4: Probar entrenamiento con un modelo
        print("\\n🤖 Probando entrenamiento (Label Spreading)...")
        model = SemiSupervisedPostulacionModel(model_type='label_spreading')
        
        # Entrenar con muestra pequeña
        metrics = model.train(X, y, X_unlabeled, validation_split=0.2)
        
        print("  ✅ Entrenamiento exitoso!")
        print(f"  📈 Precisión de entrenamiento: {metrics.get('train_accuracy', 'N/A'):.4f}")
        if 'val_accuracy' in metrics:
            print(f"  📈 Precisión de validación: {metrics['val_accuracy']:.4f}")
        
        # Paso 5: Probar predicción
        print("\\n🔮 Probando predicción...")
        if len(X_unlabeled) > 0:
            sample_unlabeled = X_unlabeled[:1]  # Solo una muestra
            prediction, confidence = model.predict_single(sample_unlabeled, preprocessor)
            print(f"  ✅ Predicción: {prediction} (confianza: {confidence:.4f})")
        
        print("\\n🎉 ¡Prueba exitosa! El sistema funciona correctamente.")
        return True
        
    except Exception as e:
        print(f"\\n❌ Error en la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal"""
    success = await test_semi_supervised_training()
    
    if success:
        print("\\n" + "=" * 60)
        print("✨ SISTEMA LISTO PARA USO")
        print("=" * 60)
        print("🚀 Comandos siguientes:")
        print("  1. python train_semi_supervised_step_by_step.py  # Entrenamiento completo")
        print("  2. uvicorn app.main:app --port 3001 --reload     # Servidor GraphQL")
        print("  3. http://localhost:3001/graphql                 # Interfaz GraphQL")
    else:
        print("\\n" + "=" * 60)
        print("⚠️  REVISAR CONFIGURACIÓN")
        print("=" * 60)
        print("🔧 Pasos recomendados:")
        print("  1. python crear_datos_semi_supervisados.py create  # Crear datos no etiquetados")
        print("  2. python verificar_sistema.py                     # Verificar sistema completo")

if __name__ == "__main__":
    asyncio.run(main())