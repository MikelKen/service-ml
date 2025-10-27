"""
Script de demostración del sistema de predicción de contratación
"""
import os
import sys
import asyncio
import json
from datetime import datetime

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.models.predictor import HiringPredictor
from app.services.ml_service import MLService


def print_banner():
    """Imprime banner del sistema"""
    print("=" * 70)
    print("  SISTEMA DE PREDICCIÓN DE CONTRATACIÓN - MICROSERVICIO ML")
    print("=" * 70)
    print()


def print_section(title):
    """Imprime sección"""
    print(f"\n{'='*10} {title} {'='*10}")


async def demo_complete_workflow():
    """Demostración completa del workflow"""
    print_banner()
    
    # 1. Verificar si existe modelo entrenado
    model_path = "trained_models/hiring_prediction_model.pkl"
    
    if not os.path.exists(model_path):
        print_section("ENTRENAMIENTO DEL MODELO")
        print("No se encontró modelo entrenado. Iniciando entrenamiento...")
        
        # Importar y ejecutar entrenamiento
        from train_model import main as train_main
        success = train_main()
        
        if not success:
            print("❌ Error en el entrenamiento. Abortando demo.")
            return
        
        print("✅ Modelo entrenado exitosamente!")
    else:
        print("✅ Modelo existente encontrado.")
    
    # 2. Inicializar servicio de ML
    print_section("INICIALIZANDO SERVICIO ML")
    ml_service = MLService()
    
    if not ml_service.is_model_loaded:
        print("❌ No se pudo cargar el modelo.")
        return
    
    print(f"✅ Modelo cargado: {ml_service.model_info['model_name']}")
    
    # 3. Datos de ejemplo para predicción
    print_section("DATOS DE EJEMPLO")
    
    examples = [
        {
            "name": "Candidato Ideal",
            "application": {
                'nombre': 'María González',
                'años_experiencia': 5,
                'nivel_educacion': 'maestría',
                'habilidades': 'python, machine learning, sql, aws',
                'idiomas': 'español, inglés',
                'certificaciones': 'aws cloud practitioner, scrum master',
                'puesto_actual': 'data scientist',
                'industria': 'tecnología',
                'fecha_postulacion': '2024-01-15'
            },
            "job_offer": {
                'titulo': 'Senior Data Scientist',
                'descripcion': 'Buscamos un data scientist senior con experiencia en ML',
                'salario': 12000,
                'ubicacion': 'santa cruz',
                'requisitos': 'python, machine learning, sql, aws, experiencia en proyectos',
                'fecha_publicacion': '2024-01-10'
            }
        },
        {
            "name": "Candidato Medio",
            "application": {
                'nombre': 'Carlos Pérez',
                'años_experiencia': 2,
                'nivel_educacion': 'licenciatura',
                'habilidades': 'java, sql',
                'idiomas': 'español',
                'certificaciones': '',
                'puesto_actual': 'desarrollador junior',
                'industria': 'finanzas',
                'fecha_postulacion': '2024-01-15'
            },
            "job_offer": {
                'titulo': 'Senior Data Scientist',
                'descripcion': 'Buscamos un data scientist senior con experiencia en ML',
                'salario': 12000,
                'ubicacion': 'santa cruz',
                'requisitos': 'python, machine learning, sql, aws, experiencia en proyectos',
                'fecha_publicacion': '2024-01-10'
            }
        },
        {
            "name": "Candidato Desajustado",
            "application": {
                'nombre': 'Ana Rodríguez',
                'años_experiencia': 1,
                'nivel_educacion': 'técnico',
                'habilidades': 'excel, word',
                'idiomas': 'español',
                'certificaciones': '',
                'puesto_actual': 'asistente administrativo',
                'industria': 'salud',
                'fecha_postulacion': '2024-01-15'
            },
            "job_offer": {
                'titulo': 'Senior Data Scientist',
                'descripcion': 'Buscamos un data scientist senior con experiencia en ML',
                'salario': 12000,
                'ubicacion': 'santa cruz',
                'requisitos': 'python, machine learning, sql, aws, experiencia en proyectos',
                'fecha_publicacion': '2024-01-10'
            }
        }
    ]
    
    # 4. Realizar predicciones
    print_section("PREDICCIONES INDIVIDUALES")
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print(f"   Experiencia: {example['application']['años_experiencia']} años")
        print(f"   Habilidades: {example['application']['habilidades']}")
        print(f"   Puesto actual: {example['application']['puesto_actual']}")
        
        try:
            result = ml_service.predict_hiring_probability(
                example['application'], 
                example['job_offer']
            )
            
            pred = result['hiring_prediction']
            
            print(f"   ✅ Probabilidad de contacto: {pred['probability']:.1%}")
            print(f"   📊 Confianza: {pred['confidence_level']}")
            print(f"   💡 Recomendación: {pred['recommendation']}")
            
            # Mostrar top 3 features más importantes
            if result['feature_importance']:
                print(f"   🔍 Top 3 factores importantes:")
                for j, feat in enumerate(result['feature_importance'][:3]):
                    print(f"      {j+1}. {feat['feature_name']}: {feat['importance']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Error en predicción: {e}")
    
    # 5. Predicción en lote
    print_section("PREDICCIÓN EN LOTE")
    
    batch_data = [
        {
            'application': ex['application'],
            'job_offer': ex['job_offer']
        }
        for ex in examples
    ]
    
    try:
        batch_result = ml_service.predict_batch(batch_data)
        
        print(f"Total de aplicaciones: {batch_result['total_applications']}")
        print(f"Predicciones exitosas: {batch_result['successful_predictions']}")
        print(f"Predicciones fallidas: {batch_result['failed_predictions']}")
        print(f"Tiempo promedio por predicción: {batch_result['predictions'][0]['processing_time_ms']:.1f}ms")
        
    except Exception as e:
        print(f"❌ Error en predicción por lotes: {e}")
    
    # 6. Información del modelo
    print_section("INFORMACIÓN DEL MODELO")
    
    model_info = ml_service.get_model_info()
    print(f"Nombre del modelo: {model_info['model_name']}")
    print(f"Estado: {'Cargado' if model_info['is_loaded'] else 'No cargado'}")
    print(f"Última actualización: {model_info['last_trained']}")
    print(f"Versión: {model_info['version']}")
    
    # 7. Métricas del modelo
    metrics = ml_service.get_model_metrics()
    if metrics:
        print(f"\nMétricas de rendimiento:")
        print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
        print(f"  Precisión: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
    
    # 8. Información del dataset
    print_section("INFORMACIÓN DEL DATASET")
    
    dataset_info = ml_service.get_dataset_info()
    print(f"Total de registros: {dataset_info['total_records']}")
    print(f"Casos positivos (contactados): {dataset_info['positive_class_count']}")
    print(f"Casos negativos (no contactados): {dataset_info['negative_class_count']}")
    print(f"Ratio de balance: {dataset_info['class_balance_ratio']:.2f}")
    
    print_section("DEMO COMPLETADA")
    print("✅ El sistema está funcionando correctamente!")
    print("🚀 El microservicio está listo para recibir consultas GraphQL.")
    print("\nEjemplos de queries GraphQL:")
    print("""
    # Obtener información del modelo
    query {
      modelInfo {
        modelName
        isLoaded
        lastTrained
        version
      }
    }
    
    # Realizar predicción
    query {
      predictHiringProbability(predictionInput: {
        application: {
          nombre: "Juan Pérez"
          añosExperiencia: 5
          nivelEducacion: "licenciatura"
          habilidades: "python, sql"
          idiomas: "español, inglés"
          certificaciones: "aws"
          puestoActual: "desarrollador"
          industria: "tecnología"
        }
        jobOffer: {
          titulo: "Senior Developer"
          descripcion: "Desarrollador senior"
          salario: 10000
          ubicacion: "Santa Cruz"
          requisitos: "python, sql, aws"
        }
      }) {
        hiringPrediction {
          probability
          confidenceLevel
          recommendation
        }
        processingTimeMs
      }
    }
    """)


def demo_simple_prediction():
    """Demostración simple de predicción"""
    print_banner()
    print("Demo rápida de predicción...")
    
    # Verificar si existe modelo
    model_path = "trained_models/hiring_prediction_model.pkl"
    
    if not os.path.exists(model_path):
        print("❌ No se encontró modelo entrenado.")
        print("Ejecute primero: python train_model.py")
        return
    
    # Crear predictor
    try:
        predictor = HiringPredictor(model_path)
        
        # Datos de ejemplo
        test_data = {
            'nombre': 'Test User',
            'años_experiencia': 3,
            'nivel_educacion': 'licenciatura',
            'habilidades': 'python, sql',
            'idiomas': 'español, inglés',
            'certificaciones': 'aws cloud practitioner',
            'puesto_actual': 'desarrollador',
            'industria': 'tecnología',
            'titulo': 'Data Scientist',
            'descripcion': 'Posición de data scientist',
            'salario': 8000,
            'ubicacion': 'santa cruz',
            'requisitos': 'python, machine learning, sql',
            'fecha_postulacion': '2024-01-15',
            'fecha_publicacion': '2024-01-10'
        }
        
        # Realizar predicción
        result = predictor.predict_single(test_data)
        
        print(f"✅ Predicción exitosa!")
        print(f"Probabilidad: {result['probability']:.1%}")
        print(f"Confianza: {result['confidence_level']}")
        print(f"Recomendación: {result['recommendation']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo del sistema de ML")
    parser.add_argument("--simple", action="store_true", help="Demo simple")
    parser.add_argument("--full", action="store_true", help="Demo completa")
    
    args = parser.parse_args()
    
    if args.simple:
        demo_simple_prediction()
    elif args.full:
        asyncio.run(demo_complete_workflow())
    else:
        print("Uso:")
        print("  python run_demo.py --simple   # Demo rápida")
        print("  python run_demo.py --full     # Demo completa")
        print("\nPrimero ejecute: python train_model.py")