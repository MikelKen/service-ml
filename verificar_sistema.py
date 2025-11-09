#!/usr/bin/env python3
"""
Script de verificación rápida para el modelo semi-supervisado
Ejecuta una serie de pruebas básicas para verificar que todo esté funcionando correctamente
"""

import asyncio
import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent))

async def test_database_connections():
    """Prueba las conexiones a las bases de datos"""
    print("🔍 Verificando conexiones a bases de datos...")
    
    try:
        from app.config.connection import get_connection
        connection = await get_connection()
        if connection:
            await connection.close()
            print("✅ PostgreSQL: Conexión exitosa")
        else:
            print("❌ PostgreSQL: Error de conexión")
            return False
    except Exception as e:
        print(f"❌ PostgreSQL: Error - {e}")
        return False
    
    try:
        from app.config.mongodb_connection import get_mongo_client
        client = await get_mongo_client()
        await client.admin.command('ping')
        print("✅ MongoDB: Conexión exitosa")
    except Exception as e:
        print(f"❌ MongoDB: Error - {e}")
        return False
    
    return True

async def test_data_extraction():
    """Prueba la extracción de datos"""
    print("\n📊 Verificando extracción de datos...")
    
    try:
        from app.ml.data.postgres_extractor import PostgresExtractor
        extractor = PostgresExtractor()
        
        # Probar extracción de resumen
        summary = await extractor.extract_estado_distribution()
        print(f"✅ Distribución de estados extraída: {len(summary)} estados encontrados")
        
        # Probar extracción de muestra de datos
        sample_data = await extractor.extract_postulaciones_with_features(limit=5)
        print(f"✅ Muestra de datos extraída: {len(sample_data)} registros")
        
        return True
    except Exception as e:
        print(f"❌ Error en extracción de datos: {e}")
        return False

def test_model_files():
    """Verifica si existen archivos de modelos entrenados"""
    print("\n🤖 Verificando archivos de modelos...")
    
    model_dir = Path("trained_models/semi_supervised")
    
    if not model_dir.exists():
        print("⚠️  Directorio de modelos no existe. Ejecuta el entrenamiento primero.")
        return False
    
    required_files = [
        "label_propagation_model.pkl",
        "label_spreading_model.pkl", 
        "self_training_model.pkl",
        "preprocessor.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            print(f"✅ {file} encontrado")
        else:
            print(f"❌ {file} no encontrado")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Archivos faltantes: {missing_files}")
        print("💡 Ejecuta 'python train_semi_supervised_step_by_step.py' para generar los modelos")
        return False
    
    return True

def test_imports():
    """Verifica que todas las importaciones funcionen"""
    print("\n📦 Verificando importaciones...")
    
    modules = [
        ("app.ml.models.semi_supervised_model", "SemiSupervisedPostulacionModel"),
        ("app.ml.preprocessing.semi_supervised_preprocessor", "SemiSupervisedPreprocessor"),
        ("app.ml.training.semi_supervised_trainer", "SemiSupervisedTrainer"),
        ("app.graphql.resolvers.semi_supervised_resolvers", None),
        ("sklearn.semi_supervised", "LabelPropagation"),
        ("pandas", None),
        ("numpy", None),
        ("motor", None),
        ("asyncpg", None)
    ]
    
    all_imports_ok = True
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            if class_name:
                getattr(module, class_name)
            print(f"✅ {module_name} importado correctamente")
        except ImportError as e:
            print(f"❌ Error importando {module_name}: {e}")
            all_imports_ok = False
        except Exception as e:
            print(f"❌ Error en {module_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

async def test_graphql_schema():
    """Verifica que el schema de GraphQL se pueda cargar"""
    print("\n🔗 Verificando schema de GraphQL...")
    
    try:
        from app.graphql.schema import schema
        print("✅ Schema de GraphQL cargado correctamente")
        
        # Verificar que las consultas semi-supervisadas estén disponibles
        query_type = schema.query_type
        mutation_type = schema.mutation_type
        
        # Obtener campos disponibles
        query_fields = list(query_type.fields.keys()) if query_type else []
        mutation_fields = list(mutation_type.fields.keys()) if mutation_type else []
        
        expected_queries = [
            "getSemiSupervisedDataSummary",
            "predictPostulacionEstado", 
            "getTrainedModelsInfo",
            "analyzeUnlabeledData"
        ]
        
        expected_mutations = [
            "trainSemiSupervisedModels"
        ]
        
        missing_queries = [q for q in expected_queries if q not in query_fields]
        missing_mutations = [m for m in expected_mutations if m not in mutation_fields]
        
        if missing_queries:
            print(f"⚠️  Consultas faltantes: {missing_queries}")
        else:
            print("✅ Todas las consultas semi-supervisadas disponibles")
            
        if missing_mutations:
            print(f"⚠️  Mutaciones faltantes: {missing_mutations}")
        else:
            print("✅ Todas las mutaciones semi-supervisadas disponibles")
        
        return len(missing_queries) == 0 and len(missing_mutations) == 0
        
    except Exception as e:
        print(f"❌ Error cargando schema de GraphQL: {e}")
        return False

def test_requirements():
    """Verifica que las dependencias estén instaladas"""
    print("\n📋 Verificando dependencias...")
    
    try:
        import pkg_resources
        
        with open("requirements.txt", "r") as f:
            requirements = f.read().splitlines()
        
        missing_packages = []
        for requirement in requirements:
            if requirement.strip() and not requirement.startswith("#"):
                package_name = requirement.split("==")[0].split(">=")[0].split("<=")[0]
                try:
                    pkg_resources.get_distribution(package_name)
                    print(f"✅ {package_name} instalado")
                except pkg_resources.DistributionNotFound:
                    print(f"❌ {package_name} no instalado")
                    missing_packages.append(package_name)
        
        if missing_packages:
            print(f"\n⚠️  Paquetes faltantes: {missing_packages}")
            print("💡 Ejecuta 'pip install -r requirements.txt' para instalar dependencias")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando dependencias: {e}")
        return False

async def run_quick_prediction_test():
    """Ejecuta una prueba rápida de predicción si los modelos están disponibles"""
    print("\n🔮 Probando predicción rápida...")
    
    model_dir = Path("trained_models/semi_supervised")
    if not (model_dir / "label_spreading_model.pkl").exists():
        print("⚠️  Modelos no encontrados. Saltando prueba de predicción.")
        return True
    
    try:
        from app.ml.models.semi_supervised_model import SemiSupervisedPostulacionModel
        from app.ml.preprocessing.semi_supervised_preprocessor import SemiSupervisedPreprocessor
        
        # Cargar modelo y preprocesador
        model = SemiSupervisedPostulacionModel()
        model.load_model(str(model_dir / "label_spreading_model.pkl"))
        
        preprocessor = SemiSupervisedPreprocessor()
        preprocessor.load_preprocessor(str(model_dir / "preprocessor.pkl"))
        
        # Datos de prueba
        test_data = {
            'nombre': 'Test User',
            'anios_experiencia': 3,
            'nivel_educacion': 'Universitario',
            'habilidades': 'Python, SQL',
            'idiomas': 'Español, Inglés',
            'oferta_titulo': 'Developer',
            'oferta_salario': 8000.0,
            'empresa_rubro': 'Tecnología'
        }
        
        # Realizar predicción
        import pandas as pd
        df_test = pd.DataFrame([test_data])
        prediction, confidence = model.predict_single(df_test, preprocessor)
        
        print(f"✅ Predicción exitosa: {prediction} (confianza: {confidence:.2f})")
        return True
        
    except Exception as e:
        print(f"❌ Error en predicción de prueba: {e}")
        return False

async def main():
    """Función principal que ejecuta todas las verificaciones"""
    print("🚀 VERIFICACIÓN RÁPIDA DEL MODELO SEMI-SUPERVISADO")
    print("=" * 60)
    
    tests = [
        ("Dependencias", test_requirements),
        ("Importaciones", test_imports),
        ("Conexiones BD", test_database_connections),
        ("Extracción de datos", test_data_extraction),
        ("Archivos de modelo", test_model_files),
        ("Schema GraphQL", test_graphql_schema),
        ("Predicción de prueba", run_quick_prediction_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Ejecutando: {test_name}")
        print("-" * 40)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ EXITOSO" if result else "❌ FALLIDO"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("\n🎉 ¡Todas las verificaciones fueron exitosas!")
        print("💡 El sistema está listo para usar. Consulta GUIA_SEMI_SUPERVISADO.md para instrucciones completas.")
    else:
        print(f"\n⚠️  {total - passed} verificaciones fallaron.")
        print("💡 Revisa los errores arriba y consulta GUIA_SEMI_SUPERVISADO.md para solución de problemas.")
        
        # Sugerencias específicas
        failed_tests = [name for name, result in results if not result]
        
        if "Dependencias" in failed_tests:
            print("\n🔧 Para instalar dependencias:")
            print("   pip install -r requirements.txt")
            
        if "Archivos de modelo" in failed_tests:
            print("\n🤖 Para entrenar modelos:")
            print("   python train_semi_supervised_step_by_step.py")
            
        if "Conexiones BD" in failed_tests:
            print("\n🗄️  Para configurar bases de datos:")
            print("   - Verifica que PostgreSQL y MongoDB estén ejecutándose")
            print("   - Revisa las variables de entorno en .env")

if __name__ == "__main__":
    asyncio.run(main())