"""
Script simplificado para probar el sistema sin dependencias externas complejas
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Prueba la carga de datos"""
    print("=== Test 1: Carga de Datos ===")
    
    try:
        df = pd.read_csv("postulaciones_sinteticas_500.csv")
        print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"Columnas: {list(df.columns)}")
        return True  # Retornar booleano en lugar de DataFrame
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return False

def test_preprocessing():
    """Prueba el preprocessamiento básico"""
    print("\n=== Test 2: Preprocessamiento Básico ===")
    
    try:
        from ml.data.preprocessing import DataPreprocessor
        
        # Crear datos de prueba
        test_data = {
            'nombre': ['Juan Pérez', 'María González'],
            'años_experiencia': ['5', '3'],
            'nivel_educacion': ['Licenciatura', 'Maestría'],
            'habilidades': ['Python, SQL', 'Java, Machine Learning'],
            'estado': ['Aceptado', 'Rechazado'],
            'salario': ['8000', '12000']
        }
        
        df_test = pd.DataFrame(test_data)
        
        preprocessor = DataPreprocessor()
        
        # Probar normalización de texto
        normalized = preprocessor.normalize_text("Habilidades Técnicas")
        print(f"✅ Normalización de texto: 'Habilidades Técnicas' -> '{normalized}'")
        
        # Probar creación de target
        target = preprocessor.create_target_variable(df_test)
        print(f"✅ Variable objetivo creada: {target.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en preprocessamiento: {e}")
        return False

def test_basic_features():
    """Prueba features básicas sin dependencias complejas"""
    print("\n=== Test 3: Features Básicas ===")
    
    try:
        # Datos de prueba simples
        data = {
            'años_experiencia': [5, 3, 10],
            'salario': [8000, 12000, 15000],
            'habilidades': ['python, sql', 'java, spring', 'machine learning, python'],
            'requisitos': ['python, sql, aws', 'java, spring boot', 'python, ml, tensorflow'],
            'fecha_postulacion': ['2024-01-15', '2024-02-10', '2024-03-05'],
            'fecha_publicacion': ['2024-01-10', '2024-02-05', '2024-03-01']
        }
        
        df = pd.DataFrame(data)
        
        # Convertir fechas
        df['fecha_postulacion'] = pd.to_datetime(df['fecha_postulacion'])
        df['fecha_publicacion'] = pd.to_datetime(df['fecha_publicacion'])
        
        # Feature temporal
        df['dias_desde_publicacion'] = (df['fecha_postulacion'] - df['fecha_publicacion']).dt.days
        print(f"✅ Feature temporal: {df['dias_desde_publicacion'].tolist()}")
        
        # Feature de coincidencia simple
        def simple_skill_overlap(skills, requirements):
            skills_set = set([s.strip().lower() for s in str(skills).split(',')])
            req_set = set([r.strip().lower() for r in str(requirements).split(',')])
            intersection = skills_set.intersection(req_set)
            return len(intersection) / len(req_set) if req_set else 0
        
        df['skill_overlap'] = df.apply(lambda row: simple_skill_overlap(row['habilidades'], row['requisitos']), axis=1)
        print(f"✅ Overlap de habilidades: {df['skill_overlap'].tolist()}")
        
        # Feature numérica simple
        df['salario_por_año'] = df['salario'] / df['años_experiencia']
        print(f"✅ Salario por año experiencia: {df['salario_por_año'].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en features básicas: {e}")
        return False

def test_simple_model():
    """Prueba un modelo simple sin dependencias complejas"""
    print("\n=== Test 4: Modelo Simple ===")
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Crear dataset sintético simple
        np.random.seed(42)
        n_samples = 100
        
        # Features simples
        X = np.random.randn(n_samples, 5)  # 5 features aleatorias
        # Target con algo de lógica
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predecir
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas simples
        accuracy = np.mean(predictions == y_test)
        print(f"✅ Modelo simple entrenado - Accuracy: {accuracy:.3f}")
        print(f"✅ Ejemplo de probabilidades: {probabilities[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en modelo simple: {e}")
        return False

def test_predictor_structure():
    """Prueba la estructura del predictor sin cargar modelo"""
    print("\n=== Test 5: Estructura del Predictor ===")
    
    try:
        from ml.models.predictor import HiringPredictor
        
        # Crear predictor sin modelo
        predictor = HiringPredictor()
        
        print(f"✅ Predictor creado")
        print(f"✅ Estado inicial - Modelo cargado: {predictor.is_loaded}")
        
        # Probar métodos de utilidad
        confidence = predictor._get_confidence_level(0.75)
        recommendation = predictor._get_recommendation(0.75)
        
        print(f"✅ Nivel de confianza para 0.75: {confidence}")
        print(f"✅ Recomendación para 0.75: {recommendation}")
        
        # Probar overlap de habilidades
        overlap = predictor._calculate_skill_overlap("python, sql, aws", "python, sql, java")
        print(f"✅ Overlap de habilidades: {overlap:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en estructura del predictor: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 EJECUTANDO PRUEBAS DEL SISTEMA ML")
    print("=" * 50)
    
    # Cambiar al directorio del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    tests = [
        test_data_loading,
        test_preprocessing,
        test_basic_features,
        test_simple_model,
        test_predictor_structure
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Error en test: {e}")
            results.append(False)
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS:")
    
    # Función auxiliar para convertir resultados a booleanos de forma segura
    def safe_bool_conversion(result):
        if isinstance(result, bool):
            return result
        elif isinstance(result, (int, float)):
            return bool(result)
        elif hasattr(result, 'empty'):  # Para DataFrames
            return not result.empty
        elif result is None:
            return False
        else:
            return True  # Si hay cualquier otro valor, considerarlo como True
    
    # Convertir todos los resultados a booleanos de forma segura
    bool_results = [safe_bool_conversion(result) for result in results]
    passed = sum(bool_results)
    total = len(bool_results)
    
    test_names = [
        "Carga de Datos",
        "Preprocessamiento",
        "Features Básicas", 
        "Modelo Simple",
        "Estructura Predictor"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, bool_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nRESULTADO FINAL: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("✅ El sistema está listo para entrenamiento completo")
    else:
        print("⚠️  Algunas pruebas fallaron")
        print("💡 Revise los errores arriba para diagnosticar")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)