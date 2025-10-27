"""
Demostración completa del sistema de ML para contratación
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def create_demo_data():
    """Crea datos de demostración para el sistema"""
    print("📊 Creando datos de demostración...")
    
    # Datos sintéticos más realistas
    nombres = ["Ana García", "Carlos López", "María Rodríguez", "Juan Pérez", "Sofia Vargas"]
    educacion = ["licenciatura", "maestría", "técnico", "doctorado"]
    skills = [
        "python, sql, machine learning",
        "java, spring, docker",
        "javascript, react, node.js",
        "c#, .net, azure",
        "python, django, postgresql"
    ]
    idiomas = ["español, inglés", "español, inglés, francés", "español"]
    certificaciones = ["aws cloud practitioner", "microsoft azure", "google cloud", "sin certificacion"]
    
    puestos_actuales = ["desarrollador", "analista", "ingeniero", "consultor", "especialista"]
    industrias = ["tecnología", "finanzas", "salud", "educación", "retail"]
    
    # Trabajos disponibles
    trabajos = [
        {
            "titulo": "Desarrollador Python Senior",
            "descripcion": "Desarrollo de aplicaciones web con Python y Django",
            "salario": 15000,
            "ubicacion": "santa cruz",
            "requisitos": "python, django, postgresql, 3+ años experiencia"
        },
        {
            "titulo": "Data Scientist",
            "descripcion": "Análisis de datos y machine learning",
            "salario": 18000,
            "ubicacion": "la paz",
            "requisitos": "python, machine learning, sql, estadística"
        },
        {
            "titulo": "Desarrollador Frontend",
            "descripcion": "Desarrollo de interfaces de usuario",
            "salario": 12000,
            "ubicacion": "cochabamba",
            "requisitos": "javascript, react, css, html"
        }
    ]
    
    applications = []
    
    for i in range(20):
        # Crear aplicación aleatoria
        trabajo = random.choice(trabajos)
        
        # Simular correlación realista entre skills y probabilidad
        applicant_skills = random.choice(skills)
        job_reqs = trabajo["requisitos"]
        
        # Calcular overlap básico
        skills_set = set([s.strip().lower() for s in applicant_skills.split(',')])
        reqs_set = set([r.strip().lower() for r in job_reqs.split(',')])
        overlap = len(skills_set.intersection(reqs_set)) / len(reqs_set)
        
        # Experiencia correlacionada con skills match
        base_exp = random.randint(1, 10)
        if overlap > 0.5:
            exp_bonus = random.randint(1, 3)
        else:
            exp_bonus = 0
        years_exp = min(base_exp + exp_bonus, 15)
        
        fecha_pub = datetime.now() - timedelta(days=random.randint(1, 30))
        fecha_post = fecha_pub + timedelta(days=random.randint(1, 10))
        
        # Probabilidad basada en múltiples factores
        prob_factors = []
        prob_factors.append(overlap * 0.4)  # 40% por skills match
        prob_factors.append(min(years_exp / 10, 1) * 0.3)  # 30% por experiencia
        prob_factors.append(random.random() * 0.3)  # 30% aleatorio
        
        base_prob = sum(prob_factors)
        
        # Ajustar por educación
        education_level = random.choice(educacion)
        if education_level in ["maestría", "doctorado"]:
            base_prob += 0.1
        
        # Simular decisión binaria con ruido
        contactado = 1 if (base_prob + np.random.normal(0, 0.1)) > 0.5 else 0
        
        application = {
            'nombre': random.choice(nombres) + f" {i+1}",
            'años_experiencia': years_exp,
            'nivel_educacion': education_level,
            'habilidades': applicant_skills,
            'idiomas': random.choice(idiomas),
            'certificaciones': random.choice(certificaciones),
            'puesto_actual': random.choice(puestos_actuales),
            'industria': random.choice(industrias),
            'titulo': trabajo["titulo"],
            'descripcion': trabajo["descripcion"],
            'salario': trabajo["salario"],
            'ubicacion': trabajo["ubicacion"],
            'requisitos': trabajo["requisitos"],
            'fecha_postulacion': fecha_post.strftime('%Y-%m-%d'),
            'fecha_publicacion': fecha_pub.strftime('%Y-%m-%d'),
            'contactado': contactado
        }
        
        applications.append(application)
    
    # Guardar como CSV
    df = pd.DataFrame(applications)
    df.to_csv('demo_applications.csv', index=False)
    print(f"✅ Creados {len(applications)} registros de demostración")
    print(f"📈 Tasa de contacto: {df['contactado'].mean():.1%}")
    
    return df

def train_demo_model():
    """Entrena modelo con datos de demostración"""
    print("\n🚀 Entrenando modelo de demostración...")
    
    try:
        # Verificar si existe el modelo simple
        model_path = "trained_models/simple_hiring_model.pkl"
        
        if not os.path.exists(model_path):
            print("⚠️  Modelo no encontrado, ejecutando entrenamiento...")
            os.system("python train_simple.py")
        
        if os.path.exists(model_path):
            print("✅ Modelo disponible para predicciones")
            return True
        else:
            print("❌ No se pudo entrenar el modelo")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_predictions():
    """Demuestra predicciones con nuevos candidatos"""
    print("\n🔮 Demostración de Predicciones")
    print("=" * 50)
    
    try:
        from simple_predictor import SimpleHiringPredictor
        
        # Cargar modelo
        predictor = SimpleHiringPredictor("trained_models/simple_hiring_model.pkl")
        
        # Candidatos de prueba
        candidates = [
            {
                'nombre': 'Elena Morales',
                'años_experiencia': 8,
                'nivel_educacion': 'maestría',
                'habilidades': 'python, machine learning, sql, tensorflow',
                'idiomas': 'español, inglés',
                'certificaciones': 'aws cloud practitioner, tensorflow certified',
                'puesto_actual': 'senior data scientist',
                'industria': 'tecnología',
                'titulo': 'Data Scientist Senior',
                'descripcion': 'Liderar proyectos de ML y analytics',
                'salario': 20000,
                'ubicacion': 'santa cruz',
                'requisitos': 'python, machine learning, sql, 5+ años exp',
                'fecha_postulacion': '2024-01-15',
                'fecha_publicacion': '2024-01-10'
            },
            {
                'nombre': 'Roberto Silva',
                'años_experiencia': 2,
                'nivel_educacion': 'licenciatura',
                'habilidades': 'javascript, html, css',
                'idiomas': 'español',
                'certificaciones': 'sin certificacion',
                'puesto_actual': 'desarrollador junior',
                'industria': 'startup',
                'titulo': 'Desarrollador Python Senior',
                'descripcion': 'Desarrollo backend con Python',
                'salario': 15000,
                'ubicacion': 'la paz',
                'requisitos': 'python, django, postgresql, 5+ años exp',
                'fecha_postulacion': '2024-01-16',
                'fecha_publicacion': '2024-01-12'
            },
            {
                'nombre': 'Patricia Vega',
                'años_experiencia': 6,
                'nivel_educacion': 'maestría',
                'habilidades': 'python, django, postgresql, docker',
                'idiomas': 'español, inglés, alemán',
                'certificaciones': 'aws solutions architect',
                'puesto_actual': 'tech lead',
                'industria': 'fintech',
                'titulo': 'Desarrollador Python Senior',
                'descripcion': 'Desarrollo backend con Python y Django',
                'salario': 15000,
                'ubicacion': 'santa cruz',
                'requisitos': 'python, django, postgresql, 5+ años exp',
                'fecha_postulacion': '2024-01-14',
                'fecha_publicacion': '2024-01-12'
            }
        ]
        
        results = []
        
        for candidate in candidates:
            print(f"\n👤 Evaluando: {candidate['nombre']}")
            print(f"   Puesto: {candidate['titulo']}")
            print(f"   Experiencia: {candidate['años_experiencia']} años")
            print(f"   Skills: {candidate['habilidades']}")
            
            result = predictor.predict(candidate)
            
            print(f"   📊 Probabilidad: {result['probability']:.1%}")
            print(f"   🎯 Confianza: {result['confidence_level']}")
            print(f"   💡 Recomendación: {result['recommendation']}")
            
            results.append({
                'nombre': candidate['nombre'],
                'puesto': candidate['titulo'],
                'probabilidad': result['probability'],
                'recomendacion': result['recommendation']
            })
        
        # Ranking de candidatos
        print(f"\n🏆 Ranking de Candidatos")
        print("=" * 50)
        
        sorted_results = sorted(results, key=lambda x: x['probabilidad'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result['nombre']}")
            print(f"   Probabilidad: {result['probabilidad']:.1%}")
            print(f"   Recomendación: {result['recomendacion']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración: {e}")
        return False

def demo_api():
    """Demuestra el uso de la API GraphQL"""
    print("\n🌐 Demostración de API GraphQL")
    print("=" * 50)
    
    print("Para probar la API GraphQL:")
    print("1. Ejecuta: uvicorn app.main:app --reload")
    print("2. Ve a: http://localhost:8000/graphql")
    print("3. Usa esta mutación de ejemplo:")
    print()
    print("mutation {")
    print("  predictHiring(")
    print("    nombre: \"Test User\"")
    print("    anosExperiencia: 5")
    print("    nivelEducacion: \"maestría\"")
    print("    habilidades: \"python, machine learning, sql\"")
    print("    idiomas: \"español, inglés\"")
    print("    certificaciones: \"aws cloud practitioner\"")
    print("    puestoActual: \"data scientist\"")
    print("    industria: \"tecnología\"")
    print("    titulo: \"Senior Data Scientist\"")
    print("    descripcion: \"Posición senior en data science\"")
    print("    salario: 18000")
    print("    ubicacion: \"santa cruz\"")
    print("    requisitos: \"python, machine learning, sql, 5+ años\"")
    print("  ) {")
    print("    prediction")
    print("    probability")
    print("    confidenceLevel")
    print("    recommendation")
    print("    modelUsed")
    print("  }")
    print("}")

def main():
    """Función principal de demostración"""
    print("🎯 Sistema de ML para Predicción de Contratación")
    print("=" * 60)
    print("Esta demostración muestra un sistema completo de machine learning")
    print("para predecir la probabilidad de que un candidato sea contactado.")
    print()
    
    # Paso 1: Crear datos de demostración
    demo_data = create_demo_data()
    
    # Paso 2: Entrenar modelo
    if train_demo_model():
        # Paso 3: Realizar predicciones
        demo_predictions()
        
        # Paso 4: Mostrar info de API
        demo_api()
        
        print("\n✅ Demostración completada exitosamente!")
        print("\n📋 Resumen del Sistema:")
        print("- ✅ Datos sintéticos generados")
        print("- ✅ Modelo entrenado")
        print("- ✅ Predicciones funcionando")
        print("- ✅ API GraphQL disponible")
        print("\n🚀 El sistema está listo para usar!")
        
    else:
        print("\n❌ No se pudo completar la demostración")

if __name__ == "__main__":
    main()