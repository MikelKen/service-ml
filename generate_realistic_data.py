"""
Genera datos sintéticos más realistas para entrenamiento del modelo ML
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_realistic_data():
    """Genera datos más realistas y lógicos para el entrenamiento"""
    print("📊 Generando datos realistas para entrenamiento...")
    
    # Listas de datos más realistas
    nombres = [
        "Ana García", "Carlos López", "María Rodríguez", "Juan Pérez", 
        "Sofia Vargas", "Diego Morales", "Lucia Santos", "Miguel Torres",
        "Carmen Ruiz", "Alejandro Silva", "Valentina Cruz", "Gabriel Ramos"
    ]
    
    educacion_levels = {
        "técnico": 1,
        "licenciatura": 2, 
        "maestría": 3,
        "doctorado": 4
    }
    
    # Skills por categorías más realistas
    skill_sets = {
        "data_science": ["python", "machine learning", "sql", "pandas", "numpy", "tensorflow", "scikit-learn"],
        "backend": ["python", "django", "flask", "postgresql", "mysql", "redis", "celery"],
        "frontend": ["javascript", "react", "vue", "angular", "css", "html", "typescript"],
        "devops": ["docker", "kubernetes", "aws", "azure", "jenkins", "terraform", "linux"],
        "mobile": ["react native", "flutter", "swift", "kotlin", "ios", "android"],
        "fullstack": ["python", "javascript", "react", "django", "postgresql", "docker"]
    }
    
    # Trabajos más específicos
    job_categories = {
        "data_scientist": {
            "titulo": "Data Scientist",
            "descripcion": "Análisis de datos y machine learning",
            "salario_base": 18000,
            "requisitos": ["python", "machine learning", "sql", "estadística"],
            "skills_relevantes": skill_sets["data_science"]
        },
        "backend_dev": {
            "titulo": "Desarrollador Backend",
            "descripcion": "Desarrollo de APIs y servicios backend",
            "salario_base": 15000,
            "requisitos": ["python", "django", "postgresql", "apis"],
            "skills_relevantes": skill_sets["backend"]
        },
        "frontend_dev": {
            "titulo": "Desarrollador Frontend",
            "descripcion": "Desarrollo de interfaces de usuario",
            "salario_base": 12000,
            "requisitos": ["javascript", "react", "css", "html"],
            "skills_relevantes": skill_sets["frontend"]
        },
        "fullstack_dev": {
            "titulo": "Desarrollador Full Stack",
            "descripcion": "Desarrollo completo frontend y backend",
            "salario_base": 16000,
            "requisitos": ["python", "javascript", "react", "django"],
            "skills_relevantes": skill_sets["fullstack"]
        }
    }
    
    certificaciones = [
        "aws cloud practitioner", "aws solutions architect", 
        "google cloud professional", "microsoft azure fundamentals",
        "scrum master", "pmp", "sin certificacion"
    ]
    
    applications = []
    
    # Generar 300 registros más realistas
    for i in range(300):
        # Seleccionar categoria de trabajo
        job_cat = random.choice(list(job_categories.keys()))
        job_info = job_categories[job_cat]
        
        # Experiencia realista (0-20 años, con distribución más realista)
        exp_weights = [20, 25, 20, 15, 10, 5, 3, 2]  # Más gente con 0-7 años
        # Crear distribución completa normalizada
        all_weights = exp_weights + [1]*13  # 1 para años 8-20
        all_weights = [w/sum(all_weights) for w in all_weights]  # Normalizar
        years_exp = np.random.choice(range(0, 21), p=all_weights)
        
        # Educación correlacionada con experiencia
        if years_exp <= 2:
            education_options = ["técnico", "licenciatura"]
            education_weights = [0.3, 0.7]
        elif years_exp <= 5:
            education_options = ["técnico", "licenciatura", "maestría"] 
            education_weights = [0.2, 0.6, 0.2]
        elif years_exp <= 10:
            education_options = ["licenciatura", "maestría", "doctorado"]
            education_weights = [0.4, 0.5, 0.1]
        else:
            education_options = ["licenciatura", "maestría", "doctorado"]
            education_weights = [0.3, 0.5, 0.2]
        
        education = np.random.choice(education_options, p=education_weights)
        
        # Skills más realistas basadas en la categoría del trabajo
        base_skills = job_info["skills_relevantes"]
        num_skills = min(random.randint(2, 6), len(base_skills))
        candidate_skills = random.sample(base_skills, num_skills)
        
        # Añadir algunas skills adicionales aleatoriamente
        other_skills = ["git", "linux", "api", "testing", "agile"]
        if random.random() < 0.3:
            candidate_skills.extend(random.sample(other_skills, random.randint(1, 2)))
        
        skills_str = ", ".join(candidate_skills)
        
        # Calcular skill match más preciso
        job_requirements = job_info["requisitos"]
        skill_overlap = len(set(candidate_skills) & set(job_requirements)) / len(job_requirements)
        
        # Salario más realista basado en experiencia y educación
        salario_base = job_info["salario_base"]
        salario_adj = salario_base + (years_exp * 800) + (educacion_levels[education] * 1000)
        salario_variation = random.uniform(0.8, 1.2)
        salario = int(salario_adj * salario_variation)
        
        # Fechas más recientes
        fecha_pub = datetime.now() - timedelta(days=random.randint(1, 60))
        fecha_post = fecha_pub + timedelta(days=random.randint(1, 14))
        
        # Lógica de contratación más realista
        hiring_score = 0
        
        # Factor experiencia (óptimo entre 3-12 años)
        if 3 <= years_exp <= 7:
            hiring_score += 0.3
        elif 8 <= years_exp <= 12:
            hiring_score += 0.25
        elif 1 <= years_exp <= 2:
            hiring_score += 0.15
        elif years_exp == 0:
            hiring_score += 0.05
        elif years_exp > 15:  # Muy senior, costoso
            hiring_score += 0.1
        else:
            hiring_score += 0.2
            
        # Factor educación
        hiring_score += educacion_levels[education] * 0.1
        
        # Factor skill match (muy importante)
        hiring_score += skill_overlap * 0.4
        
        # Factor certificaciones
        cert = random.choice(certificaciones)
        if cert != "sin certificacion":
            hiring_score += 0.1
            
        # Factor salario (no muy alto ni muy bajo)
        if salario_base * 0.8 <= salario <= salario_base * 1.3:
            hiring_score += 0.1
        elif salario > salario_base * 1.5:
            hiring_score -= 0.1
            
        # Añadir algo de ruido pero manteniendo lógica
        hiring_score += np.random.normal(0, 0.15)
        
        # Decisión final más estricta
        contactado = 1 if hiring_score > 0.6 else 0
        
        application = {
            'nombre': f"{random.choice(nombres)} {i+1}",
            'años_experiencia': years_exp,
            'nivel_educacion': education,
            'habilidades': skills_str,
            'idiomas': random.choice(["español", "español, inglés", "español, inglés, francés"]),
            'certificaciones': cert,
            'puesto_actual': random.choice(["junior", "semi-senior", "senior", "lead", "manager"]),
            'industria': random.choice(["tecnología", "finanzas", "salud", "educación", "startup"]),
            'titulo': job_info["titulo"],
            'descripcion': job_info["descripcion"],
            'salario': salario,
            'ubicacion': random.choice(["santa cruz", "la paz", "cochabamba", "sucre", "tarija"]),
            'requisitos': ", ".join(job_requirements),
            'fecha_postulacion': fecha_post.strftime('%Y-%m-%d'),
            'fecha_publicacion': fecha_pub.strftime('%Y-%m-%d'),
            'contactado': contactado
        }
        
        applications.append(application)
    
    # Crear DataFrame y guardar
    df = pd.DataFrame(applications)
    df.to_csv('datos_entrenamiento_realista.csv', index=False)
    
    print(f"✅ Generados {len(applications)} registros realistas")
    print(f"📈 Tasa de contratación: {df['contactado'].mean():.1%}")
    print(f"📊 Distribución por experiencia:")
    print(df.groupby(pd.cut(df['años_experiencia'], bins=[0,2,5,10,15,25], labels=['0-2','3-5','6-10','11-15','16+']))['contactado'].agg(['count', 'mean']))
    
    return df

if __name__ == "__main__":
    generate_realistic_data()
