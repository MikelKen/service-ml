#!/usr/bin/env python3
"""
🔍 ANÁLISIS DE DATOS PARA CLUSTERING
Analiza la estructura de candidates_features para clustering
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import settings
import pandas as pd
import numpy as np

async def analyze_candidates_data():
    """Analiza los datos de candidatos en MongoDB"""
    print("🔍 INICIANDO ANÁLISIS DE DATOS PARA CLUSTERING")
    print("="*60)
    
    # Conectar a MongoDB
    client = AsyncIOMotorClient(settings.mongodb_url)
    db = client[settings.mongodb_database]
    collection = db["candidates_features"]
    
    try:
        # Obtener muestra de datos
        print("📊 Obteniendo muestra de datos...")
        sample_cursor = collection.find().limit(10)
        sample_docs = await sample_cursor.to_list(length=10)
        
        print(f"📈 Total de documentos encontrados: {len(sample_docs)}")
        
        if sample_docs:
            print("\n🔍 ESTRUCTURA DE DATOS:")
            first_doc = sample_docs[0]
            for key, value in first_doc.items():
                print(f"  • {key}: {type(value).__name__} - '{str(value)[:50]}{'...' if len(str(value)) > 50 else ''}'")
            
            print("\n📋 CAMPOS RELEVANTES PARA CLUSTERING:")
            clustering_fields = [
                'anios_experiencia',
                'nivel_educacion', 
                'habilidades',
                'idiomas',
                'certificaciones',
                'puesto_actual'
            ]
            
            for field in clustering_fields:
                if field in first_doc:
                    print(f"  ✅ {field}: {type(first_doc[field]).__name__}")
                else:
                    print(f"  ❌ {field}: NO ENCONTRADO")
            
            # Análisis de distribución
            print("\n📊 ANÁLISIS DE DISTRIBUCIÓN:")
            
            # Contar total de documentos
            total_count = await collection.count_documents({})
            print(f"  • Total candidatos: {total_count}")
            
            # Analizar años de experiencia
            exp_pipeline = [
                {"$group": {
                    "_id": "$anios_experiencia",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            exp_data = await collection.aggregate(exp_pipeline).to_list(length=None)
            print(f"  • Distribución experiencia: {len(exp_data)} niveles únicos")
            
            # Analizar niveles de educación
            edu_pipeline = [
                {"$group": {
                    "_id": "$nivel_educacion",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            edu_data = await collection.aggregate(edu_pipeline).to_list(length=None)
            print(f"  • Niveles educación únicos: {len(edu_data)}")
            for edu in edu_data[:5]:  # Top 5
                print(f"    - {edu['_id']}: {edu['count']} candidatos")
            
            print("\n🎯 CAMPOS IDENTIFICADOS PARA CLUSTERING:")
            print("  1. 📈 anios_experiencia (numérico)")
            print("  2. 🎓 nivel_educacion (categórico)")
            print("  3. 🛠️ habilidades (texto - TF-IDF)")
            print("  4. 🌍 idiomas (texto - análisis)")
            print("  5. 🏆 certificaciones (texto - presencia)")
            print("  6. 💼 puesto_actual (categórico)")
            
            return True
        else:
            print("❌ No se encontraron datos en la colección")
            return False
            
    except Exception as e:
        print(f"❌ Error al analizar datos: {e}")
        return False
    
    finally:
        client.close()

async def main():
    success = await analyze_candidates_data()
    if success:
        print("\n✅ ANÁLISIS COMPLETADO - Datos listos para clustering")
        print("🚀 Siguiente paso: Crear preprocessor para clustering")
    else:
        print("\n❌ ANÁLISIS FALLÓ - Revisar conexión a MongoDB")

if __name__ == "__main__":
    asyncio.run(main())