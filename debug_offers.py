"""
Script para diagnosticar problema con job_offers_features
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config.mongodb_connection import get_collection_sync
from app.graphql.resolvers.feature_resolvers import get_job_offers_features
from app.graphql.types.feature_types import FeatureQueryInput

async def debug_offers():
    """Diagnosticar problema con ofertas"""
    
    print("=== DIAGNÓSTICO DE OFERTAS ===")
    
    # 1. Verificar conexión directa a MongoDB
    print("\n1. VERIFICANDO MONGODB DIRECTO:")
    try:
        collection = get_collection_sync("job_offers_features")
        total_docs = collection.count_documents({})
        print(f"   ✅ Total documentos en MongoDB: {total_docs}")
        
        # Obtener primer documento
        first_doc = collection.find_one({})
        if first_doc:
            print(f"   ✅ Primer documento encontrado:")
            print(f"       _id: {first_doc.get('_id')}")
            print(f"       titulo: {first_doc.get('titulo')}")
            print(f"       salario: {first_doc.get('salario')}")
            print(f"       ubicacion: {first_doc.get('ubicacion')}")
        else:
            print("   ❌ No se encontró ningún documento")
    except Exception as e:
        print(f"   ❌ Error en MongoDB: {e}")
    
    # 2. Verificar resolver GraphQL sin filtros
    print("\n2. VERIFICANDO RESOLVER SIN FILTROS:")
    try:
        query_input = FeatureQueryInput(limit=5, skip=0, search=None)
        result = await get_job_offers_features(query_input)
        print(f"   ✅ Total desde resolver: {result.total}")
        print(f"   ✅ Items retornados: {len(result.items)}")
        
        if result.items:
            for i, item in enumerate(result.items, 1):
                print(f"   📋 {i}. {item.titulo} - ${item.salario}")
        else:
            print("   ❌ No hay items en el resultado")
    except Exception as e:
        print(f"   ❌ Error en resolver: {e}")
    
    # 3. Verificar resolver GraphQL con filtros de búsqueda
    print("\n3. VERIFICANDO RESOLVER CON BÚSQUEDA 'Python':")
    try:
        query_input = FeatureQueryInput(limit=5, skip=0, search="Python")
        result = await get_job_offers_features(query_input)
        print(f"   ✅ Total desde resolver: {result.total}")
        print(f"   ✅ Items retornados: {len(result.items)}")
        
        if result.items:
            for i, item in enumerate(result.items, 1):
                print(f"   📋 {i}. {item.titulo} - ${item.salario}")
    except Exception as e:
        print(f"   ❌ Error en resolver con búsqueda: {e}")
    
    # 4. Verificar con query vacío (None)
    print("\n4. VERIFICANDO RESOLVER CON QUERY NONE:")
    try:
        result = await get_job_offers_features(None)
        print(f"   ✅ Total desde resolver: {result.total}")
        print(f"   ✅ Items retornados: {len(result.items)}")
        
        if result.items:
            for i, item in enumerate(result.items, 1):
                print(f"   📋 {i}. {item.titulo} - ${item.salario}")
    except Exception as e:
        print(f"   ❌ Error en resolver con None: {e}")
    
    # 5. Verificar consulta MongoDB directa con filtros
    print("\n5. VERIFICANDO MONGODB CON FILTROS:")
    try:
        collection = get_collection_sync("job_offers_features")
        
        # Sin filtros
        docs_sin_filtro = list(collection.find({}).limit(3))
        print(f"   ✅ Documentos sin filtro: {len(docs_sin_filtro)}")
        
        # Con filtro Python
        filter_python = {
            "$or": [
                {"titulo": {"$regex": "Python", "$options": "i"}},
                {"ubicacion": {"$regex": "Python", "$options": "i"}},
                {"requisitos": {"$regex": "Python", "$options": "i"}}
            ]
        }
        docs_python = list(collection.find(filter_python).limit(3))
        print(f"   ✅ Documentos con Python: {len(docs_python)}")
        
    except Exception as e:
        print(f"   ❌ Error en MongoDB con filtros: {e}")
    
    print("\n=== FIN DIAGNÓSTICO ===")

if __name__ == "__main__":
    asyncio.run(debug_offers())
