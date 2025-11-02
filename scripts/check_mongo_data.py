"""
Script para verificar datos en MongoDB
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.mongodb_connection import get_mongodb_sync

def check_mongodb_data():
    """Verifica datos en MongoDB"""
    
    print("=== VERIFICANDO DATOS EN MONGODB ===")
    
    try:
        # Conectar a MongoDB
        db = get_mongodb_sync()
        print("✓ Conexión establecida")
        
        # Listar colecciones
        collections = db.list_collection_names()
        print(f"✓ Colecciones encontradas: {collections}")
        
        # Verificar datos en cada colección
        for collection_name in ['candidates_features', 'companies_features', 'job_offers_features']:
            if collection_name in collections:
                count = db[collection_name].count_documents({})
                print(f"  - {collection_name}: {count} documentos")
                
                if count > 0:
                    # Mostrar un documento de ejemplo
                    sample = db[collection_name].find_one({})
                    print(f"    Campos del ejemplo: {list(sample.keys())}")
                    if collection_name == 'job_offers_features':
                        print(f"    Ejemplo: {sample}")
            else:
                print(f"  - {collection_name}: No existe")
        
        print("\n=== VERIFICACIÓN COMPLETADA ===")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = check_mongodb_data()
    sys.exit(0 if success else 1)