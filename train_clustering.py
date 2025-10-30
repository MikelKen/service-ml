"""
Script para entrenar el modelo de clustering de candidatos
"""
import sys
import os
import time

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.clustering_service import clustering_service


def train_clustering_model():
    """Entrena el modelo de clustering"""
    print("ENTRENAMIENTO DE MODELO DE CLUSTERING")
    print("=" * 50)
    print()
    
    try:
        # 1. Verificar dataset
        print("1. Verificando dataset...")
        df = clustering_service.load_data()
        print(f"   ✓ Dataset cargado: {len(df)} candidatos")
        print(f"   ✓ Columnas disponibles: {len(df.columns)}")
        print()
        
        # 2. Entrenar con K-Means
        print("2. Entrenando con K-Means (5 clusters)...")
        start_time = time.time()
        
        result_kmeans = clustering_service.train_clustering_model(
            algorithm="kmeans",
            n_clusters=5
        )
        
        training_time = time.time() - start_time
        print(f"   ✓ K-Means completado en {training_time:.2f}s")
        print(f"   ✓ Clusters: {result_kmeans['num_clusters']}")
        print(f"   ✓ Silhouette Score: {result_kmeans['silhouette_score']:.3f}")
        print()
        
        # 3. Entrenar con Clustering Jerárquico
        print("3. Entrenando con Clustering Jerárquico...")
        start_time = time.time()
        
        result_hierarchical = clustering_service.train_clustering_model(
            algorithm="hierarchical",
            n_clusters=4,
            linkage="ward"
        )
        
        training_time = time.time() - start_time
        print(f"   ✓ Clustering Jerárquico completado en {training_time:.2f}s")
        print(f"   ✓ Clusters: {result_hierarchical['num_clusters']}")
        print(f"   ✓ Silhouette Score: {result_hierarchical['silhouette_score']:.3f}")
        print()
        
        # 4. Entrenar con DBSCAN
        print("4. Entrenando con DBSCAN...")
        start_time = time.time()
        
        result_dbscan = clustering_service.train_clustering_model(
            algorithm="dbscan",
            eps=1.2,
            min_samples=3
        )
        
        training_time = time.time() - start_time
        print(f"   ✓ DBSCAN completado en {training_time:.2f}s")
        print(f"   ✓ Clusters: {result_dbscan['num_clusters']}")
        print(f"   ✓ Silhouette Score: {result_dbscan['silhouette_score']:.3f}")
        print()
        
        # 5. Comparar resultados
        print("5. Comparación de algoritmos:")
        print()
        algorithms = [
            ("K-Means", result_kmeans),
            ("Jerárquico", result_hierarchical), 
            ("DBSCAN", result_dbscan)
        ]
        
        best_algorithm = None
        best_score = -1
        
        for name, result in algorithms:
            score = result['silhouette_score']
            clusters = result['num_clusters']
            print(f"   {name:12} | Clusters: {clusters:2d} | Silhouette: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_algorithm = name
        
        print()
        print(f"   🏆 Mejor algoritmo: {best_algorithm} (Score: {best_score:.3f})")
        print()
        
        # 6. Guardar el mejor modelo
        print("6. Guardando mejor modelo...")
        # Volver a entrenar con el mejor algoritmo para guardar
        if best_algorithm == "K-Means":
            final_result = clustering_service.train_clustering_model(algorithm="kmeans", n_clusters=5)
        elif best_algorithm == "Jerárquico":
            final_result = clustering_service.train_clustering_model(algorithm="hierarchical", n_clusters=4)
        else:
            final_result = clustering_service.train_clustering_model(algorithm="dbscan", eps=1.2, min_samples=3)
        
        print(f"   ✓ Modelo final guardado")
        print(f"   ✓ Archivo: trained_models/candidate_clustering_model.pkl")
        print()
        
        # 7. Mostrar resumen de clusters
        print("7. Resumen de clusters del mejor modelo:")
        print()
        
        for cluster in final_result['clusters']:
            print(f"   📊 {cluster['cluster_name']} ({cluster['candidate_count']} candidatos)")
            print(f"      • Experiencia promedio: {cluster['avg_experience_years']:.1f} años")
            print(f"      • Salario promedio: ${cluster['avg_salary_expectation']:,.0f}")
            print(f"      • Habilidades: {', '.join(cluster['common_skills'][:3])}")
            print()
        
        print("=" * 50)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        return False


def main():
    """Función principal"""
    print("Script de Entrenamiento - Clustering de Candidatos")
    print()
    
    # Crear directorio de modelos si no existe
    os.makedirs("trained_models", exist_ok=True)
    
    success = train_clustering_model()
    
    if success:
        print()
        print("El modelo está listo para usar en el microservicio!")
        print("Puede iniciar el servidor con: uvicorn app.main:app --reload")
    else:
        print()
        print("Hubo errores durante el entrenamiento. Revise los logs.")
    
    return success


if __name__ == "__main__":
    main()