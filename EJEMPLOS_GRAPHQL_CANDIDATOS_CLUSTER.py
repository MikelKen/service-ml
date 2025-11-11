#!/usr/bin/env python3
"""
📚 EJEMPLOS PRÁCTICOS DE USO - getCandidatesInCluster

Ejemplos GraphQL listos para copiar y pegar en GraphiQL
"""

EJEMPLOS = {
    "1_basico_obtener_candidatos": """
    # 🔷 EJEMPLO 1: Obtener primeros 10 candidatos del cluster principal
    # Descripción: Obtiene los 10 primeros candidatos del cluster más grande (cluster 3)
    
    query ObtenerCandidatosClustersBasico {
      getCandidatesInCluster(input: {
        clusterId: 3
        algorithm: "kmeans"
        limit: 10
      }) {
        clusterId
        totalCandidates
        clusterPercentage
        candidates {
          candidateId
          name
          email
          yearsExperience
          workArea
        }
      }
    }
    """,
    
    "2_detalles_completos": """
    # 🔷 EJEMPLO 2: Obtener datos COMPLETOS de 20 candidatos
    # Descripción: Incluye skills, certificaciones, nivel de inglés y más
    
    query ObtenerDetallesCompletos {
      getCandidatesInCluster(input: {
        clusterId: 3
        algorithm: "kmeans"
        includeDetails: true
        limit: 20
      }) {
        clusterId
        totalCandidates
        clusterPercentage
        candidates {
          candidateId
          name
          email
          yearsExperience
          educationArea
          workArea
          skills
          certifications
          englishLevel
          distanceToCenter
        }
      }
    }
    """,
    
    "3_todos_sin_limite": """
    # 🔷 EJEMPLO 3: Obtener TODOS los candidatos de un cluster
    # Descripción: Sin límite - devuelve todos los candidatos
    # ⚠️  CUIDADO: Para clusters grandes puede ser lento
    
    query ObtenerTodosLosCandidatos {
      getCandidatesInCluster(input: {
        clusterId: 1
        algorithm: "kmeans"
        limit: null
      }) {
        clusterId
        totalCandidates
        clusterPercentage
        candidates {
          candidateId
          name
          skills
        }
      }
    }
    """,
    
    "4_cluster_especializado": """
    # 🔷 EJEMPLO 4: Explorar cluster especializado (pequeño)
    # Descripción: Ver candidatos de un cluster nicho (cluster 0)
    
    query ExplorarClusterEspecializado {
      getCandidatesInCluster(input: {
        clusterId: 0
        algorithm: "kmeans"
        limit: 50
      }) {
        clusterId
        totalCandidates
        clusterPercentage
        candidates {
          name
          educationArea
          certifications
          englishLevel
        }
      }
    }
    """,
    
    "5_analisis_skills": """
    # 🔷 EJEMPLO 5: Análisis de Skills en un cluster
    # Descripción: Obtener todas las habilidades técnicas de candidatos
    
    query AnalisisSkillsCluster {
      getCandidatesInCluster(input: {
        clusterId: 3
        algorithm: "kmeans"
        limit: 100
      }) {
        clusterId
        totalCandidates
        candidates {
          name
          skills
          yearsExperience
        }
      }
    }
    """,
    
    "6_distancia_centroide": """
    # 🔷 EJEMPLO 6: Candidatos más similares al centroide
    # Descripción: Ordenar por distancia (K-Means solo)
    # Los con menor distancia son más representativos del cluster
    
    query CandidatosMasRepresentativos {
      getCandidatesInCluster(input: {
        clusterId: 3
        algorithm: "kmeans"
        limit: 5
      }) {
        clusterId
        candidates {
          name
          distanceToCenter
          skills
        }
      }
    }
    """,
    
    "7_exportacion_datos": """
    # 🔷 EJEMPLO 7: Preparar datos para exportación
    # Descripción: Obtener información necesaria para CSV/Excel
    
    query ExportarCandidatos {
      getCandidatesInCluster(input: {
        clusterId: 3
        algorithm: "kmeans"
        limit: 1000
      }) {
        clusterId
        totalCandidates
        candidates {
          candidateId
          name
          email
          yearsExperience
          educationArea
          workArea
        }
      }
    }
    """,
    
    "8_comparar_clusters": """
    # 🔷 EJEMPLO 8: Comparar dos clusters diferentes
    # Descripción: Ejecutar esta query dos veces con clusterId diferente
    # Primera con clusterId: 0, luego con clusterId: 3
    
    query CompararClusterConPrincipal {
      getCandidatesInCluster(input: {
        clusterId: 0
        algorithm: "kmeans"
        limit: 20
      }) {
        clusterId
        totalCandidates
        clusterPercentage
        candidates {
          name
          yearsExperience
          educationArea
          certifications
        }
      }
    }
    """,
    
    "9_busqueda_especialista": """
    # 🔷 EJEMPLO 9: Encontrar especialistas en un cluster
    # Descripción: Candidatos con alta experiencia en área específica
    
    query BuscarEspecialistas {
      getCandidatesInCluster(input: {
        clusterId: 2
        algorithm: "kmeans"
        limit: 30
      }) {
        clusterId
        candidates {
          name
          yearsExperience
          workArea
          certifications
        }
      }
    }
    """,
    
    "10_nivel_idiomas": """
    # 🔷 EJEMPLO 10: Filtrar por nivel de inglés (análisis)
    # Descripción: Ver distribución de nivel de inglés en el cluster
    
    query AnalisisIdiomasCluster {
      getCandidatesInCluster(input: {
        clusterId: 3
        algorithm: "kmeans"
        limit: 50
      }) {
        clusterId
        candidates {
          name
          englishLevel
          yearsExperience
        }
      }
    }
    """,
    
    "11_multiples_queries": """
    # 🔷 EJEMPLO 11: Obtener datos de 2 clusters en una sola query
    # Descripción: Usar aliases para obtener múltiples clusters
    
    query ObtenerMultiplesClusters {
      clusterPrincipal: getCandidatesInCluster(input: {
        clusterId: 3
        algorithm: "kmeans"
        limit: 10
      }) {
        totalCandidates
        candidates {
          name
          skills
        }
      }
      
      clusterEspecializado: getCandidatesInCluster(input: {
        clusterId: 0
        algorithm: "kmeans"
        limit: 10
      }) {
        totalCandidates
        candidates {
          name
          certifications
        }
      }
    }
    """,
    
    "12_estadisticas": """
    # 🔷 EJEMPLO 12: Recopilar datos para estadísticas
    # Descripción: Obtener todos los datos necesarios para análisis
    
    query RecopiladorDatos {
      getCandidatesInCluster(input: {
        clusterId: 3
        algorithm: "kmeans"
        limit: null
      }) {
        clusterId
        totalCandidates
        clusterPercentage
        candidates {
          yearsExperience
          englishLevel
          certifications
        }
      }
    }
    """,
}

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║     📚 EJEMPLOS GRAPHQL - getCandidatesInCluster                          ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    for key, query in EJEMPLOS.items():
        print(query)
        print("\n" + "=" * 79 + "\n")
    
    print("""
    💡 INSTRUCCIONES DE USO:
    
    1. Copia cualquiera de los ejemplos anterior
    2. Abre http://localhost:8000/graphql en tu navegador
    3. Pega la query en el editor de la izquierda
    4. Presiona el botón ▶️ (Play) o Ctrl+Enter
    5. Verás el resultado en JSON en el panel derecho
    
    📊 NOTAS IMPORTANTES:
    
    • limit: null significa obtener TODOS los candidatos (cuidado con clusters grandes)
    • distanceToCenter solo funciona con algorithm: "kmeans"
    • Algunos campos pueden ser null si no están disponibles en la BD
    • La primera llamada tarda más mientras carga el modelo
    • Las llamadas subsecuentes son más rápidas (usa caché)
    
    🎯 PRÓXIMOS PASOS:
    
    1. Experimenta combinando diferentes clusterId (0, 1, 2, 3, etc.)
    2. Prueba con limit diferente (10, 50, 100, null)
    3. Cambia algorithm a "dbscan" para comparar
    4. Combina múltiples queries con aliases
    """)
