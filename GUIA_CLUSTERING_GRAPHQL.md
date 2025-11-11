# 🧬 **GUÍA COMPLETA DE CLUSTERING DE CANDIDATOS**

## 📊 **RESUMEN DEL SISTEMA IMPLEMENTADO**

### ✅ **SISTEMA COMPLETADO**

- **9,907 candidatos** procesados y entrenados
- **13 clusters** identificados con K-Means
- **Silhouette Score: 0.374** (buena calidad)
- **207 features** extraídas (numéricas, categóricas, TF-IDF)
- **4 archivos .pkl** generados con modelos entrenados

---

## 🎯 **OBJETIVOS LOGRADOS**

### **1. Machine Learning No Supervisado**

- ✅ **K-Means** entrenado con búsqueda automática de clusters óptimos
- ✅ **DBSCAN** implementado (requiere ajuste de parámetros)
- ✅ **Preprocessor especializado** para candidatos
- ✅ **Métricas de calidad** (Silhouette, Calinski-Harabasz, Davies-Bouldin)

### **2. Archivos .pkl Generados**

```
trained_models/clustering/
├── candidates_clustering_preprocessor_20251103_220705.pkl (0.0 MB)
├── candidates_clustering_kmeans_20251103_220705.pkl (0.1 MB)
├── candidates_clustering_dbscan_20251103_220705.pkl (0.1 MB)
└── candidates_clustering_data_20251103_220705.pkl (20.1 MB)
```

### **3. GraphQL API Implementada**

- ✅ `analyzeCandidateClusters` - Análisis completo
- ✅ `findSimilarCandidates` - Búsqueda de similares
- ✅ `getClusterProfileDetails` - Detalles de cluster

---

## 🚀 **CONSULTAS GRAPHQL DISPONIBLES**

### **1. ANÁLISIS COMPLETO DE CLUSTERING**

```graphql
query {
  analyzeCandidateClusters(input: { algorithm: "kmeans", maxResults: 10, includeOutliers: false }) {
    totalCandidates
    clustersFound
    outliersDetected
    algorithmUsed
    trainingDate
    metrics {
      silhouetteScore
      calinskiHarabaszScore
      daviesBouldinScore
      nClusters
      algorithmUsed
    }
    clusterProfiles {
      clusterId
      size
      percentage
      description
      topCharacteristics
      summary
    }
  }
}
```

**📊 Resultado Ejemplo:**

```json
{
  "totalCandidates": 9907,
  "clustersFound": 13,
  "outliersDetected": 0,
  "algorithmUsed": "kmeans",
  "metrics": {
    "silhouetteScore": 0.374,
    "nClusters": 13
  },
  "clusterProfiles": [
    {
      "clusterId": 3,
      "size": 4678,
      "percentage": 47.2,
      "description": "Cluster principal con el perfil más común",
      "topCharacteristics": ["Certificado en Administrator", "Especialista en Engineer", "Experiencia diversa"]
    }
  ]
}
```

### **2. BÚSQUEDA DE CANDIDATOS SIMILARES**

```graphql
query {
  findSimilarCandidates(
    input: {
      candidateId: "5166703f-e12d-4ab2-940e-b6cc8b120307"
      maxSimilar: 5
      algorithm: "kmeans"
      includeMetrics: true
    }
  ) {
    targetCandidateId
    targetClusterId
    similarityCriteria
    similarCandidates {
      candidateId
      clusterId
      clusterConfidence
      distanceToCenter
    }
  }
}
```

**🔍 Resultado Ejemplo:**

```json
{
  "targetCandidateId": "5166703f-e12d-4ab2-940e-b6cc8b120307",
  "targetClusterId": 3,
  "similarityCriteria": ["Mismo nivel de experiencia", "Skills técnicos similares", "Área educativa relacionada"],
  "similarCandidates": [
    {
      "candidateId": "abc123...",
      "clusterId": 3,
      "clusterConfidence": 0.85
    }
  ]
}
```

### **3. DETALLES DE CLUSTER ESPECÍFICO**

```graphql
query {
  getClusterProfileDetails(input: { clusterId: 3, algorithm: "kmeans", includeDetails: true }) {
    clusterId
    size
    percentage
    description
    topCharacteristics
    summary
  }
}
```

**📋 Resultado Ejemplo:**

```json
{
  "clusterId": 3,
  "size": 4678,
  "percentage": 47.2,
  "description": "Cluster principal con 4678 candidatos representando el perfil más común",
  "topCharacteristics": [
    "Certificado en Administrator",
    "Certificado en Certified Administrator",
    "Certificado en Engineer"
  ],
  "summary": "Cluster 3: 4678 candidatos (47.2%)"
}
```

### **4. CANDIDATOS DE UN CLUSTER ESPECÍFICO** ⭐ **NUEVA QUERY**

Obtiene todos los candidatos pertenecientes a un cluster con sus datos detallados.

```graphql
query {
  getCandidatesInCluster(input: { clusterId: 3, algorithm: "kmeans", includeDetails: true, limit: 20 }) {
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
```

**📊 Resultado Ejemplo:**

```json
{
  "clusterId": 3,
  "totalCandidates": 4678,
  "clusterPercentage": 47.2,
  "candidates": [
    {
      "candidateId": "507f1f77bcf86cd799439011",
      "name": "Juan Pérez García",
      "email": "juan.perez@example.com",
      "yearsExperience": 8.5,
      "educationArea": "Sistemas",
      "workArea": "Desarrollo",
      "skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS"],
      "certifications": ["AWS Solutions Architect", "Docker Certified Associate"],
      "englishLevel": "Avanzado",
      "distanceToCenter": 2.34
    }
  ]
}
```

```

---

## 🔧 **CARACTERÍSTICAS TÉCNICAS**

### **Features Extraídas (207 total):**

#### **Numéricas (5):**

- `anios_experiencia` - Años de experiencia
- `nivel_educacion_score` - Score ordinal de educación
- `seniority_score` - Nivel de seniority del puesto
- `num_idiomas` - Cantidad de idiomas
- `nivel_ingles` - Nivel de inglés (0-3)

#### **Categóricas (2):**

- `area_educacion` - Área de educación (sistemas, industrial, etc.)
- `area_trabajo` - Área de trabajo (desarrollo, management, etc.)

#### **TF-IDF Text (200):**

- `skills_*` - 100 features de habilidades técnicas
- `certs_*` - 100 features de certificaciones

### **Algoritmos Disponibles:**

- ✅ **K-Means** - Recomendado (13 clusters óptimos)
- ⚠️ **DBSCAN** - Requiere ajuste de parámetros
- 🔧 **Hierarchical** - Implementado pero no entrenado

---

## 📈 **ANÁLISIS DE CLUSTERS ENCONTRADOS**

### **Distribución de Candidatos:**

```

Cluster 0: 382 candidatos (3.9%) - Perfil especializado
Cluster 1: 492 candidatos (5.0%) - Perfil especializado
Cluster 2: 357 candidatos (3.6%) - Perfil especializado
Cluster 3: 4678 candidatos (47.2%) - PERFIL PRINCIPAL
Cluster 4: 461 candidatos (4.7%) - Perfil especializado
...
Cluster 12: 487 candidatos (4.9%) - Perfil especializado

````

### **Cluster Principal (Cluster 3):**

- 📊 **47.2%** de todos los candidatos
- 🎓 Educación: Predominantemente técnica
- 🏆 Certificaciones: Administrator, Engineer
- 💼 Experiencia: Diversa (0-15 años)

---

## 💡 **CASOS DE USO PRÁCTICOS**

### **1. Reclutamiento Inteligente**

```graphql
# Encontrar candidatos similares al mejor empleado actual
query {
  findSimilarCandidates(input: { candidateId: "mejor-empleado-id", maxSimilar: 10 }) {
    similarCandidates {
      candidateId
    }
  }
}
````

### **2. Análisis de Diversidad**

```graphql
# Ver distribución de perfiles en la empresa
query {
  analyzeCandidateClusters {
    clusterProfiles {
      clusterId
      size
      percentage
      topCharacteristics
    }
  }
}
```

### **3. Detección de Nichos**

```graphql
# Analizar clusters pequeños con perfiles únicos
query {
  getClusterProfileDetails(input: { clusterId: 0 }) {
    size
    topCharacteristics
    description
  }
}
```

### **4. Reclutamiento Masivo desde un Cluster** ⭐ **NUEVO**

```graphql
# Obtener candidatos completos de un cluster para exportación
query {
  getCandidatesInCluster(input: { clusterId: 3, algorithm: "kmeans", limit: 100 }) {
    clusterId
    totalCandidates
    candidates {
      candidateId
      name
      email
      yearsExperience
      skills
      certifications
    }
  }
}

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



🎉 **¡SISTEMA DE CLUSTERING DE CANDIDATOS COMPLETAMENTE OPERATIVO!** 🎉
```
