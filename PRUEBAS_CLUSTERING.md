# 🔬 Guía de Pruebas - Clustering de Candidatos por Similitud

Esta guía te ayudará a probar todas las funcionalidades del microservicio de clustering para agrupar candidatos por similitud de perfil.

## 🚀 URLs del Servicio

- **GraphQL Playground**: http://127.0.0.1:8000/graphql
- **API REST Docs**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

---

## 📋 Tabla de Contenidos

1. [Entrenar el Modelo de Clustering](#1-entrenar-el-modelo-de-clustering)
2. [Ver Resultados del Clustering](#2-ver-resultados-del-clustering)
3. [Buscar Candidatos Similares](#3-buscar-candidatos-similares)
4. [Obtener Analíticas de Clusters](#4-obtener-analíticas-de-clusters)
5. [Probar Diferentes Algoritmos](#5-probar-diferentes-algoritmos)
6. [API REST Endpoints](#6-api-rest-endpoints)

---

## 1. 🤖 Entrenar el Modelo de Clustering

### GraphQL Mutation - Entrenamiento Básico

```graphql
mutation {
  trainClustering {
    totalCandidates
    numClusters
    silhouetteScore
    processingTimeMs
    clusters {
      clusterId
      clusterName
      candidateCount
      description
      avgExperienceYears
      avgSalaryExpectation
      commonSkills
      commonIndustries
    }
  }
}
```

### GraphQL Mutation - Entrenamiento con Parámetros Específicos

```graphql
mutation {
  trainClustering(parameters: { nClusters: 4, algorithm: "kmeans", maxClusters: 8 }) {
    totalCandidates
    numClusters
    silhouetteScore
    processingTimeMs
    clusters {
      clusterId
      clusterName
      candidateCount
      description
      keyCharacteristics
      avgExperienceYears
      avgSalaryExpectation
      commonSkills
      commonIndustries
      educationLevels
    }
    candidateAssignments {
      candidateId
      candidateName
      clusterId
      clusterName
      similarityScore
      distanceToCentroid
      profileSummary
    }
  }
}
```

---

## 2. 📊 Ver Resultados del Clustering

### GraphQL Query - Resultados Completos

```graphql
query {
  clusteringResults {
    totalCandidates
    numClusters
    silhouetteScore
    processingTimeMs
    clusters {
      clusterId
      clusterName
      candidateCount
      description
      keyCharacteristics
      avgExperienceYears
      avgSalaryExpectation
      commonSkills
      commonIndustries
      educationLevels
    }
    candidateAssignments {
      candidateId
      candidateName
      clusterId
      clusterName
      similarityScore
      distanceToCentroid
      profileSummary
    }
    modelParameters {
      key
      value
    }
  }
}
```

### GraphQL Query - Información de un Cluster Específico

```graphql
query {
  clusterById(clusterId: 0) {
    clusterId
    clusterName
    candidateCount
    description
    keyCharacteristics
    avgExperienceYears
    avgSalaryExpectation
    commonSkills
    commonIndustries
    educationLevels
  }
}
```

### GraphQL Query - Candidatos en un Cluster

```graphql
query {
  candidatesInCluster(clusterId: 0) {
    candidateId
    candidateName
    clusterId
    clusterName
    similarityScore
    distanceToCentroid
    profileSummary
  }
}
```

---

## 3. 🎯 Buscar Candidatos Similares

### GraphQL Query - Búsqueda de Similitud

```graphql
query {
  similarCandidates(
    candidateName: "Juan Pérez"
    experienceYears: 5
    specialtyArea: "Desarrollo Web"
    technicalSkills: "JavaScript, React, Node.js"
    softSkills: "Liderazgo, Comunicación"
    educationLevel: "Licenciatura"
    expectedSalary: 8000.0
    maxResults: 5
  ) {
    referenceCandidate {
      nombre
      edad
      experienceYears
      areaEspecialidad
      habilidadesTecnicas
      habilidadesBlandas
      salarioEsperado
      ubicacion
    }
    similarCandidates {
      candidateId
      candidateName
      clusterId
      clusterName
      similarityScore
      distanceToCentroid
      profileSummary
    }
    similarityCriteria
    totalFound
  }
}
```

### Ejemplo con Desarrollador Senior

```graphql
query {
  similarCandidates(
    candidateName: "María García"
    experienceYears: 8
    specialtyArea: "Desarrollo Backend"
    technicalSkills: "Python, Django, PostgreSQL, Docker"
    softSkills: "Resolución de problemas, Trabajo en equipo"
    educationLevel: "Maestría"
    expectedSalary: 12000.0
    maxResults: 10
  ) {
    referenceCandidate {
      nombre
      experienceYears
      areaEspecialidad
      salarioEsperado
    }
    similarCandidates {
      candidateName
      similarityScore
      profileSummary
      clusterName
    }
    totalFound
  }
}
```

---

## 4. 📈 Obtener Analíticas de Clusters

### GraphQL Query - Analíticas Completas

```graphql
query {
  clusteringAnalytics {
    clusterDistribution {
      name
      count
    }
    skillFrequency {
      name
      count
    }
    industryDistribution {
      name
      count
    }
    educationDistribution {
      name
      count
    }
    salaryRangesByCluster {
      key
      value
    }
    experienceRangesByCluster {
      key
      value
    }
  }
}
```

### GraphQL Query - Estado del Modelo

```graphql
query {
  isClusteringTrained
  clusteringModelInfo {
    isTrained
    numFeatures
    silhouetteScore
    modelType
    dataSize
  }
}
```

---

## 5. 🔄 Probar Diferentes Algoritmos

### Clustering K-Means

```graphql
mutation {
  trainKmeansClustering(nClusters: 5) {
    totalCandidates
    numClusters
    silhouetteScore
    clusters {
      clusterId
      clusterName
      candidateCount
      avgExperienceYears
    }
  }
}
```

### Clustering Jerárquico

```graphql
mutation {
  trainHierarchicalClustering(nClusters: 4, linkage: "ward") {
    totalCandidates
    numClusters
    silhouetteScore
    clusters {
      clusterId
      clusterName
      candidateCount
      description
    }
  }
}
```

### DBSCAN Clustering

```graphql
mutation {
  trainDbscanClustering(eps: 0.5, minSamples: 3) {
    totalCandidates
    numClusters
    silhouetteScore
    clusters {
      clusterId
      clusterName
      candidateCount
      keyCharacteristics
    }
  }
}
```

---

## 6. 🌐 API REST Endpoints

### Entrenar Clustering (POST)

```bash
curl -X POST "http://127.0.0.1:8000/api/clustering/train" \
-H "Content-Type: application/json" \
-d '{
  "algorithm": "kmeans",
  "n_clusters": 4,
  "parameters": {
    "max_iter": 300,
    "random_state": 42
  }
}'
```

### Obtener Resultados (GET)

```bash
curl -X GET "http://127.0.0.1:8000/api/clustering/results"
```

### Buscar Candidatos Similares (POST)

```bash
curl -X POST "http://127.0.0.1:8000/api/clustering/find-similar" \
-H "Content-Type: application/json" \
-d '{
  "candidate_profile": {
    "nombre": "Ana López",
    "experience_years": 6,
    "area_especialidad": "Data Science",
    "habilidades_tecnicas": "Python, Machine Learning, SQL",
    "habilidades_blandas": "Análisis, Comunicación",
    "nivel_educacion": "Maestría",
    "salario_esperado": 10000,
    "modalidad_trabajo": "Remoto",
    "ubicacion": "Santa Cruz"
  },
  "max_results": 8
}'
```

### Obtener Cluster por ID (GET)

```bash
curl -X GET "http://127.0.0.1:8000/api/clustering/cluster/0"
```

### Obtener Candidatos en Cluster (GET)

```bash
curl -X GET "http://127.0.0.1:8000/api/clustering/cluster/1/candidates"
```

---

## 🧪 Casos de Prueba Recomendados

### Caso 1: Desarrollador Junior

```graphql
query {
  similarCandidates(
    candidateName: "Carlos Mendoza"
    experienceYears: 2
    specialtyArea: "Desarrollo Frontend"
    technicalSkills: "HTML, CSS, JavaScript, Vue.js"
    softSkills: "Aprendizaje rápido, Creatividad"
    educationLevel: "Técnico"
    expectedSalary: 4500.0
    maxResults: 5
  ) {
    similarCandidates {
      candidateName
      similarityScore
      profileSummary
    }
    totalFound
  }
}
```

### Caso 2: Analista de Datos

```graphql
query {
  similarCandidates(
    candidateName: "Sofia Rodriguez"
    experienceYears: 4
    specialtyArea: "Análisis de Datos"
    technicalSkills: "Python, R, SQL, Tableau"
    softSkills: "Pensamiento analítico, Atención al detalle"
    educationLevel: "Licenciatura"
    expectedSalary: 7500.0
    maxResults: 6
  ) {
    similarCandidates {
      candidateName
      similarityScore
      clusterName
    }
    totalFound
  }
}
```

### Caso 3: Arquitecto de Software

```graphql
query {
  similarCandidates(
    candidateName: "Roberto Silva"
    experienceYears: 10
    specialtyArea: "Arquitectura de Software"
    technicalSkills: "Java, Spring Boot, Microservicios, AWS"
    softSkills: "Liderazgo técnico, Mentoring"
    educationLevel: "Maestría"
    expectedSalary: 15000.0
    maxResults: 4
  ) {
    similarCandidates {
      candidateName
      similarityScore
      profileSummary
    }
    totalFound
  }
}
```

---

## 📝 Validaciones Esperadas

### ✅ Resultados Exitosos

- **Clustering entrenado**: `silhouetteScore` > 0.3
- **Candidatos agrupados**: Cada cluster debe tener al menos 2 candidatos
- **Búsqueda de similitud**: Debe retornar candidatos ordenados por `similarityScore`
- **Analíticas**: Distribuciones coherentes por skills, educación, etc.

### ⚠️ Casos de Error

- **Modelo no entrenado**: Queries de búsqueda fallarán hasta entrenar
- **Parámetros inválidos**: DBSCAN con `eps` muy alto puede crear un solo cluster
- **Datos insuficientes**: Con pocos candidatos, algunos algoritmos pueden fallar

---

## 🎯 Flujo de Prueba Recomendado

1. **Entrenar el modelo** con parámetros por defecto
2. **Verificar resultados** y número de clusters generados
3. **Probar búsqueda de similitud** con diferentes perfiles
4. **Analizar métricas** para validar calidad del clustering
5. **Experimentar con algoritmos** diferentes (K-Means, Hierarchical, DBSCAN)
6. **Validar API REST** para integración con otros servicios

¡Tu microservicio de clustering está listo para agrupar candidatos por similitud de perfil! 🚀
