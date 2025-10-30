# 🎯 Ejemplos GraphQL - Clustering de Candidatos

Esta guía contiene ejemplos específicos listos para copiar y pegar en el GraphQL Playground.

## 🚀 Acceso al GraphQL Playground

Abre tu navegador y ve a: **http://127.0.0.1:8000/graphql**

---

## 📋 Ejemplos Paso a Paso

### 1. 🏁 Paso Inicial - Entrenar el Modelo

**Copia y pega esto en GraphQL Playground:**

```graphql
mutation EntrenarClustering {
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
    candidateAssignments {
      candidateId
      candidateName
      clusterId
      clusterName
      profileSummary
    }
  }
}
```

**Resultado esperado:**

- ✅ `totalCandidates`: 50 (número de candidatos en el dataset)
- ✅ `numClusters`: 3-5 clusters generados automáticamente
- ✅ `silhouetteScore`: > 0.3 (calidad del clustering)

---

### 2. 🔍 Buscar Desarrollador Junior Similar

```graphql
query BuscarDesarrolladorJunior {
  similarCandidates(
    candidateName: "Desarrollador Junior Test"
    experienceYears: 2
    specialtyArea: "Desarrollo Frontend"
    technicalSkills: "HTML, CSS, JavaScript, React"
    softSkills: "Aprendizaje rápido, Creatividad, Adaptabilidad"
    educationLevel: "Técnico Superior"
    expectedSalary: 4500.0
    maxResults: 5
  ) {
    referenceCandidate {
      nombre
      experienceYears
      areaEspecialidad
      salarioEsperado
      ubicacion
    }
    similarCandidates {
      candidateId
      candidateName
      clusterId
      clusterName
      similarityScore
      profileSummary
    }
    totalFound
    similarityCriteria
  }
}
```

---

### 3. 🎯 Buscar Data Scientist Experimentado

```graphql
query BuscarDataScientist {
  similarCandidates(
    candidateName: "Data Scientist Senior"
    experienceYears: 7
    specialtyArea: "Ciencia de Datos"
    technicalSkills: "Python, Machine Learning, TensorFlow, SQL, R"
    softSkills: "Pensamiento analítico, Resolución de problemas"
    educationLevel: "Maestría"
    expectedSalary: 12000.0
    maxResults: 8
  ) {
    referenceCandidate {
      nombre
      experienceYears
      areaEspecialidad
      habilidadesTecnicas
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

### 4. 📊 Ver Información Completa de Clusters

```graphql
query InformacionClusters {
  clustersInfo {
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

---

### 5. 🎭 Ver Candidatos en un Cluster Específico

```graphql
query CandidatosEnCluster {
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

**💡 Tip:** Cambia `clusterId: 0` por `1`, `2`, etc. para ver otros clusters.

---

### 6. 📈 Analíticas y Estadísticas

```graphql
query AnalíticasClustering {
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

---

### 7. 🔄 Probar Diferentes Algoritmos

#### K-Means con 4 Clusters

```graphql
mutation KMeansClustering {
  trainKmeansClustering(nClusters: 4) {
    totalCandidates
    numClusters
    silhouetteScore
    processingTimeMs
    clusters {
      clusterId
      clusterName
      candidateCount
      avgExperienceYears
      commonSkills
    }
  }
}
```

#### Clustering Jerárquico

```graphql
mutation ClusteringJerarquico {
  trainHierarchicalClustering(nClusters: 3, linkage: "ward") {
    totalCandidates
    numClusters
    silhouetteScore
    clusters {
      clusterId
      clusterName
      candidateCount
      description
      keyCharacteristics
    }
  }
}
```

#### DBSCAN (Clustering por Densidad)

```graphql
mutation DBSCANClustering {
  trainDbscanClustering(eps: 0.5, minSamples: 3) {
    totalCandidates
    numClusters
    silhouetteScore
    clusters {
      clusterId
      clusterName
      candidateCount
      avgExperienceYears
      avgSalaryExpectation
    }
  }
}
```

---

### 8. 🏗️ Buscar Arquitecto de Software

```graphql
query BuscarArquitecto {
  similarCandidates(
    candidateName: "Arquitecto de Software"
    experienceYears: 10
    specialtyArea: "Arquitectura de Software"
    technicalSkills: "Java, Spring Boot, Microservicios, Docker, Kubernetes, AWS"
    softSkills: "Liderazgo técnico, Mentoring, Diseño de sistemas"
    educationLevel: "Maestría"
    expectedSalary: 18000.0
    maxResults: 4
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

### 9. ⚡ Query Rápida - Estado del Modelo

```graphql
query EstadoModelo {
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

### 10. 🎨 Buscar Diseñador UX/UI

```graphql
query BuscarDiseñador {
  similarCandidates(
    candidateName: "Diseñador UX/UI"
    experienceYears: 4
    specialtyArea: "Diseño UX/UI"
    technicalSkills: "Figma, Adobe XD, Sketch, Prototyping, HTML, CSS"
    softSkills: "Creatividad, Empatía, Comunicación visual"
    educationLevel: "Licenciatura"
    expectedSalary: 7000.0
    maxResults: 6
  ) {
    referenceCandidate {
      nombre
      experienceYears
      areaEspecialidad
    }
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

## 🎯 Flujo de Prueba Recomendado

1. **Ejecuta primero:** `EntrenarClustering` ⚠️ **OBLIGATORIO**
2. **Luego prueba:** `BuscarDesarrolladorJunior` o cualquier búsqueda de similitud
3. **Explora:** `InformacionClusters` para ver cómo se agruparon los candidatos
4. **Analiza:** `AnalíticasClustering` para estadísticas detalladas
5. **Experimenta:** Diferentes algoritmos (K-Means, Jerárquico, DBSCAN)

---

## 💡 Tips para Mejores Resultados

### Para Búsquedas de Similitud:

- **experienceYears**: Varía entre 1-15 años
- **specialtyArea**: Usa áreas como "Desarrollo Web", "Data Science", "DevOps", etc.
- **technicalSkills**: Lista 3-5 tecnologías separadas por comas
- **softSkills**: Incluye 2-3 habilidades blandas
- **expectedSalary**: Rango realista (3000-20000)

### Para Clustering:

- **K-Means**: Ideal para grupos claramente separados
- **Jerárquico**: Mejor para ver jerarquías en los datos
- **DBSCAN**: Detecta grupos de forma irregular y outliers

---

## 🚨 Solución de Problemas

### Si obtienes errores:

1. **"Model not trained"**: Ejecuta primero `EntrenarClustering`
2. **"No candidates found"**: Ajusta los parámetros de búsqueda
3. **Conexión falló**: Verifica que el servidor esté en `http://127.0.0.1:8000`

### Para validar que funciona:

- ✅ `silhouetteScore` > 0.3 indica buen clustering
- ✅ `totalFound` > 0 indica búsqueda exitosa
- ✅ `similarityScore` entre 0.0-1.0 (más alto = más similar)

¡Ahora puedes probar todo el sistema de clustering de candidatos! 🚀
