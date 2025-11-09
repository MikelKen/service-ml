# Consultas GraphQL para Modelo Semi-Supervisado de Postulaciones

Este documento contiene ejemplos de consultas GraphQL para interactuar con el modelo semi-supervisado de predicción de estados de postulaciones.

## ⚡ Nota Importante sobre Tipos GraphQL

Los campos que representan diccionarios (como `estadoDistribution`, `tableStats`, `probabilityDistribution`, etc.) ahora usan tipos estructurados con subcampos `key` y `value`. Asegúrate de incluir estos subcampos en tus consultas:

```graphql
estadoDistribution {
  key    # Nombre del estado (ej: "Aprobado")
  value  # Cantidad o valor numérico
}
```

## 🔍 Consultas de Análisis de Datos

### 🚀 Consulta Simple de Prueba

```graphql
query TestSimpleQuery {
  getSemiSupervisedDataSummary {
    totalPostulaciones
    labeledPostulaciones
    unlabeledPostulaciones
    labeledPercentage
    canTrainSemiSupervised
    recommendations
  }
}
```

### 1. Obtener resumen de datos para semi-supervisado

```graphql
query GetSemiSupervisedDataSummary {
  getSemiSupervisedDataSummary {
    totalPostulaciones
    labeledPostulaciones
    unlabeledPostulaciones
    labeledPercentage
    estadoDistribution {
      key
      value
    }
    canTrainSemiSupervised
    recommendations
    missingDataPercentage
    completenessScore
    tableStats {
      key
      value
    }
  }
}
```

### 2. Analizar datos no etiquetados

```graphql
query AnalyzeUnlabeledData {
  analyzeUnlabeledData {
    totalUnlabeled
    predictedEstados {
      key
      value
    }
    confidenceStats {
      totalPredictions
      highConfidenceCount
      mediumConfidenceCount
      lowConfidenceCount
      confidenceDistribution {
        key
        value
      }
      reliablePredictions
      reviewNeededPredictions
      manualVerificationNeeded
    }
    commonPatterns
    labelingStrategy
  }
}
```

## 🤖 Entrenamiento de Modelos

### 3. Entrenar modelos semi-supervisados

```graphql
mutation TrainSemiSupervisedModels($config: SemiSupervisedTrainingInput) {
  trainSemiSupervisedModels(config: $config) {
    success
    message
    totalSamples
    labeledSamples
    unlabeledSamples
    featuresCount
    classesFound
    modelsTrained
    bestModelType
    bestModelScore
    trainingTime
    filesGenerated
    unlabeledPredictionsGenerated
    highConfidencePredictions
    errors
    modelsInfo {
      modelType
      isTrained
      trainAccuracy
      valAccuracy
      labeledSamples
      unlabeledSamples
      predictionConfidenceMean
      predictionDistribution {
        key
        value
      }
    }
  }
}
```

**Variables:**

```json
{
  "config": {
    "saveToMongo": true,
    "validationSplit": 0.2
  }
}
```

### 4. Obtener información de modelos entrenados

```graphql
query GetTrainedModelsInfo {
  getTrainedModelsInfo {
    modelType
    isTrained
    trainingTimestamp
    trainAccuracy
    valAccuracy
    labeledSamples
    unlabeledSamples
    totalSamples
    classes
    unlabeledPredictionsCount
    predictionConfidenceMean
    predictionDistribution {
      key
      value
    }
    modelPath
    metricsAvailable
  }
}
```

## 🔮 Predicciones

### 5. Predicción individual por ID de postulación

```graphql
query PredictPostulacionEstadoById($inputData: PostulacionEstadoPredictionInput!, $modelType: String) {
  predictPostulacionEstado(inputData: $inputData, modelType: $modelType) {
    postulacionId
    predictedEstado
    confidence
    confidenceLevel
    modelUsed
    predictionTimestamp
    keyFactors
    experienceScore
    skillsScore
    educationScore
    processingTimeMs
    probabilityDistribution {
      key
      value
    }
    error
  }
}
```

**Variables (por ID):**

```json
{
  "inputData": {
    "postulacionId": "123e4567-e89b-12d3-a456-426614174000"
  },
  "modelType": "label_propagation"
}
```

### 6. Predicción individual con datos manuales

```graphql
query PredictPostulacionEstadoManual($inputData: PostulacionEstadoPredictionInput!, $modelType: String) {
  predictPostulacionEstado(inputData: $inputData, modelType: $modelType) {
    predictedEstado
    confidence
    confidenceLevel
    modelUsed
    predictionTimestamp
    keyFactors
    experienceScore
    skillsScore
    educationScore
    processingTimeMs
    probabilityDistribution {
      key
      value
    }
    error
  }
}
```

**Variables (datos manuales):**

```json
{
  "inputData": {
    "nombre": "María González",
    "aniosExperiencia": 5,
    "nivelEducacion": "Universitario",
    "habilidades": "Python, Machine Learning, SQL, TensorFlow, scikit-learn",
    "idiomas": "Español, Inglés",
    "certificaciones": "AWS Certified, Google Cloud Professional",
    "puestoActual": "Data Scientist",
    "ofertaTitulo": "Senior Data Scientist",
    "ofertaSalario": 12000.0,
    "ofertaRequisitos": "5+ años experiencia, Python, ML, estadística",
    "empresaRubro": "Tecnología"
  },
  "modelType": "label_spreading"
}
```

### 7. Predicción en lote (batch)

```graphql
query PredictBatchEstados($inputData: BatchEstadoPredictionInput!) {
  predictBatchEstados(inputData: $inputData) {
    totalProcessed
    successCount
    errorCount
    modelUsed
    processingTime
    summaryStats {
      key
      value
    }
    predictions {
      predictedEstado
      confidence
      confidenceLevel
      keyFactors
      error
    }
  }
}
```

**Variables (batch):**

```json
{
  "inputData": {
    "postulaciones": [
      {
        "nombre": "Juan Pérez",
        "aniosExperiencia": 3,
        "nivelEducacion": "Técnico",
        "habilidades": "JavaScript, React, Node.js",
        "idiomas": "Español",
        "ofertaTitulo": "Frontend Developer",
        "ofertaSalario": 6000.0,
        "empresaRubro": "Startup"
      },
      {
        "nombre": "Ana Silva",
        "aniosExperiencia": 8,
        "nivelEducacion": "Postgrado",
        "habilidades": "Java, Spring, Microservices, Kubernetes",
        "idiomas": "Español, Inglés, Francés",
        "ofertaTitulo": "Tech Lead",
        "ofertaSalario": 18000.0,
        "empresaRubro": "Banca"
      },
      {
        "nombre": "Carlos López",
        "aniosExperiencia": 1,
        "nivelEducacion": "Universitario",
        "habilidades": "Python, Django, PostgreSQL",
        "idiomas": "Español, Inglés",
        "ofertaTitulo": "Backend Developer Jr",
        "ofertaSalario": 4500.0,
        "empresaRubro": "E-commerce"
      }
    ],
    "modelType": "self_training"
  }
}
```

## 📊 Consultas con Aliases (camelCase)

### 8. Usando aliases camelCase para compatibilidad

```graphql
query SemiSupervisedQueries {
  # Alias camelCase
  dataSummary: getSemiSupervisedDataSummary {
    totalPostulaciones
    labeledPostulaciones
    canTrainSemiSupervised
  }

  # Alias camelCase para modelos
  modelsInfo: getTrainedModelsInfo {
    modelType
    isTrained
    trainAccuracy
  }

  # Alias camelCase para análisis
  unlabeledAnalysis: analyzeUnlabeledData {
    totalUnlabeled
    predictedEstados
  }
}
```

## 🎯 Casos de Uso Específicos

### 9. Consulta completa para dashboard

```graphql
query SemiSupervisedDashboard {
  # Resumen general
  dataSummary: getSemiSupervisedDataSummary {
    totalPostulaciones
    labeledPostulaciones
    unlabeledPostulaciones
    labeledPercentage
    estadoDistribution {
      key
      value
    }
    canTrainSemiSupervised
  }

  # Modelos disponibles
  models: getTrainedModelsInfo {
    modelType
    isTrained
    trainAccuracy
    valAccuracy
    predictionConfidenceMean
    classes
  }

  # Análisis de datos no etiquetados
  unlabeledInsights: analyzeUnlabeledData {
    totalUnlabeled
    confidenceStats {
      reliablePredictions
      reviewNeededPredictions
      manualVerificationNeeded
    }
    labelingStrategy
  }
}
```

### 10. Predicción con análisis detallado

```graphql
query DetailedPrediction($inputData: PostulacionEstadoPredictionInput!) {
  prediction: predictPostulacionEstado(inputData: $inputData) {
    predictedEstado
    confidence
    confidenceLevel
    modelUsed
    predictionTimestamp
    keyFactors
    experienceScore
    skillsScore
    educationScore
    probabilityDistribution {
      key
      value
    }
    processingTimeMs
  }
}
```

**Variables:**

```json
{
  "inputData": {
    "nombre": "Laura Martínez",
    "aniosExperiencia": 4,
    "nivelEducacion": "Universitario",
    "habilidades": "Python, Data Science, Pandas, NumPy, Matplotlib, SQL",
    "idiomas": "Español, Inglés",
    "certificaciones": "Google Analytics, Tableau Certified",
    "puestoActual": "Data Analyst",
    "ofertaTitulo": "Data Scientist Senior",
    "ofertaSalario": 10000.0,
    "ofertaRequisitos": "Python, estadística, machine learning, 3+ años experiencia",
    "empresaRubro": "Consultoría"
  }
}
```

## 🚀 Flujo Completo de Uso

### Paso 1: Verificar viabilidad de datos

```graphql
query CheckDataViability {
  getSemiSupervisedDataSummary {
    canTrainSemiSupervised
    recommendations
    labeledPercentage
  }
}
```

### Paso 2: Entrenar modelos (si es viable)

```graphql
mutation TrainModels {
  trainSemiSupervisedModels(config: { saveToMongo: true }) {
    success
    bestModelType
    bestModelScore
  }
}
```

### Paso 3: Realizar predicciones

```graphql
query MakePredictions($data: PostulacionEstadoPredictionInput!) {
  predictPostulacionEstado(inputData: $data) {
    predictedEstado
    confidence
    confidenceLevel
  }
}
```

### Paso 4: Analizar resultados

```graphql
query AnalyzeResults {
  analyzeUnlabeledData {
    confidenceStats {
      reliablePredictions
      manualVerificationNeeded
    }
    labelingStrategy
  }
}
```

## 📝 Notas de Implementación

1. **Estados posibles**: Los estados predichos dependen de los datos de entrenamiento, típicamente: "Pendiente", "En Proceso", "Aprobado", "Rechazado", etc.

2. **Tipos de modelos**:

   - `label_propagation`: Propagación de etiquetas usando grafos
   - `label_spreading`: Variante mejorada de propagación
   - `self_training`: Auto-entrenamiento iterativo

3. **Niveles de confianza**:

   - `high`: > 0.8
   - `medium`: 0.6 - 0.8
   - `low`: < 0.6

4. **Archivos generados**: Los modelos se guardan como archivos `.pkl` en `trained_models/semi_supervised/`

5. **MongoDB**: Los resultados se almacenan en colecciones para análisis posterior
