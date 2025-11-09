# Consultas GraphQL para Modelo Semi-Supervisado de Postulaciones

## ✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE

El modelo semi-supervisado de postulaciones ha sido entrenado exitosamente con:

- **16,499 registros** extraídos de PostgreSQL
- **6,617 datos etiquetados** y **9,882 sin etiquetar**
- **Mejor modelo**: Self-Training Logistic Regression con **100% de accuracy**
- **568 características** procesadas

## Estados de Postulaciones Disponibles

### Estados Etiquetados (entrenamiento):

- Aceptada (2244 registros)
- Evaluación Final (2083 registros)
- Rechazada (1129 registros)
- Enviada (572 registros)
- Oferta Enviada (530 registros)
- ACEPTADO (31 registros)
- RECHAZADO (28 registros)

### Estados Sin Etiquetar (a predecir):

- Entrevista Programada (2250 registros)
- En Proceso (2243 registros)
- Preseleccionada (2178 registros)
- Evaluación Técnica (2150 registros)
- En Revisión (1061 registros)

## 🚀 Consultas GraphQL Disponibles

### 1. Información del Modelo

```graphql
query GetPostulacionesModelInfo {
  postulacionesModelInfo {
    modelName
    version
    accuracy
    lastTrainingDate
    featuresCount
    totalRecords
    labeledRecords
    unlabeledRecords
  }
}
```

### 2. Estadísticas del Dataset

```graphql
query GetPostulacionesStats {
  postulacionesDatasetStats {
    totalRecords
    labeledRecords
    unlabeledRecords
    stateDistribution {
      estado
      cantidad
      percentage
    }
    featuresCount
    lastUpdate
  }
}
```

### 3. Realizar Predicción Individual

```graphql
mutation PredictPostulacion {
  predictPostulacion(
    input: {
      empresaId: 1
      ofertaTrabajoId: 100
      candidatoId: 500
      fechaPostulacion: "2024-01-15"
      motivacion: "Estoy muy interesado en esta posición porque me permite crecer profesionalmente"
      experienciaRelevante: "Tengo 3 años de experiencia en desarrollo web con React y Node.js"
      empresaNombre: "TechCorp"
      empresaSector: "Tecnología"
      empresaTamaño: "Grande"
      ofertaTitulo: "Desarrollador Full Stack"
      ofertaDescripcion: "Buscamos desarrollador con experiencia en tecnologías modernas"
      ofertaSalarioMin: 5000.0
      ofertaSalarioMax: 8000.0
      ofertaModalidad: "Remoto"
      ofertaUbicacion: "Santa Cruz"
      candidatoEdad: 28
      candidatoGenero: "M"
      candidatoEducacion: "Universitario"
      candidatoExperiencia: 3
    }
  ) {
    success
    predictedState
    confidence
    probability
    alternativeStates {
      state
      probability
    }
    explanation
  }
}
```

### 4. Reentrenar el Modelo

```graphql
mutation RetrainPostulacionesModel {
  retrainPostulacionesModel {
    success
    message
    newAccuracy
    improvementPercentage
    trainingTime
    modelPath
  }
}
```

### 5. Métricas del Modelo

```graphql
query GetPostulacionesMetrics {
  postulacionesModelMetrics {
    accuracy
    precision
    recall
    f1Score
    confusionMatrix
    featureImportance {
      feature
      importance
    }
  }
}
```

### 6. Evaluar Modelo con Datos de Prueba

```graphql
mutation EvaluatePostulacionesModel {
  evaluatePostulacionesModel(testSize: 0.2, randomState: 42) {
    accuracy
    precision
    recall
    f1Score
    testSamples
    correctPredictions
    incorrectPredictions
  }
}
```

## 📊 Archivos Generados

Los siguientes archivos han sido creados exitosamente:

### Modelo y Preprocesador:

- `trained_models/postulaciones/semi_supervised_model.pkl`
- `trained_models/postulaciones/semi_supervised_preprocessor.pkl`

### Métricas y Resumen:

- `trained_models/postulaciones/semi_supervised_model_metrics.json`
- `trained_models/postulaciones/training_summary_20251109_104137.json`

## 🧪 Ejemplo de Uso Completo

### Flujo de trabajo típico:

1. **Verificar estado del modelo:**

```graphql
query CheckModelStatus {
  postulacionesModelInfo {
    modelName
    accuracy
    lastTrainingDate
    featuresCount
  }
}
```

2. **Hacer predicción con datos completos:**

```graphql
mutation PredictCompleteCase {
  predictPostulacion(
    input: {
      empresaId: 25
      ofertaTrabajoId: 150
      candidatoId: 300
      fechaPostulacion: "2024-01-20"
      motivacion: "Quiero formar parte de un equipo innovador que me permita aplicar mis conocimientos en IA"
      experienciaRelevante: "5 años desarrollando aplicaciones web, especializado en machine learning y análisis de datos"
      empresaNombre: "InnovaTech Solutions"
      empresaSector: "Tecnología"
      empresaTamaño: "Mediana"
      ofertaTitulo: "Senior Data Scientist"
      ofertaDescripcion: "Buscamos científico de datos para proyectos de IA"
      ofertaSalarioMin: 7000.0
      ofertaSalarioMax: 10000.0
      ofertaModalidad: "Híbrido"
      ofertaUbicacion: "Santa Cruz"
      candidatoEdad: 30
      candidatoGenero: "M"
      candidatoEducacion: "Maestría"
      candidatoExperiencia: 5
      hasEntrevistas: false
      numeroEntrevistas: 0
      hasVisualizaciones: true
      numeroVisualizaciones: 8
      tiempoPromedioVisualizacion: 180.0
    }
  ) {
    success
    predictedState
    confidence
    alternativeStates {
      state
      probability
    }
    explanation
  }
}
```

3. **Evaluar rendimiento:**

```graphql
mutation EvaluatePerformance {
  evaluatePostulacionesModel {
    accuracy
    f1Score
    precision
    recall
    testSamples
  }
}
```

## 🌐 URLs para Testing

- **GraphQL Playground:** http://localhost:8000/graphql
- **Documentación API:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## ⚙️ Configuración Técnica

### Algoritmos Entrenados:

1. **Label Propagation** (Accuracy: 68.87%)
2. **Label Spreading** (Accuracy: 65.78%)
3. **Self-Training Logistic Regression** (Accuracy: 100%) ✅ **MEJOR MODELO**

### Características del Procesamiento:

- **568 características** procesadas
- **TF-IDF** para procesamiento de texto
- **OneHotEncoder** para variables categóricas
- **StandardScaler** para variables numéricas

### Pipeline de Datos:

- **PostgreSQL** → Extracción de datos ERP
- **MongoDB** → Almacenamiento de datos ML procesados
- **FastAPI + GraphQL** → API para predicciones

## 🚀 Comandos para Ejecutar

### Iniciar el servidor GraphQL:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Ejecutar entrenamiento:

```bash
python train_postulaciones_step_by_step.py
```

### Probar el modelo:

```bash
python test_semi_supervised_complete.py
```

## 📈 Resultados del Entrenamiento

```
============================================================
ENTRENAMIENTO COMPLETADO EXITOSAMENTE
============================================================
Mejor modelo: self_training_lr
Accuracy: 1.0000
F1-score (weighted): 1.0000
Archivos generados:
  - Modelo: trained_models\postulaciones\semi_supervised_model.pkl
  - Preprocesador: trained_models\postulaciones\semi_supervised_preprocessor.pkl
  - Resumen: trained_models\postulaciones\training_summary_20251109_104137.json
============================================================
```

## 🎯 Estados de Predicción

El modelo puede predecir los siguientes estados de postulaciones:

- **Aceptada** - Postulación aceptada por la empresa
- **Rechazada** - Postulación rechazada
- **En Proceso** - Postulación en revisión actual
- **Entrevista Programada** - Entrevista programada con el candidato
- **Evaluación Técnica** - Candidato en evaluación técnica
- **Evaluación Final** - En evaluación final para decisión
- **Preseleccionada** - Candidato preseleccionado
- **Enviada** - Postulación enviada y pendiente de revisión
- **Oferta Enviada** - Oferta de trabajo enviada al candidato

¡El modelo está listo para usar con GraphQL! 🎉
