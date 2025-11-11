# 🎯 GUÍA COMPLETA - PIPELINE SEMI-SUPERVISADO

Esta guía explica cómo usar el sistema completo de machine learning semi-supervisado para etiquetar postulaciones.

## 📋 ÍNDICE

1. [Prerrequisitos](#prerrequisitos)
2. [Configuración inicial](#configuración-inicial)
3. [Validación del sistema](#validación-del-sistema)
4. [Migración de datos](#migración-de-datos)
5. [Entrenamiento de modelos](#entrenamiento-de-modelos)
6. [Uso de GraphQL](#uso-de-graphql)
7. [Monitoreo y mantenimiento](#monitoreo-y-mantenimiento)

## 🔧 PRERREQUISITOS

### Bases de datos requeridas:

- **PostgreSQL**: Base de datos principal con tablas de postulaciones, ofertas, empresas
- **MongoDB**: Base de datos para almacenar datos procesados y modelos ML

### Dependencias Python:

```bash
pip install -r requirements.txt
```

### Variables de entorno:

Configurar en `app/config/settings.py`:

- `POSTGRES_URL`: URL de conexión a PostgreSQL
- `MONGODB_URL`: URL de conexión a MongoDB

## ⚙️ CONFIGURACIÓN INICIAL

### 1. Crear colecciones MongoDB

```bash
python create_mongo_collections_ml.py
```

### 2. Validar sistema completo

```bash
python validate_semi_supervised_pipeline.py
```

## 🧪 VALIDACIÓN DEL SISTEMA

El script de validación verifica:

- ✅ Conexiones a bases de datos
- ✅ Estructura de datos en PostgreSQL
- ✅ Colecciones MongoDB creadas
- ✅ Todos los componentes del pipeline
- ✅ GraphQL types, resolvers y mutations

## 📊 MIGRACIÓN DE DATOS

### 1. Extraer datos de PostgreSQL

```bash
python extract_postgres_data.py
```

### 2. Migrar a MongoDB con transformaciones ML

```bash
python migrate_postgres_to_mongo_ml.py
```

**Transformaciones aplicadas:**

- Vectorización de texto con TF-IDF
- Cálculo de compatibilidad candidato-oferta
- Ingeniería de características
- Preparación para semi-supervisado

## 🤖 ENTRENAMIENTO DE MODELOS

### Entrenamiento automático de todos los algoritmos:

```bash
python train_semi_supervised_step_by_step.py
```

**Algoritmos incluidos:**

- `label_propagation`: Propagación de etiquetas basada en grafos
- `label_spreading`: Similar a propagación pero más suave
- `self_training_rf`: Auto-entrenamiento con Random Forest
- `self_training_lr`: Auto-entrenamiento con Regresión Logística
- `self_training_gb`: Auto-entrenamiento con Gradient Boosting

**Proceso automático:**

1. 🔄 Preparación de datos (etiquetados/no etiquetados)
2. 🧠 Entrenamiento de cada algoritmo
3. 📊 Evaluación con métricas (accuracy, F1-score, precision, recall)
4. 💾 Guardado de modelos entrenados
5. 📈 Comparación de rendimiento
6. 📋 Generación de reportes

## 🔗 USO DE GRAPHQL

### Iniciar servidor FastAPI con GraphQL:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoint GraphQL:

```
http://localhost:8000/graphql
```

### Queries disponibles:

#### 1. Consultar modelos entrenados

```graphql
query GetModels {
  getSemiSupervisedModels {
    modelId
    algorithm
    version
    createdAt
    isActive
    performanceMetrics {
      trainAccuracy
      trainF1
      valAccuracy
      valF1
    }
    totalSamples
    labeledSamples
    unlabeledSamples
  }
}
```

#### 2. Obtener predicciones

```graphql
query GetPredictions($limit: Int) {
  getSemiSupervisedPredictions(limit: $limit) {
    applicationId
    predictedLabel
    confidence
    algorithm
    predictionDate
  }
}
```

#### 3. Rendimiento del modelo

```graphql
query GetPerformance($modelId: String!) {
  getModelPerformance(modelId: $modelId) {
    trainAccuracy
    trainF1
    valAccuracy
    valF1
  }
}
```

#### 4. Estadísticas del dataset

```graphql
query GetDatasetStats {
  datasetStatistics {
    totalApplications
    labeledApplications
    unlabeledApplications
    labeledRatio
    acceptedApplications
    rejectedApplications
    pendingApplications
    applicationsLastWeek
    applicationsLastMonth
    dataQualityScore
  }
}
```

#### 5. Información del modelo activo

```graphql
query GetActiveModel {
  activeModelInfo {
    modelId
    algorithm
    version
    createdAt
    isActive
    performanceMetrics {
      trainAccuracy
      trainPrecision
      trainRecall
      trainF1
      valAccuracy
      valPrecision
      valRecall
      valF1
      valRocAuc
      cvF1Mean
      cvF1Std
    }
    totalSamples
    labeledSamples
    unlabeledSamples
    labeledRatio
    positiveSamples
    negativeSamples
    nFeatures
  }
}
```

### Mutations disponibles:

**⚠️ Nota importante: Para usar las mutaciones, primero necesitas obtener IDs reales de aplicaciones de tu base de datos.**

#### 0. Obtener IDs de aplicaciones disponibles

```graphql
query {
  applicationsWithPredictions(pagination: { page: 1, pageSize: 10 }) {
    applications {
      applicationId
      candidateId
      offerId
      isLabeled
      mlTarget
    }
    totalApplications
    currentPage
  }
}
```

**O consultar aplicaciones específicas:**

```graphql
query {
  datasetStatistics {
    totalApplications
    labeledApplications
    unlabeledApplications
    sampleApplicationIds
  }
}
```

#### 1. Entrenar modelo

```graphql
mutation TrainModel($parameters: TrainingParameters!) {
  semiSupervised {
    trainSemiSupervisedModel(parameters: $parameters) {
      success
      modelId
      algorithm
      metrics {
        trainAccuracy
        trainF1
      }
      trainingTime
      message
    }
  }
}
```

#### 2. Hacer predicciones en lote

```graphql
mutation {
  semiSupervised {
    predictBatchApplications(
      batchInput: {
        applicationIds: ["app_001", "app_002", "app_003"]
        includeFeatures: false
        confidenceThreshold: 0.5
        updateDatabase: true
      }
    ) {
      batchId
      totalPredictions
      successfulPredictions
      failedPredictions
      predictions {
        applicationId
        candidateId
        offerId
        prediction
        probability
        confidenceLevel
        compatibilityScore
        predictedAt
        modelAlgorithm
      }
      errors
    }
  }
}
```

#### 3. Re-entrenar con nuevas etiquetas

```graphql
mutation RetrainModel($modelId: String!, $newLabels: [LabelInput!]!) {
  semiSupervised {
    retrainModelWithNewLabels(modelId: $modelId, newLabels: $newLabels) {
      success
      newModelId
      improvementMetrics {
        accuracyImprovement
        f1Improvement
      }
      message
    }
  }
}
```

#### 4. Activar modelo específico

```graphql
mutation ActivateModel($modelId: String!) {
  semiSupervised {
    activateModel(modelId: $modelId) {
      success
      message
      timestamp
    }
  }
}
```

## 📈 MONITOREO Y MANTENIMIENTO

### 1. Verificar estado del sistema:

```bash
python validate_semi_supervised_pipeline.py
```

### 2. Monitorear modelos en MongoDB:

```javascript
// Conectar a MongoDB y consultar
db.ml_model_tracking.find({ is_active: true }).sort({ trained_at: -1 });
```

### 3. Revisar reportes de entrenamiento:

- Ubicación: `training_reports/`
- Formato: JSON con métricas detalladas
- Contenido: Rendimiento, recomendaciones, comparaciones

### 4. Logs del sistema:

- Ubicación: `logs/`
- Archivos: Logs por fecha y componente
- Niveles: INFO, WARNING, ERROR

## 🚀 FLUJO DE TRABAJO COMPLETO

### Proceso inicial (primera vez):

1. **Configurar** bases de datos y variables de entorno
2. **Validar** sistema con `validate_semi_supervised_pipeline.py`
3. **Crear** colecciones MongoDB con `create_mongo_collections_ml.py`
4. **Migrar** datos con `migrate_postgres_to_mongo_ml.py`
5. **Entrenar** modelos con `train_semi_supervised_step_by_step.py`
6. **Iniciar** servidor GraphQL
7. **Probar** queries y mutations

### Proceso regular (uso continuo):

1. **Hacer predicciones** via GraphQL mutations
2. **Recopilar feedback** de usuarios sobre predicciones
3. **Re-entrenar** modelos con feedback acumulado
4. **Monitorear** rendimiento y métricas
5. **Actualizar** datos cuando sea necesario

## 📝 EJEMPLOS DE USO

### Ejemplo 1: Entrenar modelo Label Propagation

```bash
# 1. Migrar datos actualizados
python migrate_postgres_to_mongo_ml.py

# 2. Entrenar modelo específico
python -c "
import asyncio
from app.ml.models.semi_supervised_model import SemiSupervisedClassifier
from app.ml.preprocessing.semi_supervised_preprocessor import SemiSupervisedPreprocessor

async def train_lp():
    preprocessor = SemiSupervisedPreprocessor()
    X_labeled, y_labeled, X_unlabeled, _ = preprocessor.fit_transform()

    model = SemiSupervisedClassifier('label_propagation')
    result = model.train(X_labeled, y_labeled, X_unlabeled)
    model.save_model('trained_models/semi_supervised/label_propagation_custom.pkl')
    print(f'Modelo entrenado con F1-Score: {result[\"metrics\"][\"train_f1\"]:.4f}')

asyncio.run(train_lp())
"
```

### Ejemplo 2: Predicción via GraphQL

```bash
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation { predictApplicationLabels(input: { applicationIds: [\"app123\", \"app456\"], algorithm: \"label_propagation\" }) { predictions { applicationId predictedLabel confidence } success } }"
  }'
```

## 🆘 SOLUCIÓN DE PROBLEMAS

### Error de conexión a PostgreSQL:

- Verificar URL en `app/config/settings.py`
- Confirmar que PostgreSQL está ejecutándose
- Verificar permisos de usuario

### Error de conexión a MongoDB:

- Verificar URL en `app/config/settings.py`
- Confirmar que MongoDB está ejecutándose
- Verificar que las colecciones existen

### Errores de entrenamiento:

- Verificar que hay datos migrados en MongoDB
- Confirmar que hay suficientes datos etiquetados
- Revisar logs en `logs/` para detalles

### Errores de GraphQL:

- Verificar que el servidor FastAPI está ejecutándose
- Confirmar que los modelos están entrenados
- Revisar sintaxis de queries/mutations

## 📊 MÉTRICAS Y EVALUACIÓN

### Métricas clave:

- **Accuracy**: Porcentaje de predicciones correctas
- **F1-Score**: Media armónica de precisión y recall
- **Precision**: Porcentaje de positivos predichos que son correctos
- **Recall**: Porcentaje de positivos reales que se detectaron

### Interpretación:

- **F1-Score > 0.8**: Excelente rendimiento
- **F1-Score 0.6-0.8**: Buen rendimiento
- **F1-Score < 0.6**: Necesita mejoras (más datos etiquetados)

### Recomendaciones para mejora:

- **Datos insuficientes**: Etiquetar más postulaciones manualmente
- **Desbalance de clases**: Usar técnicas de balanceeo
- **Features pobres**: Mejorar ingeniería de características
- **Overfitting**: Ajustar hiperparámetros o usar regularización

---

¡El sistema está listo para etiquetar postulaciones de forma semi-supervisada! 🎉
