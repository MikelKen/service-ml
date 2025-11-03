# Sistema ML de Compatibilidad Candidato-Oferta

Este sistema utiliza Machine Learning para predecir la compatibilidad entre candidatos y ofertas de trabajo, entrenando modelos con datos de MongoDB.

## 🚀 Características

- **Entrenamiento automático** de modelos ML desde datos de MongoDB
- **Predicciones en tiempo real** a través de GraphQL
- **Múltiples algoritmos** (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- **Explicabilidad** de predicciones con feature importance
- **Ranking automático** de candidatos para ofertas específicas
- **API GraphQL** completa para interactuar con el sistema

## 📊 Datos Utilizados

El sistema extrae y combina datos de tres colecciones de MongoDB:

### Candidatos (candidates)

- `anios_experiencia`: Años de experiencia
- `habilidades`: Lista de habilidades técnicas
- `idiomas`: Idiomas que maneja
- `certificaciones`: Certificaciones obtenidas
- `nivel_educacion`: Nivel educativo
- `puesto_actual`: Posición actual

### Ofertas (job_offers)

- `titulo`: Título del puesto
- `salario`: Salario ofrecido
- `ubicacion`: Ubicación del trabajo
- `requisitos`: Requisitos técnicos y de experiencia

### Empresas (companies)

- `nombre`: Nombre de la empresa
- `rubro`: Sector de la empresa

## 🔧 Instalación y Configuración

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

Copia `.env.example` a `.env` y configura los valores:

```bash
cp .env.example .env
```

Edita `.env` con tu configuración de MongoDB:

```env
# MongoDB Configuration
DB_URL_MONGODB=mongodb://localhost:27017/
MONGODB_DATABASE=rrhh_db
```

### 3. Crear directorios necesarios

```bash
mkdir -p trained_models
mkdir -p data/processed
mkdir -p data/raw
```

## 🎯 Entrenamiento del Modelo

### Opción 1: Script directo

```bash
python scripts/train_model.py
```

### Opción 2: A través de GraphQL

```graphql
mutation {
  ml {
    trainCompatibilityModel {
      success
      message
      bestModel
      metrics
      trainingTime
    }
  }
}
```

## 🔮 Uso de Predicciones

### 1. Predicción individual

```graphql
query {
  predictCompatibility(
    input: { candidateId: "860d3462-51b2-4edc-8648-8a2198b92470", offerId: "1949bff6-245d-4f12-aff0-f1d8c83d8154" }
  ) {
    candidateId
    offerId
    probability
    prediction
    confidence
    modelUsed
  }
}
```

### 2. Top candidatos para una oferta

```graphql
query {
  getTopCandidatesForOffer(input: { offerId: "1949bff6-245d-4f12-aff0-f1d8c83d8154", topN: 10 }) {
    candidateId
    probability
    prediction
    confidence
    ranking
  }
}
```

### 3. Predicción batch

```graphql
query {
  predictBatchCompatibility(
    input: {
      pairs: [{ candidateId: "candidate1", offerId: "offer1" }, { candidateId: "candidate2", offerId: "offer1" }]
    }
  ) {
    predictions {
      candidateId
      offerId
      probability
      confidence
    }
    totalProcessed
    successCount
    errorCount
  }
}
```

### 4. Explicación de predicción

```graphql
query {
  explainPrediction(
    candidateId: "860d3462-51b2-4edc-8648-8a2198b92470"
    offerId: "1949bff6-245d-4f12-aff0-f1d8c83d8154"
  ) {
    prediction {
      probability
      confidence
    }
    keyFactors
    featureImportance {
      featureName
      importance
    }
    recommendation
  }
}
```

## 📈 Información del Modelo

### Ver información general

```graphql
query {
  modelInfo {
    modelName
    modelType
    isLoaded
    metrics
    featureImportanceCount
    topFeatures
  }
}
```

### Ver feature importance

```graphql
query {
  featureImportance(topN: 15) {
    features {
      featureName
      importance
    }
    totalFeatures
  }
}
```

### Ver métricas de rendimiento

```graphql
query {
  modelPerformance {
    accuracy
    precision
    recall
    f1Score
    rocAuc
    confusionMatrix
  }
}
```

### Estado del sistema

```graphql
query {
  modelStatus
  isModelLoaded
}
```

## 🧠 Algoritmo y Features

### Algoritmos Disponibles

1. **Random Forest** (recomendado)
2. **Gradient Boosting**
3. **Logistic Regression**
4. **Support Vector Machine**

### Features Principales

1. **Compatibilidad de habilidades**: Overlap entre skills del candidato y requisitos
2. **Experiencia vs salario**: Ratio de salario por año de experiencia
3. **Conteo de skills**: Número de habilidades técnicas
4. **Conteo de idiomas**: Cantidad de idiomas
5. **Score de educación**: Nivel educativo numerizado
6. **Similitud de posición**: Similitud entre puesto actual y ofertado
7. **Vectorización TF-IDF**: De habilidades y requisitos
8. **Encoding categórico**: De ubicación y posición actual

### Proceso de Entrenamiento

1. **Extracción**: Datos de MongoDB (candidatos, ofertas, empresas)
2. **Combinación**: Creación de pares candidato-oferta
3. **Generación de target**: 30% positivos, 70% negativos (configurable)
4. **Preprocesamiento**: Normalización, vectorización, encoding
5. **Entrenamiento**: Múltiples algoritmos con validación cruzada
6. **Selección**: Mejor modelo basado en ROC AUC
7. **Guardado**: Modelo y preprocessor en archivos .pkl

## 🎨 Estructura del Proyecto

```
app/
├── ml/
│   ├── data/
│   │   └── data_extractor.py          # Extracción desde MongoDB
│   ├── preprocessing/
│   │   ├── data_preprocessor.py       # Legacy preprocessor
│   │   └── mongo_preprocessor.py      # Nuevo preprocessor para MongoDB
│   ├── training/
│   │   └── model_trainer.py           # Entrenamiento de modelos
│   └── models/
│       └── predictor.py               # Predicciones
├── graphql/
│   ├── types/
│   │   └── ml_types.py                # Tipos GraphQL para ML
│   ├── resolvers/
│   │   └── ml_resolvers.py            # Resolvers ML
│   ├── mutations/
│   │   └── ml_mutations.py            # Mutaciones ML
│   └── schema.py                      # Esquema principal
└── config/
    ├── settings.py                    # Configuración
    └── mongodb_connection.py          # Conexión MongoDB
```

## 🔧 Configuración Avanzada

### Ajustar parámetros de entrenamiento

En `.env`:

```env
# Configuración de entrenamiento
ML_TEST_SIZE=0.2
ML_CROSS_VALIDATION_FOLDS=5
ML_ENABLE_HYPERPARAMETER_TUNING=true

# Umbrales de calidad
ML_MIN_ROC_AUC=0.75
ML_MIN_PRECISION=0.65
```

### Configurar generación de datos

```python
# En data_extractor.py
training_data = data_extractor.create_training_dataset(
    positive_samples_ratio=0.4,      # 40% positivos
    negative_samples_multiplier=3    # 3x más negativos
)
```

## 🚀 Ejecutar el Servidor

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 3001
```

Acceder a GraphiQL: http://localhost:3001/graphql

## 🏆 Ejemplos de Consultas Completas

### Flujo completo de entrenamiento y predicción

```graphql
# 1. Verificar estado del sistema
query {
  modelStatus
}

# 2. Entrenar modelo si es necesario
mutation {
  ml {
    trainCompatibilityModel {
      success
      message
      bestModel
      metrics
    }
  }
}

# 3. Verificar que el modelo esté cargado
query {
  isModelLoaded
  modelInfo {
    modelName
    isLoaded
  }
}

# 4. Realizar predicción
query {
  predictCompatibility(
    input: { candidateId: "860d3462-51b2-4edc-8648-8a2198b92470", offerId: "1949bff6-245d-4f12-aff0-f1d8c83d8154" }
  ) {
    probability
    prediction
    confidence
  }
}

# 5. Obtener explicación
query {
  explainPrediction(
    candidateId: "860d3462-51b2-4edc-8648-8a2198b92470"
    offerId: "1949bff6-245d-4f12-aff0-f1d8c83d8154"
  ) {
    recommendation
    featureImportance {
      featureName
      importance
    }
  }
}
```

## 📚 Notas Técnicas

### Calidad de Datos

- El sistema genera datos sintéticos de entrenamiento combinando candidatos y ofertas reales
- La proporción de muestras positivas/negativas es configurable
- Se aplica normalización de texto y encoding de variables categóricas

### Rendimiento

- Modelos entrenados se guardan en archivos `.pkl`
- Preprocessor incluido para mantener consistencia
- Predicciones optimizadas para tiempo real

### Escalabilidad

- Conexión asíncrona a MongoDB
- Procesamiento batch para múltiples predicciones
- Cacheable con Redis (futuro enhancement)

## 🐛 Troubleshooting

### Error: "No se pudieron obtener datos de entrenamiento"

- Verificar conexión a MongoDB
- Asegurar que existen datos en las colecciones `candidates`, `job_offers`, `companies`

### Error: "Modelo no cargado"

```bash
# Re-entrenar modelo
python scripts/train_model.py

# O via GraphQL
mutation { ml { trainCompatibilityModel { success message } } }
```

### Error de predicción

- Verificar que los IDs de candidato y oferta existen en MongoDB
- Revisar logs para errores específicos

## 📞 Soporte

Para problemas o mejoras, revisar los logs en `training.log` y consultar la documentación del código.
