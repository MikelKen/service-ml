# 🤖 Microservicio ML - Sistema de Predicción de Contratación

Microservicio FastAPI con GraphQL para predecir la probabilidad de que un postulante sea contactado para una oferta de trabajo utilizando Machine Learning.

## 🎯 Características Principales

- **Predicción de Contratación**: Modelo ML supervisado que predice probabilidad de contacto
- **API GraphQL**: Interfaz moderna y flexible para consultas y mutaciones
- **Múltiples Algoritmos**: Logistic Regression, Random Forest, LightGBM, XGBoost
- **Feature Engineering**: Procesamiento avanzado de texto y features numéricas
- **Entrenamiento Asíncrono**: Entrenar modelos sin bloquear la API
- **Calibración de Probabilidades**: Probabilidades confiables y bien calibradas
- **Explicabilidad**: Feature importance para interpretabilidad

## 📊 Estructura del Proyecto

```
service_ml/
├── app/                          # Aplicación FastAPI
│   ├── config/                   # Configuración
│   ├── graphql/                  # Schemas GraphQL
│   │   ├── ml_queries.py         # Queries de ML
│   │   └── ml_mutations.py       # Mutaciones de ML
│   ├── schemas/                  # Schemas Pydantic/Strawberry
│   │   └── ml_schemas.py         # Schemas para ML
│   └── services/                 # Lógica de negocio
│       └── ml_service.py         # Servicio principal de ML
├── ml/                           # Módulos de Machine Learning
│   ├── data/                     # Procesamiento de datos
│   │   └── preprocessing.py      # Limpieza y preprocessado
│   ├── features/                 # Ingeniería de características
│   │   └── feature_engineering.py # Creación de features
│   ├── models/                   # Modelos de ML
│   │   ├── trainer.py            # Entrenamiento
│   │   └── predictor.py          # Predicción
│   └── utils/                    # Utilidades
├── trained_models/               # Modelos entrenados
├── notebooks/                    # Jupyter notebooks (EDA)
├── postulaciones_sinteticas_500.csv # Dataset de ejemplo
├── train_model.py               # Script de entrenamiento
├── run_demo.py                  # Script de demostración
└── requirements.txt             # Dependencias
```

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd service_ml

# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Entrenar el Modelo

```bash
# Entrenar modelo con datos de ejemplo
python train_model.py
```

### 3. Ejecutar el Servicio

```bash
# Iniciar servidor FastAPI
python -m uvicorn app.main:app --reload --port 3001

# O usar el script run.py si existe
python run.py
```

### 4. Probar el Sistema

```bash
# Demo rápida
python run_demo.py --simple

# Demo completa
python run_demo.py --full
```

## 📡 API GraphQL

El servicio expone una API GraphQL en `/graphql` con las siguientes capacidades:

### Queries

#### 🔍 Predicción de Contratación

```graphql
query PredictHiring {
  predictHiringProbability(predictionInput: {
    application: {
      nombre: "María González"
      añosExperiencia: 5
      nivelEducacion: "maestría"
      habilidades: "python, machine learning, sql"
      idiomas: "español, inglés"
      certificaciones: "aws cloud practitioner"
      puestoActual: "data scientist"
      industria: "tecnología"
    }
    jobOffer: {
      titulo: "Senior Data Scientist"
      descripcion: "Posición senior en data science"
      salario: 12000
      ubicacion: "Santa Cruz"
      requisitos: "python, machine learning, sql, aws"
    }
  }) {
    hiringPrediction {
      probability
      prediction
      confidenceLevel
      recommendation
      modelUsed
    }
    featureImportance {
      featureName
      importance
    }
    processingTimeMs
  }
}
```

#### 📊 Información del Modelo

```graphql
query ModelInfo {
  modelInfo {
    modelName
    isLoaded
    lastTrained
    version
  }

  modelMetrics {
    rocAuc
    precision
    recall
    f1Score
    accuracy
  }
}
```

#### 📈 Estado del Entrenamiento

```graphql
query TrainingStatus {
  trainingStatus {
    isTraining
    progress
    statusMessage
    estimatedCompletion
  }
}
```

### Mutaciones

#### 🔄 Predicción en Lote

```graphql
mutation BatchPredict {
  predictHiringBatch(predictions: [
    {
      application: { /* datos aplicante 1 */ }
      jobOffer: { /* datos oferta 1 */ }
    }
    {
      application: { /* datos aplicante 2 */ }
      jobOffer: { /* datos oferta 2 */ }
    }
  ]) {
    totalApplications
    successfulPredictions
    failedPredictions
    predictions {
      hiringPrediction {
        probability
        recommendation
      }
    }
  }
}
```

#### 🏋️ Entrenar Modelo

```graphql
mutation TrainModel {
  trainModel(dataPath: "postulaciones_sinteticas_500.csv") {
    isTraining
    progress
    statusMessage
  }
}
```

## 🧠 Modelo de Machine Learning

### Algoritmos Soportados

1. **Logistic Regression**: Baseline rápido y interpretable
2. **Random Forest**: Robusto con feature importance
3. **LightGBM**: Gradient boosting eficiente
4. **XGBoost**: Gradient boosting de alta performance

### Features Utilizadas

#### Numéricas

- `años_experiencia`: Años de experiencia laboral
- `salario`: Salario ofrecido
- `dias_desde_publicacion`: Días entre publicación y postulación
- `coincidencia_habilidades`: Overlap entre skills y requisitos
- `num_habilidades`: Número de habilidades del candidato
- `num_idiomas`: Número de idiomas del candidato

#### Categóricas (Encoded)

- `nivel_educacion`: Técnico, Licenciatura, Maestría, etc.
- `industria`: Sector de la empresa
- `ubicacion`: Ciudad/región
- `puesto_actual`: Posición actual del candidato

#### Texto (TF-IDF)

- Combinación de descripción del trabajo + requisitos
- Habilidades + certificaciones del candidato
- Título del puesto

#### Temporales

- `mes_postulacion`: Mes de la postulación
- `dia_semana_postulacion`: Día de la semana

### Métricas de Evaluación

- **ROC AUC**: Área bajo la curva ROC
- **PR AUC**: Área bajo la curva Precision-Recall
- **Precision**: Precisión para clase positiva
- **Recall**: Sensibilidad para clase positiva
- **F1 Score**: Media armónica de precision y recall

## 📊 Datos de Entrada

### Postulante (JobApplication)

```json
{
  "nombre": "string",
  "años_experiencia": "integer",
  "nivel_educacion": "string",
  "habilidades": "string (comma-separated)",
  "idiomas": "string (comma-separated)",
  "certificaciones": "string (optional)",
  "puesto_actual": "string",
  "industria": "string",
  "url_cv": "string (optional)",
  "fecha_postulacion": "string (ISO date, optional)"
}
```

### Oferta de Trabajo (JobOffer)

```json
{
  "titulo": "string",
  "descripcion": "string",
  "salario": "float",
  "ubicacion": "string",
  "requisitos": "string (comma-separated)",
  "fecha_publicacion": "string (ISO date, optional)"
}
```

## 🔧 Configuración

### Variables de Entorno (.env)

```env
# Servidor
HOST=0.0.0.0
PORT=3001
DEBUG=true
ENVIRONMENT=development

# Modelo
MODEL_PATH=trained_models/hiring_prediction_model.pkl
DATA_PATH=postulaciones_sinteticas_500.csv
```

## 🧪 Testing

```bash
# Demo rápida del sistema
python run_demo.py --simple

# Demo completa con todas las funcionalidades
python run_demo.py --full

# Verificar que el modelo funciona
python -c "from ml.models.predictor import HiringPredictor; print('✅ Importación exitosa')"
```

## 📈 Ejemplos de Uso

### Predicción Simple

```python
from ml.models.predictor import HiringPredictor

# Cargar modelo entrenado
predictor = HiringPredictor("trained_models/hiring_prediction_model.pkl")

# Datos de ejemplo
data = {
    'nombre': 'Juan Pérez',
    'años_experiencia': 3,
    'nivel_educacion': 'licenciatura',
    'habilidades': 'python, sql',
    'idiomas': 'español, inglés',
    'titulo': 'Data Scientist',
    'requisitos': 'python, machine learning',
    'salario': 8000
}

# Realizar predicción
result = predictor.predict_single(data)
print(f"Probabilidad: {result['probability']:.1%}")
print(f"Recomendación: {result['recommendation']}")
```

### Uso del Servicio ML

```python
from app.services.ml_service import MLService

# Inicializar servicio
ml_service = MLService()

# Verificar estado
if ml_service.is_model_loaded:
    print("✅ Modelo cargado correctamente")

    # Realizar predicción
    result = ml_service.predict_hiring_probability(
        application_data, job_offer_data
    )
    print(f"Probabilidad: {result['hiring_prediction']['probability']}")
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 3001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3001"]
```

### Docker Compose

```yaml
version: "3.8"
services:
  ml-service:
    build: .
    ports:
      - "3001:3001"
    environment:
      - ENV=production
    volumes:
      - ./trained_models:/app/trained_models
```

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

**⚡ ¡El futuro de la contratación inteligente comienza aquí!**
