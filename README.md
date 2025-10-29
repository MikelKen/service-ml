# 🤖 Microservicio de ML para Predicción de Contratación

Sistema de Machine Learning que predice la probabilidad de que un candidato sea contactado por reclutadores, integrado con FastAPI + GraphQL.

## 🚀 Características

- **Predicción inteligente**: Modelo RandomForest que evalúa candidatos realísticamente
- **API GraphQL**: Interfaz moderna para consultas y mutaciones
- **Datos realistas**: Generación de datos sintéticos con lógica empresarial
- **Validación integrada**: Ejemplos de prueba categorizados
- **Dockerizado**: Listo para despliegue en contenedores

## 📁 Estructura del Proyecto

```
service_ml/
├── app/                          # API FastAPI + GraphQL
│   ├── __init__.py
│   ├── main.py                   # Aplicación principal
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Configuración
│   ├── graphql/
│   │   ├── __init__.py
│   │   ├── simple_ml.py          # Schema GraphQL ML
│   │   ├── ml_queries.py         # Consultas GraphQL
│   │   └── ml_mutations.py       # Mutaciones GraphQL
│   ├── routers/
│   │   ├── __init__.py
│   │   └── health.py             # Health check
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── ml_schemas.py         # Schemas Pydantic
│   └── services/
│       ├── __init__.py
│       └── ml_service.py         # Servicios ML
├── ml/                           # Módulos de Machine Learning
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Preprocesamiento
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predictor.py          # Predictor base
│   │   └── trainer.py            # Entrenador
│   └── utils/
│       └── __init__.py
├── trained_models/               # Modelos entrenados
│   └── simple_hiring_model.pkl  # Modelo principal
├── notebooks/                    # Jupyter notebooks
│   └── eda_analisis_exploratorio.ipynb
├── simple_predictor.py          # ✅ Predictor principal
├── train_simple.py              # ✅ Script de entrenamiento
├── generate_realistic_data.py   # ✅ Generador de datos
├── datos_entrenamiento_realista.csv  # ✅ Datos de entrenamiento
├── EJEMPLOS_VALIDACION.md       # ✅ Ejemplos para pruebas
├── Dockerfile                   # Configuración Docker
├── docker-compose.yml           # Orquestación Docker
└── README.md                    # Este archivo
```

## 🛠️ Instalación y Configuración

### Prerrequisitos

- Python 3.11+
- pip
- Docker (opcional)

### Instalación Local

1. **Clonar repositorio**

```bash
git clone <repository-url>
cd service_ml
```

2. **Crear entorno virtual**

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Instalar dependencias**

```bash
pip install pandas numpy scikit-learn joblib fastapi strawberry-graphql uvicorn
```

4. **Generar datos y entrenar modelo**

```bash
python generate_realistic_data.py
python train_simple.py
```

5. **Ejecutar servidor**

```bash
python -m uvicorn app.main:app --reload --port 8000
```

## 🎯 Uso de la API

### GraphQL Endpoint

- **URL**: http://localhost:8000/graphql
- **Interfaz gráfica**: Incluida para pruebas

### Mutación Principal: predictHiring

```graphql
mutation {
  predictHiring(
    nombre: "Juan Pérez"
    anosExperiencia: 5
    nivelEducacion: "licenciatura"
    habilidades: "python, machine learning, sql"
    idiomas: "español, inglés"
    certificaciones: "aws certified developer"
    titulo: "Data Scientist"
    requisitos: "python, sql, machine learning"
    salario: 18000
  ) {
    prediction # 0 o 1 (no/sí contactar)
    probability # Probabilidad (0.0-1.0)
    confidenceLevel # "Muy Baja" a "Muy Alta"
    recommendation # Recomendación textual
    modelUsed # Modelo utilizado
  }
}
```

### Consulta de Estado del Modelo

```graphql
query {
  modelStatus {
    isLoaded
    modelName
    accuracy
    featuresCount
  }
}
```

## 📊 Validación del Modelo

Utiliza el archivo `EJEMPLOS_VALIDACION.md` que contiene 10 casos de prueba categorizados:

- 🟢 **Muy Alta Probabilidad** (>80%): Candidatos ideales
- 🟡 **Alta Probabilidad** (60-80%): Buenos candidatos
- 🟠 **Probabilidad Media** (40-60%): Considerar cuidadosamente
- 🔴 **Baja Probabilidad** (20-40%): Probablemente no
- ⚫ **Muy Baja Probabilidad** (<20%): Definitivamente no

## 🧠 Cómo Funciona el Modelo

### Features Principales

1. **Años de experiencia** con optimización (penaliza extremos)
2. **Skill match** (coincidencia de habilidades requeridas)
3. **Salario por experiencia** (detecta candidatos caros/baratos)
4. **Nivel educativo** (técnico=1, licenciatura=2, maestría=3, doctorado=4)
5. **Certificaciones** (binario: tiene/no tiene)
6. **Número de habilidades**
7. **Tiempo desde publicación** del trabajo

### Lógica Empresarial

- **Experiencia óptima**: 3-12 años (penaliza 0-2 años y >15 años)
- **Skills relevantes**: Mayor coincidencia = mayor probabilidad
- **Costo-beneficio**: Salarios muy altos reducen probabilidad
- **Sobrecalificación**: Detecta candidatos demasiado senior para el puesto

## 🐳 Docker

### Construcción

```bash
docker-compose build
```

### Ejecución

```bash
docker-compose up
```

La aplicación estará disponible en http://localhost:8000

## 📈 Métricas del Modelo

- **Accuracy**: ~73%
- **Algoritmo**: RandomForest con 100 estimadores
- **Balanceado**: class_weight='balanced' para manejar desbalance
- **Features**: 12 características engineered

## 🔄 Reentrenamiento

Para reentrenar el modelo con nuevos datos:

```bash
# Generar nuevos datos sintéticos
python generate_realistic_data.py

# Entrenar modelo
python train_simple.py
```

## 🚨 Limitaciones Conocidas

1. **Datos sintéticos**: Basado en datos generados, no reales
2. **Simplicidad**: Modelo básico para demostración
3. **Features limitadas**: Solo considera información básica del CV
4. **Sin feedback loop**: No aprende de decisiones reales de RH

## 🔧 Configuración

Variables de entorno disponibles:

- `MODEL_PATH`: Ruta al modelo entrenado
- `DEBUG`: Modo debug (true/false)
- `HOST`: Host del servidor (default: 0.0.0.0)
- `PORT`: Puerto del servidor (default: 8000)

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver archivo [LICENSE](LICENSE) para detalles.

## 👥 Autores

- **Tu Nombre** - _Trabajo inicial_ - [TuGitHub](https://github.com/tuusuario)

## 🙏 Agradecimientos

- scikit-learn por el framework de ML
- FastAPI por la API moderna
- Strawberry GraphQL por la integración GraphQL
