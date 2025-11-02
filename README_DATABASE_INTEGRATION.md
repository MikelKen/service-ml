# Microservicio ML - Integración con Base de Datos

## 🔄 Actualización: Integración con Base de Datos PostgreSQL

Este microservicio ahora se conecta directamente a tu base de datos PostgreSQL para obtener datos reales de postulaciones, ofertas de trabajo, entrevistas y evaluaciones para entrenar el modelo de Machine Learning.

## 🚀 Nuevas Funcionalidades

### 1. **Entrenamiento desde Base de Datos**

- Obtiene datos directamente de las tablas de tu sistema ERP
- Combina información de candidatos, ofertas, empresas y entrevistas
- Genera automáticamente la variable objetivo (contactado/no contactado)

### 2. **Predicciones en Tiempo Real**

- Predice probabilidad de contratación para nuevas postulaciones
- Utiliza datos actuales de la base de datos
- Filtra por empresa u oferta específica

### 3. **Análisis de Calidad de Datos**

- Valida balance de clases automáticamente
- Verifica tamaño y diversidad del dataset
- Proporciona recomendaciones de mejora

### 4. **APIs REST y GraphQL**

- Endpoints REST para integración directa
- Consultas GraphQL para análisis flexibles
- Documentación automática con FastAPI

## 🛠️ Configuración Rápida

### 1. Configurar Base de Datos

Crea un archivo `.env` con tu configuración:

```env
DB_URL_POSTGRES=postgresql://usuario:password@localhost:5432/hr_database
```

### 2. Ejecutar Script de Prueba

```bash
python test_database_integration.py
```

Este script:

- ✅ Verifica la conexión a la base de datos
- ✅ Obtiene datos de entrenamiento
- ✅ Entrena el modelo automáticamente
- ✅ Realiza predicciones de ejemplo
- ✅ Valida la calidad de los datos

### 3. Usar la API

```bash
# Iniciar el servicio
python -m uvicorn app.main:app --reload --port 3001

# Entrenar modelo desde BD
curl -X POST http://localhost:3001/api/ml/database/train-model

# Obtener predicciones
curl http://localhost:3001/api/ml/database/predict-applications
```

## 📊 Mapeo de Datos

### Entidades Utilizadas

| Tabla               | Campos Utilizados                                                                 | Propósito              |
| ------------------- | --------------------------------------------------------------------------------- | ---------------------- |
| **postulaciones**   | nombre, anios_experiencia, nivel_educacion, habilidades, idiomas, certificaciones | Features del candidato |
| **ofertas_trabajo** | titulo, salario, ubicacion, requisitos                                            | Features de la oferta  |
| **empresas**        | nombre, rubro                                                                     | Features de la empresa |
| **entrevistas**     | id (existencia)                                                                   | Variable objetivo      |
| **evaluaciones**    | calificaciones promedio                                                           | Métricas adicionales   |

### Variable Objetivo

```sql
CASE WHEN COUNT(entrevistas.id) > 0 THEN 1 ELSE 0 END as target_contactado
```

- **1**: El candidato fue contactado (tiene al menos una entrevista)
- **0**: El candidato no fue contactado

## 🎯 Endpoints Principales

### REST API

| Endpoint                                | Método | Descripción                     |
| --------------------------------------- | ------ | ------------------------------- |
| `/api/ml/database/training-data`        | GET    | Obtener datos de entrenamiento  |
| `/api/ml/database/train-model`          | POST   | Entrenar modelo desde BD        |
| `/api/ml/database/predict-applications` | GET    | Predicciones para postulaciones |
| `/api/ml/database/dataset-info`         | GET    | Información del dataset         |
| `/api/ml/database/validate-data`        | GET    | Validar calidad de datos        |

### GraphQL

```graphql
# Entrenar modelo desde base de datos
mutation {
  trainModelFromDatabase {
    isTraining
    progress
    statusMessage
  }
}

# Obtener información del dataset
query {
  databaseDatasetInfo {
    totalRecords
    positiveClassCount
    negativeClassCount
    classBalanceRatio
    companiesCount
    avgSalary
  }
}

# Predicciones por lotes
mutation {
  predictNewApplicationsBatch(empresaId: "uuid") {
    totalApplications
    successfulPredictions
    predictions {
      hiringPrediction {
        probability
        recommendation
      }
    }
  }
}
```

## 📈 Ejemplo de Uso Completo

### 1. Verificar Datos

```bash
# Verificar conexión y datos
curl http://localhost:3001/api/ml/database/health-check

# Ver información del dataset
curl http://localhost:3001/api/ml/database/dataset-info
```

**Respuesta:**

```json
{
  "status": "success",
  "dataset_info": {
    "total_records": 150,
    "positive_class_count": 45,
    "negative_class_count": 105,
    "class_balance_ratio": 0.3,
    "companies_count": 5,
    "avg_salary": 5500.0
  }
}
```

### 2. Entrenar Modelo

```bash
curl -X POST http://localhost:3001/api/ml/database/train-model
```

**Respuesta:**

```json
{
  "status": "success",
  "message": "Entrenamiento completado exitosamente",
  "model_info": {
    "model_name": "RandomForestClassifier",
    "is_loaded": true,
    "last_trained": "2024-11-01T12:00:00"
  }
}
```

### 3. Obtener Predicciones

```bash
# Todas las postulaciones
curl http://localhost:3001/api/ml/database/predict-applications

# Solo una empresa específica
curl "http://localhost:3001/api/ml/database/predict-applications?empresa_id=uuid-empresa"
```

**Respuesta:**

```json
{
  "status": "success",
  "total_predictions": 25,
  "predictions": [
    {
      "postulacion_id": "uuid-postulacion",
      "candidato_nombre": "Juan Pérez",
      "hiring_prediction": {
        "probability": 0.85,
        "recommendation": "Contactar - Alta probabilidad de éxito",
        "confidence_level": "Alto"
      },
      "feature_importance": [
        { "feature_name": "anios_experiencia", "importance": 0.3 },
        { "feature_name": "nivel_educacion", "importance": 0.25 }
      ]
    }
  ]
}
```

## 🔍 Validación de Datos

El sistema incluye validaciones automáticas:

- **Balance de Clases**: Alerta si hay menos del 10% o más del 90% de casos positivos
- **Tamaño del Dataset**: Requiere mínimo 100 registros, recomienda 500+
- **Diversidad**: Verifica variedad en niveles educativos y sectores

```bash
curl http://localhost:3001/api/ml/database/validate-data
```

## 🎛️ Configuración Avanzada

### Variables de Entorno

```env
# Base de datos principal
DB_URL_POSTGRES=postgresql://user:pass@host:5432/database

# Configuración de la aplicación
APP_NAME=service_ml
PORT=3001
DEBUG=true

# Opcional: MongoDB para analytics
DB_URL_MONGODB=mongodb://localhost:27017
MONGODB_DATABASE=ml_analytics
```

### Estructura de Carpetas

```
service_ml/
├── app/
│   ├── database/
│   │   ├── ml_queries.py       # 🆕 Consultas ML específicas
│   │   └── connection.py       # Conexión a PostgreSQL
│   ├── services/
│   │   └── ml_service.py       # 🔄 Actualizado con BD
│   ├── routers/
│   │   └── ml_database.py      # 🆕 Endpoints de BD
│   └── graphql/
│       ├── ml_queries.py       # 🔄 Queries actualizadas
│       └── ml_mutations.py     # 🔄 Mutations actualizadas
├── test_database_integration.py  # 🆕 Script de pruebas
├── CONFIGURACION_BD.md          # 🆕 Guía de configuración
└── INTEGRACION_BASE_DATOS.md    # 🆕 Documentación completa
```

## 🚨 Troubleshooting

### Problemas Comunes

1. **Error de conexión a BD:**

   - Verificar credenciales en `.env`
   - Confirmar que PostgreSQL esté ejecutándose
   - Probar conexión: `curl http://localhost:3001/api/ml/database/health-check`

2. **No hay datos de entrenamiento:**

   - Verificar que las tablas tengan datos
   - Ejecutar script: `python test_database_integration.py`
   - Revisar datos con: `curl http://localhost:3001/api/ml/database/training-data`

3. **Modelo no se entrena:**
   - Verificar mínimo 50 registros en postulaciones
   - Asegurar que algunos candidatos tengan entrevistas
   - Revisar balance de clases con validación

### Logs y Debugging

```bash
# Ver logs detallados
python -m uvicorn app.main:app --reload --port 3001 --log-level debug

# Probar funciones específicas
python -c "
import asyncio
from app.database.ml_queries import ml_db_queries
async def test():
    data = await ml_db_queries.get_training_data_aggregated()
    print(f'Datos obtenidos: {len(data)}')
asyncio.run(test())
"
```

## 📚 Documentación Adicional

- **[CONFIGURACION_BD.md](CONFIGURACION_BD.md)**: Guía detallada de configuración de PostgreSQL
- **[INTEGRACION_BASE_DATOS.md](INTEGRACION_BASE_DATOS.md)**: Documentación completa de la integración
- **[API Docs](http://localhost:3001/docs)**: Documentación interactiva de FastAPI
- **[GraphQL Playground](http://localhost:3001/graphql)**: Interfaz para probar GraphQL

## 🎉 ¡Listo!

Tu microservicio ML ahora está completamente integrado con tu base de datos PostgreSQL. El modelo se entrena automáticamente con datos reales y proporciona predicciones precisas para nuevas postulaciones.

**Próximos pasos:**

1. Configurar tu base de datos con el script proporcionado
2. Ejecutar las pruebas de integración
3. Integrar las predicciones en tu aplicación principal
4. Configurar reentrenamiento periódico del modelo
