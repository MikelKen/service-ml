# Integración con Base de Datos para ML

Este documento explica cómo usar las nuevas funcionalidades de integración con la base de datos PostgreSQL para el entrenamiento y predicción del modelo de Machine Learning.

## Configuración de Base de Datos

### Variables de Entorno Requeridas

Asegúrate de tener estas variables en tu archivo `.env`:

```env
DB_URL_POSTGRES=postgresql://usuario:password@localhost:5432/hr_database
```

### Estructura de Datos

El sistema usa las siguientes tablas de tu esquema de base de datos:

- **empresas**: Información de las empresas
- **ofertas_trabajo**: Ofertas de trabajo publicadas
- **postulaciones**: Aplicaciones de candidatos
- **entrevistas**: Entrevistas realizadas
- **evaluaciones**: Evaluaciones de las entrevistas

## Endpoints REST API

### 1. Obtener Datos de Entrenamiento

```http
GET /api/ml/database/training-data
```

Obtiene los datos procesados para entrenamiento desde la base de datos.

**Respuesta:**

```json
{
  "status": "success",
  "total_records": 500,
  "data": [...],
  "message": "Se obtuvieron 500 registros de entrenamiento"
}
```

### 2. Información del Dataset

```http
GET /api/ml/database/dataset-info
```

Obtiene estadísticas del dataset desde la base de datos.

### 3. Entrenar Modelo con Datos de BD

```http
POST /api/ml/database/train-model
```

Entrena el modelo usando datos directamente de la base de datos.

### 4. Predicciones para Nuevas Postulaciones

```http
GET /api/ml/database/predict-applications?empresa_id=uuid&limit=50
```

Realiza predicciones para postulaciones existentes en la base de datos.

### 5. Validar Calidad de Datos

```http
GET /api/ml/database/validate-data
```

Valida la calidad y consistencia de los datos de entrenamiento.

## GraphQL Queries y Mutations

### Queries

#### Información del Dataset desde BD

```graphql
query {
  databaseDatasetInfo {
    totalRecords
    positiveClassCount
    negativeClassCount
    classBalanceRatio
    companiesCount
    jobOffersCount
    avgExperienceYears
    avgSalary
    lastUpdated
    source
  }
}
```

### Mutations

#### Entrenar Modelo desde Base de Datos

```graphql
mutation {
  trainModelFromDatabase {
    isTraining
    progress
    statusMessage
    estimatedCompletion
  }
}
```

#### Predicciones para Nuevas Postulaciones

```graphql
mutation {
  predictNewApplicationsBatch(empresaId: "uuid-empresa") {
    totalApplications
    successfulPredictions
    failedPredictions
    predictions {
      hiringPrediction {
        prediction
        probability
        confidenceLevel
        recommendation
      }
      featureImportance {
        featureName
        importance
      }
      processingTimeMs
    }
  }
}
```

## Flujo de Trabajo Completo

### 1. Verificar Conexión y Datos

```bash
# Verificar health check
curl http://localhost:3001/api/ml/database/health-check

# Validar calidad de datos
curl http://localhost:3001/api/ml/database/validate-data
```

### 2. Obtener Información del Dataset

```bash
curl http://localhost:3001/api/ml/database/dataset-info
```

### 3. Entrenar el Modelo

```bash
curl -X POST http://localhost:3001/api/ml/database/train-model
```

### 4. Realizar Predicciones

```bash
# Predicciones para todas las postulaciones
curl http://localhost:3001/api/ml/database/predict-applications

# Predicciones para una empresa específica
curl "http://localhost:3001/api/ml/database/predict-applications?empresa_id=uuid-empresa"
```

## Consultas SQL Utilizadas

### Datos de Entrenamiento Agregados

La consulta principal combina todas las entidades relevantes:

```sql
SELECT
    -- Datos del postulante
    p.id as postulacion_id,
    p.nombre as candidato_nombre,
    p.anios_experiencia,
    p.nivel_educacion,
    p.habilidades,
    p.idiomas,
    p.certificaciones,
    p.puesto_actual,

    -- Datos de la oferta
    o.titulo as oferta_titulo,
    o.salario,
    o.ubicacion,
    o.requisitos,

    -- Datos de la empresa
    e.nombre as empresa_nombre,
    e.rubro as empresa_rubro,

    -- Métricas agregadas
    COUNT(DISTINCT ent.id) as num_entrevistas,
    AVG(ev.calificacion_tecnica) as avg_calificacion_tecnica,
    AVG(ev.calificacion_actitud) as avg_calificacion_actitud,
    AVG(ev.calificacion_general) as avg_calificacion_general,

    -- Variable objetivo
    CASE WHEN COUNT(ent.id) > 0 THEN 1 ELSE 0 END as target_contactado

FROM postulaciones p
JOIN ofertas_trabajo o ON p.oferta_id = o.id
JOIN empresas e ON o.empresa_id = e.id
LEFT JOIN entrevistas ent ON p.id = ent.postulacion_id
LEFT JOIN evaluaciones ev ON ent.id = ev.entrevista_id
GROUP BY p.id, o.id, e.id
```

## Características del Modelo

### Variables Predictoras (Features)

**Candidato:**

- Años de experiencia
- Nivel educativo
- Habilidades técnicas
- Idiomas
- Certificaciones
- Puesto actual

**Oferta:**

- Título del puesto
- Salario ofrecido
- Ubicación
- Requisitos

**Empresa:**

- Nombre/reputación
- Sector/rubro

### Variable Objetivo (Target)

- `target_contactado`: 1 si el candidato fue llamado a entrevista, 0 si no

### Interpretación de Resultados

- **Predicción = 1**: El modelo predice que el candidato será contactado
- **Probabilidad**: Confianza del modelo (0.0 - 1.0)
- **Confidence Level**: Nivel de confianza categórico (Alto/Medio/Bajo)
- **Feature Importance**: Factores más importantes en la decisión

## Consideraciones de Calidad de Datos

### Validaciones Automáticas

1. **Balance de Clases**: Verifica que no haya desbalance extremo
2. **Tamaño del Dataset**: Mínimo 100 registros, recomendado 500+
3. **Diversidad**: Múltiples niveles educativos y sectores industriales

### Recomendaciones

- Mantener al menos 30% de candidatos contactados en los datos
- Incluir diversidad en perfiles de candidatos y tipos de ofertas
- Actualizar el modelo regularmente con nuevos datos
- Monitorear el rendimiento del modelo en producción

## Monitoreo y Mantenimiento

### Métricas a Seguir

- Precisión del modelo en nuevas predicciones
- Distribución de probabilidades predichas
- Tiempo de respuesta de las predicciones
- Calidad de los datos de entrada

### Reentrenamiento

Se recomienda reentrenar el modelo:

- Cada 3 meses con nuevos datos
- Cuando el rendimiento baje significativamente
- Al introducir nuevos tipos de ofertas o sectores
