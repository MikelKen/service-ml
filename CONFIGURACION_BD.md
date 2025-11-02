# Configuración de Base de Datos para el Microservicio ML

## Variables de Entorno (.env)

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
# Database Configuration
DB_URL_POSTGRES=postgresql://usuario:password@localhost:5432/hr_database

# MongoDB Configuration (opcional)
DB_URL_MONGODB=mongodb://localhost:27017
MONGODB_USERNAME=admin
MONGODB_PASSWORD=password
MONGODB_HOST=localhost:27017
MONGODB_DATABASE=ml_analytics

# Application Configuration
APP_NAME=service_ml
APP_VERSION=1.0.0
HOST=0.0.0.0
PORT=3001
DEBUG=true
ENVIRONMENT=development
```

## Configuración de PostgreSQL

### 1. Crear la Base de Datos

```sql
CREATE DATABASE hr_database;
```

### 2. Crear las Tablas

```sql
-- Tabla de empresas
CREATE TABLE empresas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nombre VARCHAR(255) NOT NULL,
    correo VARCHAR(255) NOT NULL,
    rubro VARCHAR(255) NOT NULL
);

-- Tabla de ofertas de trabajo
CREATE TABLE ofertas_trabajo (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    titulo VARCHAR(255) NOT NULL,
    descripcion TEXT,
    salario DECIMAL(10,2),
    ubicacion VARCHAR(255),
    requisitos TEXT,
    fecha_publicacion VARCHAR(50),
    empresa_id UUID NOT NULL REFERENCES empresas(id)
);

-- Tabla de postulaciones
CREATE TABLE postulaciones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nombre VARCHAR(255),
    anios_experiencia INTEGER,
    nivel_educacion VARCHAR(100),
    habilidades TEXT,
    idiomas VARCHAR(255),
    certificaciones TEXT,
    puesto_actual VARCHAR(255),
    url_cv VARCHAR(500),
    fecha_postulacion VARCHAR(50),
    estado VARCHAR(50),
    oferta_id UUID NOT NULL REFERENCES ofertas_trabajo(id)
);

-- Tabla de entrevistas
CREATE TABLE entrevistas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fecha VARCHAR(50),
    duracion_min INTEGER,
    objetivos_totales TEXT,
    objetivos_cubiertos TEXT,
    entrevistador VARCHAR(255),
    postulacion_id UUID NOT NULL REFERENCES postulaciones(id)
);

-- Tabla de evaluaciones
CREATE TABLE evaluaciones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calificacion_tecnica DECIMAL(3,2),
    calificacion_actitud DECIMAL(3,2),
    calificacion_general DECIMAL(3,2),
    comentarios TEXT,
    entrevista_id UUID NOT NULL REFERENCES entrevistas(id)
);

-- Tabla de visualizaciones de ofertas
CREATE TABLE visualizaciones_oferta (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fecha_visualizacion VARCHAR(50),
    origen VARCHAR(100),
    oferta_id UUID NOT NULL REFERENCES ofertas_trabajo(id)
);
```

### 3. Insertar Datos de Ejemplo

```sql
-- Empresas
INSERT INTO empresas (nombre, correo, rubro) VALUES
('TechCorp SA', 'contacto@techcorp.com', 'Tecnología'),
('InnovateLab', 'hr@innovatelab.com', 'Software'),
('DataSolutions', 'rrhh@datasolutions.com', 'Análisis de Datos');

-- Ofertas de trabajo
INSERT INTO ofertas_trabajo (titulo, descripcion, salario, ubicacion, requisitos, fecha_publicacion, empresa_id) VALUES
('Desarrollador Full Stack', 'Desarrollador con experiencia en React y Node.js', 5000.00, 'Santa Cruz', 'React, Node.js, PostgreSQL', '2024-10-01',
 (SELECT id FROM empresas WHERE nombre = 'TechCorp SA')),
('Data Scientist', 'Científico de datos para análisis predictivo', 6000.00, 'La Paz', 'Python, Machine Learning, SQL', '2024-10-15',
 (SELECT id FROM empresas WHERE nombre = 'DataSolutions'));

-- Postulaciones
INSERT INTO postulaciones (nombre, anios_experiencia, nivel_educacion, habilidades, idiomas, certificaciones, puesto_actual, fecha_postulacion, estado, oferta_id) VALUES
('Juan Pérez', 3, 'Licenciatura', 'React, Node.js, JavaScript', 'Español, Inglés', 'AWS Certified', 'Desarrollador Jr', '2024-10-02', 'aplicado',
 (SELECT id FROM ofertas_trabajo WHERE titulo = 'Desarrollador Full Stack')),
('María García', 5, 'Maestría', 'Python, Machine Learning, SQL', 'Español, Inglés', 'Google Cloud Certified', 'Analista de Datos', '2024-10-16', 'aplicado',
 (SELECT id FROM ofertas_trabajo WHERE titulo = 'Data Scientist'));

-- Entrevistas (para algunos candidatos)
INSERT INTO entrevistas (fecha, duracion_min, objetivos_totales, objetivos_cubiertos, entrevistador, postulacion_id) VALUES
('2024-10-05', 60, 'Evaluar habilidades técnicas y culturales', 'Habilidades técnicas excelentes', 'Carlos Manager',
 (SELECT id FROM postulaciones WHERE nombre = 'Juan Pérez'));

-- Evaluaciones
INSERT INTO evaluaciones (calificacion_tecnica, calificacion_actitud, calificacion_general, comentarios, entrevista_id) VALUES
(8.5, 9.0, 8.8, 'Candidato muy prometedor con buenas habilidades técnicas y actitud positiva',
 (SELECT id FROM entrevistas WHERE entrevistador = 'Carlos Manager'));
```

## Verificación de la Configuración

### 1. Probar Conexión

```bash
# Iniciar el servicio
python -m uvicorn app.main:app --reload --port 3001

# Verificar health check
curl http://localhost:3001/health

# Verificar conexión a base de datos ML
curl http://localhost:3001/api/ml/database/health-check
```

### 2. Obtener Datos de Entrenamiento

```bash
curl http://localhost:3001/api/ml/database/training-data
```

### 3. Verificar Información del Dataset

```bash
curl http://localhost:3001/api/ml/database/dataset-info
```

## Estructura de Datos para ML

### Features Utilizadas

**Del Candidato (postulaciones):**

- `anios_experiencia`: Años de experiencia laboral
- `nivel_educacion`: Nivel educativo (Bachillerato, Licenciatura, Maestría, etc.)
- `habilidades`: Habilidades técnicas
- `idiomas`: Idiomas que domina
- `certificaciones`: Certificaciones profesionales
- `puesto_actual`: Puesto de trabajo actual

**De la Oferta (ofertas_trabajo):**

- `titulo`: Título del puesto
- `salario`: Salario ofrecido
- `ubicacion`: Ubicación del trabajo
- `requisitos`: Requisitos del puesto

**De la Empresa (empresas):**

- `rubro`: Sector industrial
- `nombre`: Nombre de la empresa (para reputación)

### Variable Objetivo

- `target_contactado`: 1 si el candidato fue llamado a entrevista, 0 si no

### Métricas Derivadas

- `num_entrevistas`: Número de entrevistas realizadas
- `avg_calificacion_tecnica`: Promedio de calificaciones técnicas
- `avg_calificacion_actitud`: Promedio de calificaciones de actitud
- `avg_calificacion_general`: Promedio de calificaciones generales

## Troubleshooting

### Error de Conexión a la Base de Datos

1. Verificar que PostgreSQL esté ejecutándose
2. Verificar credenciales en el archivo `.env`
3. Verificar que la base de datos `hr_database` existe
4. Verificar que las tablas estén creadas

### No Hay Datos de Entrenamiento

1. Verificar que las tablas tengan datos
2. Ejecutar los scripts de inserción de datos de ejemplo
3. Verificar que las relaciones entre tablas estén correctas

### Modelo No Se Entrena

1. Verificar que haya al menos 50 registros en `postulaciones`
2. Verificar que haya algunos candidatos con entrevistas (datos positivos)
3. Revisar los logs del servicio para errores específicos

## Comandos Útiles

### Reiniciar Datos de Prueba

```sql
-- Limpiar datos existentes
TRUNCATE TABLE evaluaciones, entrevistas, visualizaciones_oferta, postulaciones, ofertas_trabajo, empresas CASCADE;

-- Reinsertar datos de ejemplo
-- (ejecutar los scripts de INSERT anteriores)
```

### Verificar Datos

```sql
-- Contar registros por tabla
SELECT 'empresas' as tabla, COUNT(*) as registros FROM empresas
UNION ALL
SELECT 'ofertas_trabajo', COUNT(*) FROM ofertas_trabajo
UNION ALL
SELECT 'postulaciones', COUNT(*) FROM postulaciones
UNION ALL
SELECT 'entrevistas', COUNT(*) FROM entrevistas
UNION ALL
SELECT 'evaluaciones', COUNT(*) FROM evaluaciones;

-- Verificar datos de entrenamiento
SELECT
    COUNT(*) as total_postulaciones,
    COUNT(CASE WHEN e.id IS NOT NULL THEN 1 END) as con_entrevista,
    COUNT(CASE WHEN e.id IS NULL THEN 1 END) as sin_entrevista
FROM postulaciones p
LEFT JOIN entrevistas e ON p.id = e.postulacion_id;
```
