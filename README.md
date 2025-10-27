# ML Service - Microservicio de Machine Learning

Este es un microservicio de Machine Learning construido con FastAPI que proporciona APIs REST y GraphQL para recomendaciones de productos y análisis ML.

## 🚀 Características

- **FastAPI**: Framework web moderno y rápido
- **GraphQL**: API GraphQL con Strawberry
- **REST API**: Endpoints REST tradicionales
- **Machine Learning**: Integración con scikit-learn
- **Docker**: Completamente dockerizado
- **Health Checks**: Monitoreo de salud del servicio

## 📋 Requisitos

- Python 3.11+
- Docker y Docker Compose (recomendado)

## 🐳 Ejecutar con Docker

### Usando Docker Compose (Recomendado)

```bash
# Construir y ejecutar el servicio
docker-compose up --build

# Ejecutar en segundo plano
docker-compose up -d --build

# Ver logs
docker-compose logs -f

# Detener el servicio
docker-compose down
```

### Usando Docker directamente

```bash
# Construir la imagen
docker build -t ml-service .

# Ejecutar el contenedor
docker run -p 3001:3001 --name ml-service ml-service

# Ejecutar en segundo plano
docker run -d -p 3001:3001 --name ml-service ml-service
```

## 🛠️ Desarrollo Local

### 1. Crear entorno virtual

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
# Copiar archivo de ejemplo
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Editar .env con tus configuraciones
```

### 4. Ejecutar la aplicación

```bash
# Opción 1: Ejecutar directamente
python -m app.main

# Opción 2: Usar el script de ejecución
python run.py
```

## 📡 Endpoints

Una vez ejecutándose, el servicio estará disponible en:

- **Aplicación principal**: http://localhost:3001
- **Documentación API (Swagger)**: http://localhost:3001/docs
- **Documentación alternativa (ReDoc)**: http://localhost:3001/redoc
- **GraphQL Playground**: http://localhost:3001/graphql
- **API REST**: http://localhost:3001/api
- **Health Check**: http://localhost:3001/health

## 🔧 Configuración

### Variables de Entorno

| Variable       | Descripción              | Valor por defecto |
| -------------- | ------------------------ | ----------------- |
| `HOST`         | Host del servidor        | `0.0.0.0`         |
| `PORT`         | Puerto del servidor      | `3001`            |
| `DEBUG`        | Modo debug               | `true`            |
| `ENVIRONMENT`  | Entorno de ejecución     | `development`     |
| `CORS_ORIGINS` | Orígenes CORS permitidos | `*`               |
| `GRAPHQL_PATH` | Ruta de GraphQL          | `/graphql`        |
| `API_PREFIX`   | Prefijo de la API REST   | `/api`            |

## 🧪 Pruebas

```bash
# Ejecutar pruebas
pytest

# Ejecutar con cobertura
pytest --cov=app

# Ejecutar pruebas específicas
pytest tests/test_products.py
```

## 📦 Estructura del Proyecto

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # Aplicación principal
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py      # Configuraciones
│   ├── graphql/
│   │   ├── __init__.py
│   │   ├── schema.py        # Schema GraphQL
│   │   ├── queries.py       # Queries GraphQL
│   │   └── mutations.py     # Mutations GraphQL
│   ├── models/
│   │   ├── __init__.py
│   │   └── product.py       # Modelos de datos
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── products.py      # Rutas de productos
│   │   └── health.py        # Health checks
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── product_schema.py # Schemas Pydantic
│   └── services/
│       ├── __init__.py
│       └── product_service.py # Lógica de negocio
├── .env.example             # Variables de entorno de ejemplo
├── .gitignore              # Archivos ignorados por Git
├── .dockerignore           # Archivos ignorados por Docker
├── Dockerfile              # Configuración Docker
├── docker-compose.yml      # Configuración Docker Compose
├── requirements.txt        # Dependencias Python
├── run.py                  # Script de ejecución
└── README.md              # Este archivo
```

## 🔍 Monitoreo

### Health Checks

El servicio incluye health checks automáticos:

- **Endpoint**: `/health`
- **Docker Health Check**: Configurado en el Dockerfile
- **Intervalo**: 30 segundos
- **Timeout**: 10 segundos

### Logs

```bash
# Ver logs en Docker Compose
docker-compose logs -f ml-service

# Ver logs de contenedor específico
docker logs -f ml-service
```

## 🚀 Despliegue en Producción

### Consideraciones de Seguridad

1. **Variables de Entorno**: Configurar apropiadamente para producción
2. **CORS**: Restringir orígenes permitidos
3. **Debug**: Desactivar modo debug (`DEBUG=false`)
4. **Secrets**: Usar gestores de secretos para datos sensibles

### Configuración de Producción

```bash
# .env para producción
ENVIRONMENT=production
DEBUG=false
HOST=0.0.0.0
PORT=3001
CORS_ORIGINS=https://tu-dominio.com
```

## 🤝 Contribución

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

Si tienes problemas o preguntas:

1. Revisa la documentación
2. Verifica los logs del servicio
3. Crea un issue en el repositorio
4. Contacta al equipo de desarrollo
