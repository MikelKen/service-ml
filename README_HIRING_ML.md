# Microservicio ML para Sistema de RRHH

Este microservicio proporciona capacidades de machine learning para el sistema de recursos humanos, específicamente para predecir la probabilidad de contratación de candidatos.

## 🎯 Características

- **Predicción de Contratación**: Modelo supervisado que predice si un candidato será contactado
- **API GraphQL**: Integración completa con GraphQL para consultas y mutaciones
- **FastAPI**: Framework moderno y rápido para APIs
- **Procesamiento de Texto**: Análisis de habilidades y requisitos usando NLP
- **Múltiples Algoritmos**: Soporte para diferentes modelos de ML
- **Sistema Completo**: Desde datos sintéticos hasta predicciones en producción

## 📁 Estructura del Proyecto

```
service_ml/
├── app/                     # Aplicación FastAPI
│   ├── main.py             # Punto de entrada
│   ├── config/             # Configuración
│   ├── graphql/            # Esquemas y resolvers GraphQL
│   ├── routers/            # Rutas REST (opcional)
│   ├── schemas/            # Esquemas Pydantic
│   └── services/           # Lógica de negocio
├── ml/                     # Módulos de Machine Learning
│   ├── data/               # Procesamiento de datos
│   ├── features/           # Ingeniería de características
│   ├── models/             # Entrenamiento y predicción
│   └── utils/              # Utilidades
├── notebooks/              # Jupyter notebooks para análisis
├── trained_models/         # Modelos entrenados
├── demo_complete.py        # Demostración completa del sistema
├── train_simple.py         # Entrenamiento simplificado
├── simple_predictor.py     # Predictor simple para pruebas
└── requirements.txt        # Dependencias
```

## 🚀 Inicio Rápido

### Demostración Completa

La forma más rápida de ver el sistema funcionando:

```bash
python demo_complete.py
```

Este script:

1. ✅ Genera datos sintéticos realistas
2. ✅ Entrena un modelo simple pero efectivo
3. ✅ Demuestra predicciones con candidatos de ejemplo
4. ✅ Muestra cómo usar la API GraphQL

### Instalación Paso a Paso

1. **Instalar dependencias**:

```bash
pip install fastapi uvicorn strawberry-graphql pandas scikit-learn numpy joblib
```

2. **Entrenar modelo simple**:

```bash
python train_simple.py
```

3. **Probar predicciones**:

```bash
python simple_predictor.py
```

4. **Ejecutar API**:

```bash
uvicorn app.main:app --reload
```

5. **Acceder a GraphQL**: `http://localhost:8000/graphql`

## 🔮 Uso del Sistema

### Predicción Simple

```python
from simple_predictor import SimpleHiringPredictor

# Cargar modelo entrenado
predictor = SimpleHiringPredictor("trained_models/simple_hiring_model.pkl")

# Datos del candidato
candidato = {
    'nombre': 'Ana García',
    'años_experiencia': 5,
    'nivel_educacion': 'maestría',
    'habilidades': 'python, machine learning, sql',
    'certificaciones': 'aws cloud practitioner',
    'titulo': 'Data Scientist',
    'requisitos': 'python, machine learning, sql, 3+ años'
}

# Realizar predicción
resultado = predictor.predict(candidato)
print(f"Probabilidad: {resultado['probability']:.1%}")
print(f"Recomendación: {resultado['recommendation']}")
```

### API GraphQL

**Mutación para Predicción:**

```graphql
mutation {
  predictHiring(
    nombre: "Elena Morales"
    anosExperiencia: 8
    nivelEducacion: "maestría"
    habilidades: "python, machine learning, sql, tensorflow"
    idiomas: "español, inglés"
    certificaciones: "aws cloud practitioner"
    puestoActual: "senior data scientist"
    industria: "tecnología"
    titulo: "Data Scientist Senior"
    descripcion: "Liderar proyectos de ML"
    salario: 20000
    ubicacion: "santa cruz"
    requisitos: "python, machine learning, sql, 5+ años exp"
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Respuesta Esperada:**

```json
{
  "data": {
    "predictHiring": {
      "prediction": 1,
      "probability": 0.85,
      "confidenceLevel": "Muy Alta",
      "recommendation": "Fuertemente recomendado para entrevista",
      "modelUsed": "RandomForestClassifier"
    }
  }
}
```

## 🧠 Cómo Funciona el Modelo

### Features Principales

1. **Análisis de Habilidades**:

   - Calcula overlap entre habilidades del candidato y requisitos
   - Cuenta número de habilidades relevantes

2. **Experiencia**:

   - Años de experiencia
   - Ratio salario/experiencia

3. **Educación**:

   - Nivel educativo (técnico=1, licenciatura=2, maestría=3, doctorado=4)

4. **Certificaciones**:

   - Presencia de certificaciones profesionales

5. **Temporales**:
   - Días desde publicación del trabajo
   - Mes de postulación

### Interpretación de Resultados

- **Probabilidad > 70%**: "Fuertemente recomendado para entrevista"
- **Probabilidad 50-70%**: "Recomendado para entrevista"
- **Probabilidad 30-50%**: "Considerar para entrevista"
- **Probabilidad < 30%**: "No recomendado en esta ronda"

## 📊 Datos de Entrenamiento

El sistema puede trabajar con:

1. **Datos Sintéticos** (incluidos): 500 registros generados automáticamente
2. **Datos Reales**: CSV con columnas específicas
3. **Datos de Demostración**: Generados dinámicamente para pruebas

### Formato de Datos

```csv
nombre,años_experiencia,nivel_educacion,habilidades,idiomas,certificaciones,
puesto_actual,industria,titulo,descripcion,salario,ubicacion,requisitos,
fecha_postulacion,fecha_publicacion,contactado
```

## 🔧 Personalización

### Agregar Nuevas Features

Editar `ml/features/feature_engineering.py`:

```python
def create_custom_feature(self, df):
    """Tu nueva feature aquí"""
    df['mi_feature'] = df['campo1'] * df['campo2']
    return df
```

### Cambiar Modelo

Editar `train_simple.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Reemplazar RandomForestClassifier
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
```

## 🐳 Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ml-hiring-service .
docker run -p 8000:8000 ml-hiring-service
```

## 🔍 Troubleshooting

### Error: Modelo no encontrado

```bash
# Entrenar modelo primero
python train_simple.py
```

### Error: Dependencias faltantes

```bash
pip install -r requirements.txt
```

### Error: TfidfVectorizer

- Ya corregido en `simple_predictor.py`
- Usa procesamiento de texto simplificado

### API no responde

```bash
# Verificar que el puerto esté libre
uvicorn app.main:app --port 8001
```

## 📈 Métricas de Rendimiento

El modelo incluye métricas estándar:

- **Accuracy**: Precisión general
- **Precision**: Precisión de predicciones positivas
- **Recall**: Cobertura de casos positivos
- **F1-Score**: Balance entre precision y recall
- **ROC-AUC**: Área bajo la curva ROC

## 🤝 Contribución

1. Fork el proyecto
2. Crear feature branch (`git checkout -b feature/nueva-feature`)
3. Commit cambios (`git commit -am 'Agregar nueva feature'`)
4. Push al branch (`git push origin feature/nueva-feature`)
5. Crear Pull Request

## 📄 Licencia

MIT License

---

## 🎉 ¡Sistema Listo!

Tu sistema de ML para predicción de contratación está completo y funcionando. Incluye:

✅ **Modelo entrenado y probado**
✅ **API GraphQL funcional**  
✅ **Predicciones en tiempo real**
✅ **Datos de demostración**
✅ **Documentación completa**

**¡Empieza probando con `python demo_complete.py`!**
