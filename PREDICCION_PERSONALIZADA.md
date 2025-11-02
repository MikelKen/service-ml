# 🚀 PREDICCIÓN CON DATOS PERSONALIZADOS

## ✨ **NUEVA FUNCIONALIDAD IMPLEMENTADA**

Ahora puedes hacer predicciones de compatibilidad **SIN necesidad** de que los datos estén en la base de datos. Solo proporcionas los datos del candidato y la oferta directamente en la consulta GraphQL.

---

## 🎯 **CONSULTA GRAPHQL COMPLETA**

### **Con tus datos exactos:**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 7
        nivelEducacion: "Ingeniería Comercial"
        habilidades: "AR/VR, Unity3D, Unreal Engine, Oculus SDK, ARKit, ARCore, Vuforia"
        idiomas: "Español (Nativo), Inglés (Avanzado)"
        certificaciones: "Machine Learning Coursera Certificate, Deep Learning Specialization"
        puestoActual: "Mobile Developer en AppFactory"
      }
      offerData: {
        titulo: "Desarrollador Full Stack"
        salario: 7375.24
        ubicacion: "Santa Cruz de la Sierra"
        requisitos: "Licenciatura en Ingeniería de Sistemas, experiencia en desarrollo web"
      }
    }
  ) {
    candidateId
    offerId
    probability
    prediction
    confidence
    modelUsed
    predictionDate
    error
  }
}
```

---

## 📊 **RESULTADO OBTENIDO**

```json
{
  "data": {
    "predictCustomCompatibility": {
      "candidateId": "custom_candidate",
      "offerId": "custom_offer",
      "probability": 0.3263,
      "prediction": false,
      "confidence": "Media",
      "modelUsed": "gradient_boosting",
      "predictionDate": "2025-11-02T...",
      "error": null
    }
  }
}
```

### 🔍 **INTERPRETACIÓN:**

- **Probabilidad: 32.63%** - Compatibilidad baja-media
- **Predicción: false** - No compatible según el modelo
- **Confianza: Media** - El modelo está moderadamente seguro
- **Recomendación**: Revisar requisitos específicos

---

## 🛠️ **CAMPOS REQUERIDOS**

### **👤 Datos del Candidato (`candidateData`):**

| Campo              | Tipo     | Requerido | Descripción                                  |
| ------------------ | -------- | --------- | -------------------------------------------- |
| `aniosExperiencia` | `int`    | ✅        | Años de experiencia laboral                  |
| `nivelEducacion`   | `string` | ✅        | Nivel educativo (ej: "Ingeniería Comercial") |
| `habilidades`      | `string` | ✅        | Skills técnicos separados por comas          |
| `idiomas`          | `string` | ❌        | Idiomas que maneja                           |
| `certificaciones`  | `string` | ❌        | Certificaciones obtenidas                    |
| `puestoActual`     | `string` | ❌        | Posición laboral actual                      |

### **💼 Datos de la Oferta (`offerData`):**

| Campo        | Tipo     | Requerido | Descripción           |
| ------------ | -------- | --------- | --------------------- |
| `titulo`     | `string` | ✅        | Título del puesto     |
| `salario`    | `float`  | ✅        | Salario ofrecido      |
| `ubicacion`  | `string` | ✅        | Ubicación del trabajo |
| `requisitos` | `string` | ✅        | Requisitos del puesto |

---

## 🚀 **CASOS DE USO**

### **1. Evaluación Rápida de CV**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 5
        nivelEducacion: "Ingeniería de Sistemas"
        habilidades: "Python, Django, PostgreSQL, React"
      }
      offerData: {
        titulo: "Desarrollador Python"
        salario: 8000.0
        ubicacion: "La Paz"
        requisitos: "Python, Django, 3+ años experiencia"
      }
    }
  ) {
    probability
    prediction
    confidence
  }
}
```

### **2. Análisis 'What-If'**

```graphql
# ¿Qué pasa si el candidato tuviera más experiencia?
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 10 # Cambiar experiencia
        nivelEducacion: "Ingeniería Comercial"
        habilidades: "AR/VR, Unity3D, Unreal Engine"
      }
      offerData: {
        titulo: "Desarrollador Full Stack"
        salario: 7375.24
        ubicacion: "Santa Cruz de la Sierra"
        requisitos: "Experiencia en desarrollo web"
      }
    }
  ) {
    probability
    prediction
  }
}
```

### **3. Comparación de Candidatos**

```graphql
# Candidato A vs Candidato B para la misma oferta
query {
  candidatoA: predictCustomCompatibility(
    input: {
      candidateData: { aniosExperiencia: 7, nivelEducacion: "Ingeniería Comercial", habilidades: "AR/VR, Unity3D" }
      offerData: {
        titulo: "Desarrollador Full Stack"
        salario: 7375.24
        ubicacion: "Santa Cruz"
        requisitos: "Desarrollo web"
      }
    }
  ) {
    probability
    prediction
  }

  candidatoB: predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 5
        nivelEducacion: "Ingeniería de Sistemas"
        habilidades: "JavaScript, React, Node.js"
      }
      offerData: {
        titulo: "Desarrollador Full Stack"
        salario: 7375.24
        ubicacion: "Santa Cruz"
        requisitos: "Desarrollo web"
      }
    }
  ) {
    probability
    prediction
  }
}
```

---

## 💡 **VENTAJAS DE ESTA FUNCIONALIDAD**

### ✅ **Beneficios Principales:**

1. **No requiere base de datos** - Datos directos en la consulta
2. **Evaluación instantánea** - Predicción en tiempo real
3. **Flexibilidad total** - Cualquier combinación de datos
4. **Análisis comparativo** - Múltiples escenarios fácilmente
5. **Integración simple** - Solo una consulta GraphQL

### 🎯 **Casos de Uso Ideales:**

- **Screening inicial** de CVs
- **Análisis de sensibilidad** (cambiar experiencia, skills, etc.)
- **Comparación de candidatos** para la misma posición
- **Evaluación de perfiles** antes de añadir a BD
- **Simulaciones** de compatibilidad

---

## 🔧 **CÓMO USAR AHORA**

### **1. Iniciar servidor:**

```bash
python -m uvicorn app.main:app --reload
```

### **2. Ir a GraphQL Playground:**

```
http://localhost:8000/graphql
```

### **3. Copiar consulta de ejemplo:**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: { aniosExperiencia: 7, nivelEducacion: "TU_EDUCACION", habilidades: "TUS_SKILLS" }
      offerData: { titulo: "TITULO_PUESTO", salario: 5000.0, ubicacion: "TU_CIUDAD", requisitos: "REQUISITOS_PUESTO" }
    }
  ) {
    probability
    prediction
    confidence
    modelUsed
  }
}
```

### **4. ¡Obtener predicción instantánea!** 🎉

---

## 🔄 **AMBAS FUNCIONALIDADES DISPONIBLES**

### **📊 Con datos de BD (existente):**

```graphql
query {
  predictCompatibility(input: { candidateId: "ID_EN_BD", offerId: "ID_EN_BD" }) {
    probability
    prediction
  }
}
```

### **⚡ Con datos personalizados (nuevo):**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: { /* datos directos */ }
      offerData: { /* datos directos */ }
    }
  ) { probability prediction }
}
```

---

## 🎯 **PRÓXIMOS PASOS RECOMENDADOS**

1. **Probar con diferentes perfiles** y ofertas
2. **Analizar qué factores** influyen más en la predicción
3. **Comparar candidatos** para la misma posición
4. **Simular mejoras** en perfiles (más experiencia, skills, etc.)
5. **Integrar en tu aplicación** para screening automático

**¡Ya puedes hacer predicciones con cualquier dato sin necesidad de base de datos!** 🚀✨
