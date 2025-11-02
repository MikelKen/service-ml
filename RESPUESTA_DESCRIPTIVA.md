# 🎯 CONSULTA GRAPHQL DESCRIPTIVA COMPLETA

## ✨ **NUEVA FUNCIONALIDAD: RESPUESTA SUPER DESCRIPTIVA**

Tu consulta ahora puede obtener un análisis completo y detallado con recomendaciones específicas.

---

## 🔥 **CONSULTA COMPLETA CON ANÁLISIS DETALLADO:**

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
    # Información básica
    probability
    prediction
    confidence
    probabilityPercentage
    compatibilityLevel

    # Análisis ejecutivo
    summary
    recommendation

    # Análisis detallado
    strengths
    weaknesses
    suggestions
    decisionFactors
    detailedAnalysis

    # Información técnica
    modelUsed
    confidenceScore
    predictionDate
  }
}
```

---

## 📊 **CAMPOS DISPONIBLES EN LA RESPUESTA:**

### **📈 Información Básica:**

- `probability`: Probabilidad numérica (0.0 - 1.0)
- `probabilityPercentage`: Porcentaje formateado (ej: "32.63%")
- `prediction`: true/false (compatible/no compatible)
- `compatibilityLevel`: Nivel visual (🟢 ALTA, 🟡 MODERADA, 🟠 BAJA-MEDIA, 🔴 BAJA)
- `confidence`: Nivel de confianza del modelo

### **🎯 Análisis Ejecutivo:**

- `summary`: Resumen ejecutivo completo
- `recommendation`: Recomendación específica con porcentaje

### **🔍 Análisis Detallado:**

- `strengths`: Array de fortalezas del candidato
- `weaknesses`: Array de áreas de mejora identificadas
- `suggestions`: Array de sugerencias específicas para mejorar
- `decisionFactors`: Factores clave que influyeron en la decisión
- `detailedAnalysis`: Análisis completo con perfiles y resultados

### **🛠️ Información Técnica:**

- `modelUsed`: Modelo ML utilizado
- `confidenceScore`: Score numérico de confianza
- `predictionDate`: Timestamp de la predicción

---

## 📋 **EJEMPLO DE RESPUESTA DESCRIPTIVA REAL:**

````json
{
  "data": {
    "predictCustomCompatibility": {
      "probability": 0.3263407382747913,
      "prediction": false,
      "confidence": "Media",
      "probabilityPercentage": "32.63%",
      "compatibilityLevel": "🟠 COMPATIBILIDAD BAJA-MEDIA",

      "summary": "🎯 RESUMEN EJECUTIVO:\nCandidato con 7 años de experiencia en Mobile Developer en AppFactory, \nformación en Ingeniería Comercial, presenta 32.6% de compatibilidad \npara el puesto de Desarrollador Full Stack. Requiere evaluación basado en análisis de ML.",

      "recommendation": "⚠️ EVALUACIÓN REQUERIDA: 32.6% de compatibilidad sugiere revisar requisitos específicos antes de descartar.",

      "strengths": [
        "💼 Experiencia sólida: 7 años en el campo",
        "🛠️ Skills técnicos: Unity3D, AR/VR",
        "🏆 Certificaciones: Machine Learning Coursera Certificate, Deep Learni...",
        "🌍 Manejo de inglés (ventaja competitiva)"
      ],

      "weaknesses": [
        "📚 Educación en área diferente (Comercial vs Técnica)",
        "🎯 Especialización muy específica (AR/VR) para puesto generalista"
      ],

      "suggestions": [
        "📈 Desarrollar skills en tecnologías web (HTML, CSS, JavaScript)",
        "🎓 Considerar certificaciones en desarrollo Full Stack",
        "💼 Buscar experiencia práctica en proyectos web",
        "🔧 Complementar formación con bootcamp técnico",
        "🛠️ Ampliar portfolio de tecnologías"
      ],

      "decisionFactors": "📊 FACTORES CLAVE DE LA PREDICCIÓN:\n• Experiencia: 7 años (✅ Adecuada)\n• Educación: Ingeniería Comercial (⚠️ No técnica)\n• Skills: 2 tecnologías identificadas (⚠️ Limitadas)\n• Especialización: 🎯 Muy específica\n• Match puesto: ⚠️ Medio",

      "detailedAnalysis": "📋 ANÁLISIS DETALLADO DE COMPATIBILIDAD:\n\n🔍 PERFIL DEL CANDIDATO:\n• Experiencia: 7 años como Mobile Developer en AppFactory\n• Educación: Ingeniería Comercial\n• Tecnologías: AR/VR, Unity3D, Unreal Engine, Oculus SDK, ARKit, ARCore, Vuforia\n• Idiomas: Español (Nativo), Inglés (Avanzado)\n\n💼 PERFIL DE LA OFERTA:\n• Posición: Desarrollador Full Stack\n• Salario: $7,375.24\n• Ubicación: Santa Cruz de la Sierra\n• Requisitos: Licenciatura en Ingeniería de Sistemas, experiencia en desarrollo web\n\n🎯 RESULTADO DE COMPATIBILIDAD:\n• Probabilidad: 32.63% (🟠 COMPATIBILIDAD BAJA-MEDIA)\n• Predicción: ❌ No compatible\n• Confianza del modelo: Media\n• Modelo utilizado: Gradient Boosting\n\n📈 NIVEL DE RECOMENDACIÓN:\n⚠️ EVALUACIÓN REQUERIDA: 32.6% de compatibilidad sugiere revisar requisitos específicos antes de descartar.\n\n� FACTORES DETERMINANTES:\n📊 FACTORES CLAVE DE LA PREDICCIÓN:\n• Experiencia: 7 años (✅ Adecuada)\n• Educación: Ingeniería Comercial (⚠️ No técnica)\n• Skills: 2 tecnologías identificadas (⚠️ Limitadas)\n• Especialización: 🎯 Muy específica\n• Match puesto: ⚠️ Medio",

      "modelUsed": "gradient_boosting",
      "confidenceScore": 0.3263407382747913,
      "predictionDate": "2025-11-02T18:17:59.607043"
    }
  }
}
```---

## 🎯 **BENEFICIOS DE LA RESPUESTA DESCRIPTIVA:**

### ✅ **Para Reclutadores:**

- **Análisis instantáneo** completo del candidato
- **Recomendaciones específicas** basadas en ML
- **Factores clave** que influyen en la decisión
- **Sugerencias de mejora** para el candidato

### ✅ **Para Candidatos:**

- **Feedback detallado** sobre su perfil
- **Áreas de mejora** específicas identificadas
- **Sugerencias concretas** para aumentar compatibilidad
- **Fortalezas reconocidas** por el sistema

### ✅ **Para Empresas:**

- **Decisiones informadas** basadas en datos
- **Justificación clara** de cada recomendación
- **Análisis de riesgo** detallado
- **Optimización** del proceso de selección

---

## 🚀 **CASOS DE USO MEJORADOS:**

### **1. Evaluación Completa de CV:**

- Análisis automático de compatibilidad
- Reporte detallado con fortalezas/debilidades
- Recomendaciones específicas

### **2. Feedback para Candidatos:**

- Informe personalizado de evaluación
- Sugerencias de mejora profesional
- Identificación de skills a desarrollar

### **3. Justificación de Decisiones:**

- Documentación completa del análisis
- Factores objetivos considerados
- Transparencia en el proceso

### **4. Optimización de Perfiles:**

- Identificación de gaps específicos
- Roadmap de mejora personalizado
- Tracking de progreso

---

## 💡 **CÓMO INTERPRETAR LOS RESULTADOS:**

### **🟢 70-100% - ALTA COMPATIBILIDAD:**

- ✅ Proceder inmediatamente con entrevista
- ✅ Candidato altamente recomendado
- ✅ Alta probabilidad de éxito

### **🟡 50-69% - COMPATIBILIDAD MODERADA:**

- ⚡ Continuar con evaluación técnica
- ⚡ Buen candidato con potencial
- ⚡ Revisar skills específicos

### **🟠 30-49% - COMPATIBILIDAD BAJA-MEDIA:**

- ⚠️ Evaluar pros y contras cuidadosamente
- ⚠️ Revisar requisitos específicos
- ⚠️ Considerar entrenamiento adicional

### **🔴 0-29% - BAJA COMPATIBILIDAD:**

- ❌ No recomendado para esta posición
- ❌ Considerar solo si hay escasez
- ❌ Requiere desarrollo significativo

---

## 🛠️ **CONSULTA SIMPLIFICADA (SOLO ESENCIAL):**

Si prefieres una respuesta más concisa:

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: { /* tus datos */ }
      offerData: { /* datos oferta */ }
    }
  ) {
    probabilityPercentage
    compatibilityLevel
    recommendation
    summary
    strengths
    suggestions
  }
}
````

---

## 🎉 **¡AHORA TIENES ANÁLISIS COMPLETO Y DESCRIPTIVO!**

Tu consulta GraphQL ahora proporciona:

- ✅ **Análisis detallado** de compatibilidad
- ✅ **Recomendaciones específicas** basadas en ML
- ✅ **Fortalezas y debilidades** identificadas
- ✅ **Sugerencias de mejora** personalizadas
- ✅ **Justificación completa** de la decisión

**¡Perfecto para tomar decisiones informadas en el proceso de reclutamiento!** 🚀✨
