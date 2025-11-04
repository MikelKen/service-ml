# 🎯 EJEMPLOS DE CONSULTAS GRAPHQL - CASOS DE USO

## 📚 **GUÍA COMPLETA DE CONSULTAS PARA DIFERENTES ESCENARIOS**

---

## 🚀 **CASO 1: DESARROLLADOR SENIOR CON ALTA COMPATIBILIDAD**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 8
        nivelEducacion: "Ingeniería de Sistemas"
        habilidades: "React, Node.js, JavaScript, TypeScript, MongoDB, Express, Git, AWS"
        idiomas: "Español (Nativo), Inglés (Avanzado)"
        certificaciones: "AWS Certified Developer, React Professional Certificate"
        puestoActual: "Full Stack Developer en TechCorp"
      }
      offerData: {
        titulo: "Desarrollador Full Stack Senior"
        salario: 9500.00
        ubicacion: "Santa Cruz de la Sierra"
        requisitos: "Licenciatura en Ingeniería de Sistemas, 5+ años experiencia Full Stack"
      }
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
```

**📈 Resultado Esperado:**

- ✅ Alta compatibilidad (70-85%)
- 🟢 Recomendación: Proceder inmediatamente con entrevista
- 💪 Fortalezas: Educación alineada, experiencia relevante, stack tecnológico perfecto

---

## 🎓 **CASO 2: RECIÉN GRADUADO CON POTENCIAL**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 1
        nivelEducacion: "Ingeniería de Sistemas"
        habilidades: "Python, Django, PostgreSQL, Git, HTML, CSS, JavaScript"
        idiomas: "Español (Nativo), Inglés (Intermedio)"
        certificaciones: "Python for Everybody Specialization, Web Development Bootcamp"
        puestoActual: "Desarrollador Junior en StartupTech"
      }
      offerData: {
        titulo: "Desarrollador Backend Junior"
        salario: 4500.00
        ubicacion: "Santa Cruz de la Sierra"
        requisitos: "Recién graduado en Ingeniería de Sistemas, conocimientos en Python"
      }
    }
  ) {
    probabilityPercentage
    compatibilityLevel
    recommendation
    summary
    strengths
    weaknesses
    suggestions
    detailedAnalysis
  }
}
```

**📈 Resultado Esperado:**

- ✅ Compatibilidad moderada-alta (60-75%)
- 🟡 Recomendación: Continuar con evaluación técnica
- 💪 Fortalezas: Educación correcta, stack tecnológico alineado, potencial de crecimiento

---

## ⚠️ **CASO 3: CAMBIO DE CARRERA - COMPATIBILIDAD BAJA**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 0
        nivelEducacion: "Licenciatura en Psicología"
        habilidades: "Microsoft Office, Photoshop, Gestión de equipos"
        idiomas: "Español (Nativo)"
        certificaciones: "Certificado en Gestión de Recursos Humanos"
        puestoActual: "Coordinador de RRHH en ConsultoraXYZ"
      }
      offerData: {
        titulo: "Desarrollador Frontend"
        salario: 6000.00
        ubicacion: "Santa Cruz de la Sierra"
        requisitos: "Ingeniería de Sistemas, experiencia en desarrollo web"
      }
    }
  ) {
    probabilityPercentage
    compatibilityLevel
    recommendation
    summary
    strengths
    weaknesses
    suggestions
    detailedAnalysis
    decisionFactors
  }
}
```

**📈 Resultado Esperado:**

- ❌ Baja compatibilidad (5-20%)
- 🔴 Recomendación: No recomendado sin formación adicional
- ⚠️ Debilidades: Sin experiencia técnica, educación no relacionada

---

## 🛠️ **CASO 4: ESPECIALISTA EN MIGRACIÓN TECNOLÓGICA**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 6
        nivelEducacion: "Ingeniería Informática"
        habilidades: "Java, Spring Boot, Angular, MySQL, Docker, Kubernetes, Jenkins"
        idiomas: "Español (Nativo), Inglés (Avanzado), Portugués (Básico)"
        certificaciones: "Oracle Java Certified, Angular Certified Developer, Docker Certified"
        puestoActual: "Java Developer en FinTech Solutions"
      }
      offerData: {
        titulo: "Desarrollador Full Stack"
        salario: 8200.00
        ubicacion: "Santa Cruz de la Sierra"
        requisitos: "Experiencia en desarrollo web moderno, conocimientos en microservicios"
      }
    }
  ) {
    probability
    prediction
    confidence
    probabilityPercentage
    compatibilityLevel
    summary
    recommendation
    strengths
    weaknesses
    suggestions
    decisionFactors
    detailedAnalysis
    modelUsed
    confidenceScore
    predictionDate
  }
}
```

**📈 Resultado Esperado:**

- ✅ Alta compatibilidad (75-90%)
- 🟢 Recomendación: Candidato altamente recomendado
- 💪 Fortalezas: Stack completo, experiencia sólida, certificaciones relevantes

---

## 🎯 **CASO 5: CONSULTA SIMPLIFICADA PARA SCREENING RÁPIDO**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 3
        nivelEducacion: "Ingeniería de Sistemas"
        habilidades: "PHP, Laravel, MySQL, Bootstrap"
        idiomas: "Español (Nativo)"
        certificaciones: "Laravel Certified"
        puestoActual: "Web Developer"
      }
      offerData: {
        titulo: "Desarrollador PHP"
        salario: 5500.00
        ubicacion: "Santa Cruz de la Sierra"
        requisitos: "Experiencia en PHP y frameworks"
      }
    }
  ) {
    probabilityPercentage
    compatibilityLevel
    recommendation
  }
}
```

**📈 Resultado Esperado:**

- ✅ Alta compatibilidad (80-95%)
- 🟢 Recomendación: Proceder con entrevista técnica
- ✨ Match perfecto de tecnologías

---

## 📊 **CASO 6: ANÁLISIS COMPLETO PARA DECISIÓN EJECUTIVA**

```graphql
query {
  predictCustomCompatibility(
    input: {
      candidateData: {
        aniosExperiencia: 5
        nivelEducacion: "Técnico Superior en Programación"
        habilidades: "C#, .NET Core, SQL Server, Azure, Git, Scrum"
        idiomas: "Español (Nativo), Inglés (Intermedio)"
        certificaciones: "Microsoft Azure Fundamentals, Scrum Master"
        puestoActual: ".NET Developer en SoftwareHouse"
      }
      offerData: {
        titulo: "Desarrollador .NET Senior"
        salario: 7800.00
        ubicacion: "Santa Cruz de la Sierra"
        requisitos: "Licenciatura preferible, 3+ años experiencia .NET, conocimientos Azure"
      }
    }
  ) {
    # Datos completos para reporte ejecutivo
    probability
    prediction
    confidence
    probabilityPercentage
    compatibilityLevel
    summary
    recommendation
    strengths
    weaknesses
    suggestions
    decisionFactors
    detailedAnalysis
    modelUsed
    confidenceScore
    predictionDate
  }
}
```

**📈 Resultado Esperado:**

- ⚠️ Compatibilidad media-alta (65-75%)
- 🟡 Recomendación: Evaluar pros y contras
- ⚠️ Consideración: Educación técnica vs licenciatura requerida

---

## 🎨 **GUÍA DE INTERPRETACIÓN RÁPIDA:**

### **🟢 ALTA (70-100%):**

```
compatibilityLevel: "🟢 COMPATIBILIDAD ALTA"
recommendation: "✅ PROCEDER: Candidato altamente recomendado..."
```

### **🟡 MODERADA (50-69%):**

```
compatibilityLevel: "🟡 COMPATIBILIDAD MODERADA"
recommendation: "⚡ CONTINUAR: Buen candidato con potencial..."
```

### **🟠 BAJA-MEDIA (30-49%):**

```
compatibilityLevel: "🟠 COMPATIBILIDAD BAJA-MEDIA"
recommendation: "⚠️ EVALUACIÓN REQUERIDA: Revisar requisitos..."
```

### **🔴 BAJA (0-29%):**

```
compatibilityLevel: "🔴 COMPATIBILIDAD BAJA"
recommendation: "❌ NO RECOMENDADO: Requiere desarrollo..."
```

---

## 🛠️ **TIPS PARA OPTIMIZAR CONSULTAS:**

### **1. Para Screening Masivo:**

- Usar solo: `probabilityPercentage`, `compatibilityLevel`, `recommendation`
- Respuesta rápida para filtrado inicial

### **2. Para Análisis Detallado:**

- Incluir: `strengths`, `weaknesses`, `suggestions`, `detailedAnalysis`
- Perfecto para entrevistas y feedback

### **3. Para Reportes Ejecutivos:**

- Consulta completa con todos los campos
- Documentación completa de decisiones

### **4. Para Feedback a Candidatos:**

- Enfocarse en: `summary`, `strengths`, `suggestions`
- Información constructiva y profesional

---

## 🚀 **¡EXPLORA DIFERENTES COMBINACIONES!**

Puedes mezclar y combinar campos según tus necesidades:

- **Reclutamiento ágil:** Solo campos esenciales
- **Análisis profundo:** Campos descriptivos completos
- **Documentación:** Todos los campos técnicos
- **Feedback:** Campos orientados a mejora

## 🎯 **PRÓXIMOS PASOS RECOMENDADOS**

1. **Probar con diferentes candidatos/ofertas**:

   ```graphql
   # Usar otros IDs de tu base de datos
   query {
     predictCompatibility(input: { candidateId: "OTRO_ID", offerId: "OTRA_OFERTA" }) {
       probability
       prediction
       confidence
     }
   }
   ```

2. **Encontrar mejores matches**:

   ```graphql
   query {
     getTopCandidatesForOffer(input: { offerId: "TU_OFERTA_ID", topN: 10 }) {
       candidateId
       probability
       ranking
     }
   }
   ```

3. **Analizar factores de decisión**:
   ```graphql
   query {
     explainPrediction(candidateId: "ID", offerId: "ID") {
       recommendation
       keyFactors {
         skillsOverlap
         experienceMatch
       }
     }
   }
   ```

---
