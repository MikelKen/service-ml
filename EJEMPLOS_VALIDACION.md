# 🧪 EJEMPLOS PARA VALIDAR EL MODELO DE CONTRATACIÓN

## 📋 Instrucciones:

1. Abre GraphQL en: http://127.0.0.1:8000/graphql
2. Copia cada mutación y ejecutala
3. Evalúa si el resultado tiene sentido
4. Anota tus observaciones

---

## 🟢 **CATEGORÍA 1: MUY ALTA PROBABILIDAD (Esperado: 80%+)**

_Candidatos que definitivamente deberían ser contactados_

### Ejemplo 1A: Ana García - Perfect Match

```graphql
mutation {
  predictHiring(
    nombre: "Ana García - Perfect Match"
    anosExperiencia: 5
    nivelEducacion: "maestría"
    habilidades: "python, machine learning, sql, tensorflow, pandas"
    idiomas: "español, inglés"
    certificaciones: "aws certified developer"
    titulo: "Data Scientist"
    requisitos: "python, machine learning, sql"
    salario: 18000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser alta:** Experiencia óptima (5 años), skills perfectos, salario razonable

### Ejemplo 1B: Carlos López - Senior Ideal

```graphql
mutation {
  predictHiring(
    nombre: "Carlos López - Senior Ideal"
    anosExperiencia: 7
    nivelEducacion: "licenciatura"
    habilidades: "javascript, react, node.js, typescript, aws"
    idiomas: "español, inglés"
    certificaciones: "aws solutions architect"
    titulo: "Senior Full Stack Developer"
    requisitos: "javascript, react, node.js"
    salario: 22000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser alta:** 7 años de experiencia, match perfecto de skills

---

## 🟡 **CATEGORÍA 2: ALTA PROBABILIDAD (Esperado: 60-80%)**

_Buenos candidatos que probablemente deberían ser contactados_

### Ejemplo 2A: María Rodríguez - Junior Promisorio

```graphql
mutation {
  predictHiring(
    nombre: "María Rodríguez - Junior Promisorio"
    anosExperiencia: 2
    nivelEducacion: "licenciatura"
    habilidades: "python, sql, pandas, numpy"
    idiomas: "español, inglés básico"
    certificaciones: "google analytics"
    titulo: "Junior Data Analyst"
    requisitos: "python, sql, excel"
    salario: 12000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser alta:** Junior con buena base técnica y salario apropiado

### Ejemplo 2B: Diego Fernández - Mid Level

```graphql
mutation {
  predictHiring(
    nombre: "Diego Fernández - Mid Level"
    anosExperiencia: 4
    nivelEducacion: "licenciatura"
    habilidades: "java, spring boot, mysql, docker"
    idiomas: "español, inglés"
    certificaciones: "oracle certified java programmer"
    titulo: "Backend Developer"
    requisitos: "java, spring, mysql"
    salario: 16000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser alta:** Experiencia sólida con tecnologías requeridas

---

## 🟠 **CATEGORÍA 3: PROBABILIDAD MEDIA (Esperado: 40-60%)**

_Candidatos que requieren consideración cuidadosa_

### Ejemplo 3A: Luis Torres - Skills Parciales

```graphql
mutation {
  predictHiring(
    nombre: "Luis Torres - Skills Parciales"
    anosExperiencia: 3
    nivelEducacion: "técnico"
    habilidades: "php, mysql, html, css"
    idiomas: "español"
    certificaciones: "sin certificacion"
    titulo: "Full Stack Developer"
    requisitos: "javascript, react, node.js, mongodb"
    salario: 14000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser media:** Experiencia decente pero skills no coinciden bien

### Ejemplo 3B: Carmen Silva - Cambio de Carrera

```graphql
mutation {
  predictHiring(
    nombre: "Carmen Silva - Cambio de Carrera"
    anosExperiencia: 1
    nivelEducacion: "maestría"
    habilidades: "python, estadística, excel, r"
    idiomas: "español, inglés, portugués"
    certificaciones: "coursera data science"
    titulo: "Data Scientist"
    requisitos: "python, machine learning, sql, tensorflow"
    salario: 15000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser media:** Alta educación pero poca experiencia práctica

---

## 🔴 **CATEGORÍA 4: BAJA PROBABILIDAD (Esperado: 20-40%)**

_Candidatos probablemente no recomendados_

### Ejemplo 4A: Roberto Méndez - Sobrecalificado/Caro

```graphql
mutation {
  predictHiring(
    nombre: "Roberto Méndez - Sobrecalificado/Caro"
    anosExperiencia: 15
    nivelEducacion: "doctorado"
    habilidades: "python, machine learning, deep learning, scala, spark"
    idiomas: "español, inglés, alemán"
    certificaciones: "aws certified solutions architect professional"
    titulo: "Junior Data Scientist"
    requisitos: "python, sql, excel"
    salario: 45000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser baja:** Demasiado senior y caro para puesto junior

### Ejemplo 4B: Sandra Vega - Sin Skills Relevantes

```graphql
mutation {
  predictHiring(
    nombre: "Sandra Vega - Sin Skills Relevantes"
    anosExperiencia: 8
    nivelEducacion: "licenciatura"
    habilidades: "marketing, photoshop, illustrator, social media"
    idiomas: "español, francés"
    certificaciones: "google ads certified"
    titulo: "Software Developer"
    requisitos: "java, spring boot, angular, postgresql"
    salario: 20000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser baja:** Skills completamente diferentes al puesto

---

## ⚫ **CATEGORÍA 5: MUY BAJA PROBABILIDAD (Esperado: <20%)**

_Candidatos que definitivamente NO deberían ser contactados_

### Ejemplo 5A: El Super Candidato - Irreal

```graphql
mutation {
  predictHiring(
    nombre: "El Super Candidato - Irreal"
    anosExperiencia: 100
    nivelEducacion: "doctorado"
    habilidades: "python, machine learning, sql, tensorflow, aws, kubernetes"
    idiomas: "español, inglés, francés"
    certificaciones: "aws solutions architect, google cloud professional"
    titulo: "Senior Data Scientist"
    requisitos: "python, machine learning, sql"
    salario: 25000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser muy baja:** Experiencia irreal - 100 años es imposible

### Ejemplo 5B: Juan Pérez - Sin Experiencia ni Skills

```graphql
mutation {
  predictHiring(
    nombre: "Juan Pérez - Sin Experiencia ni Skills"
    anosExperiencia: 0
    nivelEducacion: "técnico"
    habilidades: "word, excel básico"
    idiomas: "español"
    certificaciones: "sin certificacion"
    titulo: "Senior Software Architect"
    requisitos: "microservices, kubernetes, docker, aws, terraform"
    salario: 8000
  ) {
    prediction
    probability
    confidenceLevel
    recommendation
    modelUsed
  }
}
```

**Por qué debería ser muy baja:** Sin experiencia para puesto senior complejo

---

## 📊 **TABLA DE RESULTADOS**

| Ejemplo              | Categoría Esperada | Probabilidad Obtenida | ¿Correcto? | Observaciones |
| -------------------- | ------------------ | --------------------- | ---------- | ------------- |
| 1A - Ana García      | 🟢 >80%            | \_\_\_%               | ☐ Sí ☐ No  |               |
| 1B - Carlos López    | 🟢 >80%            | \_\_\_%               | ☐ Sí ☐ No  |               |
| 2A - María Rodríguez | 🟡 60-80%          | \_\_\_%               | ☐ Sí ☐ No  |               |
| 2B - Diego Fernández | 🟡 60-80%          | \_\_\_%               | ☐ Sí ☐ No  |               |
| 3A - Luis Torres     | 🟠 40-60%          | \_\_\_%               | ☐ Sí ☐ No  |               |
| 3B - Carmen Silva    | 🟠 40-60%          | \_\_\_%               | ☐ Sí ☐ No  |               |
| 4A - Roberto Méndez  | 🔴 20-40%          | \_\_\_%               | ☐ Sí ☐ No  |               |
| 4B - Sandra Vega     | 🔴 20-40%          | \_\_\_%               | ☐ Sí ☐ No  |               |
| 5A - Super Candidato | ⚫ <20%            | \_\_\_%               | ☐ Sí ☐ No  |               |
| 5B - Juan Pérez      | ⚫ <20%            | \_\_\_%               | ☐ Sí ☐ No  |               |

**Total de predicciones correctas: \_\_\_/10**

---

## 🎯 **EVALUACIÓN FINAL**

- **9-10 correctas**: 🟢 Modelo excelente
- **7-8 correctas**: 🟡 Modelo bueno
- **5-6 correctas**: 🟠 Modelo regular, necesita ajustes
- **0-4 correctas**: 🔴 Modelo malo, requiere revisión

**¿Qué te parecen los resultados? ¿Hay algún caso que no tiene sentido?**
