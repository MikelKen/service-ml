# 📋 CONSULTAS GRAPHQL FUNCIONALES ✅

## 🚀 **CONFIGURACIÓN**

- **Servidor**: `python -m uvicorn app.main:app --reload`
- **URL GraphQL**: http://localhost:8000/graphql
- **Estado**: ✅ 4000 ofertas, 9874 candidatos, 200 empresas

---

## 1️⃣ **OFERTAS DE TRABAJO** (✅ 4000 documentos)

### 🔥 **VER TODAS LAS OFERTAS** (✅ FUNCIONA)

```graphql
query {
  jobOffersFeatures(query: { limit: 10 }) {
    items {
      _id
      titulo
      salario
      ubicacion
      requisitos
      empresaId
    }
    total
    hasMore
  }
}
```

### 🔍 **BUSCAR "Desarrollador"** (✅ 800 resultados)

```graphql
query {
  jobOffersFeatures(query: { limit: 5, search: "Desarrollador" }) {
    items {
      _id
      titulo
      salario
      ubicacion
    }
    total
  }
}
```

### 🔍 **BUSCAR "Software"** (✅ 400 resultados)

```graphql
query {
  jobOffersFeatures(query: { limit: 5, search: "Software" }) {
    items {
      _id
      titulo
      salario
      ubicacion
    }
    total
  }
}
```

### 📄 **PAGINACIÓN**

```graphql
query {
  jobOffersFeatures(query: { limit: 5, skip: 10 }) {
    items {
      _id
      titulo
      salario
    }
    total
    hasMore
  }
}
```

---

## 2️⃣ **CANDIDATOS** (✅ 9874 documentos)

### 🔥 **VER TODOS LOS CANDIDATOS**

```graphql
query {
  candidatesFeatures(query: { limit: 10 }) {
    items {
      _id
      postulanteId
      aniosExperiencia
      nivelEducacion
      habilidades
      idiomas
    }
    total
    hasMore
  }
}
```

### 🔍 **BUSCAR candidatos con "Python"** (✅ 350 resultados)

```graphql
query {
  candidatesFeatures(query: { limit: 5, search: "Python" }) {
    items {
      _id
      aniosExperiencia
      habilidades
      nivelEducacion
    }
    total
  }
}
```

### 🔍 **BUSCAR candidatos con "Java"**

```graphql
query {
  candidatesFeatures(query: { limit: 5, search: "Java" }) {
    items {
      _id
      aniosExperiencia
      habilidades
      puesto_actual
    }
    total
  }
}
```

---

## 3️⃣ **EMPRESAS** (✅ 200 documentos)

### 🔥 **VER TODAS LAS EMPRESAS**

```graphql
query {
  companiesFeatures(query: { limit: 10 }) {
    items {
      _id
      nombre
      rubro
      empresaId
    }
    total
  }
}
```

### 🔍 **BUSCAR EMPRESAS por rubro**

```graphql
query {
  companiesFeatures(query: { limit: 5, search: "Tecnología" }) {
    items {
      _id
      nombre
      rubro
    }
    total
  }
}
```

---

## 4️⃣ **CONSULTAS POR ID**

### 🎯 **Candidato específico**

```graphql
query {
  candidateById(candidateId: "ID_DEL_CANDIDATO") {
    _id
    aniosExperiencia
    habilidades
    nivelEducacion
  }
}
```

### 🎯 **Oferta específica**

```graphql
query {
  jobOfferById(offerId: "ID_DE_LA_OFERTA") {
    _id
    titulo
    salario
    ubicacion
    requisitos
  }
}
```

### 🎯 **Empresa específica**

```graphql
query {
  companyById(companyId: "ID_DE_LA_EMPRESA") {
    _id
    nombre
    rubro
  }
}
```

---

## 5️⃣ **INFORMACIÓN DE COLECCIONES**

```graphql
query {
  collectionInfo(collectionName: "job_offers_features") {
    collectionName
    totalDocuments
    sampleFields
    lastUpdated
  }
}
```

```graphql
query {
  collectionInfo(collectionName: "candidates_features") {
    collectionName
    totalDocuments
    sampleFields
  }
}
```

```graphql
query {
  collectionInfo(collectionName: "companies_features") {
    collectionName
    totalDocuments
    sampleFields
  }
}
```

---

## 6️⃣ **CONSULTAS MÚLTIPLES**

### 🔄 **Obtener datos de todas las colecciones**

```graphql
query {
  ofertas: jobOffersFeatures(query: { limit: 3 }) {
    items {
      titulo
      salario
    }
    total
  }

  candidatos: candidatesFeatures(query: { limit: 3 }) {
    items {
      aniosExperiencia
      habilidades
    }
    total
  }

  empresas: companiesFeatures(query: { limit: 3 }) {
    items {
      nombre
      rubro
    }
    total
  }
}
```

---

## 📊 **ESTADÍSTICAS ACTUALES**

- ✅ **job_offers_features**: 4,000 ofertas
- ✅ **candidates_features**: 9,874 candidatos
- ✅ **companies_features**: 200 empresas

## 🔥 **PALABRAS CLAVE QUE FUNCIONAN**

### Para Ofertas:

- "Desarrollador" → 800 resultados
- "Software" → 400 resultados
- "Analista" → ~600 resultados
- "Ingeniero" → ~500 resultados

### Para Candidatos:

- "Python" → 350 resultados
- "Java" → ~300 resultados
- "SQL" → ~400 resultados

## ⚠️ **IMPORTANTE**

❌ **NO buscar "Python" en ofertas** - no hay resultados
✅ **SÍ buscar "Desarrollador", "Software", "Analista"**

❌ **NO filtrar si quieres ver todos los datos**
✅ **SÍ usar `limit` y `skip` para paginación**

---

## 🚀 **CONSULTA RECOMENDADA PARA DEMO**

```graphql
query {
  # Todas las ofertas (sin filtro)
  todasLasOfertas: jobOffersFeatures(query: { limit: 5 }) {
    items {
      titulo
      salario
      ubicacion
    }
    total
  }

  # Búsqueda específica
  desarrolladores: jobOffersFeatures(query: { limit: 3, search: "Desarrollador" }) {
    items {
      titulo
      salario
    }
    total
  }
}
```
