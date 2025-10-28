"""
Aplicación FastAPI mínima para ML de contratación
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="ML Hiring Service",
    description="Machine Learning microservice for hiring prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class HiringRequest(BaseModel):
    nombre: str
    anos_experiencia: int
    nivel_educacion: str
    habilidades: str
    idiomas: str
    certificaciones: Optional[str] = None
    puesto_actual: Optional[str] = None
    industria: Optional[str] = None
    titulo: str = "Desarrollador"
    descripcion: str = "Desarrollo de software"
    salario: float = 10000
    ubicacion: str = "Santa Cruz"
    requisitos: str = "Experiencia en desarrollo"
    fecha_postulacion: Optional[str] = None
    fecha_publicacion: Optional[str] = None

class HiringResponse(BaseModel):
    prediction: int
    probability: float
    confidence_level: str
    recommendation: str
    model_used: str

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "ML Hiring Service",
        "status": "running",
        "version": "1.0.0",
        "description": "Machine Learning microservice for hiring prediction",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    """Verificación de salud"""
    try:
        model_path = "trained_models/simple_hiring_model.pkl"
        model_loaded = os.path.exists(model_path)
        
        return {
            "status": "healthy",
            "service": "ml-hiring-service",
            "version": "1.0.0",
            "model_loaded": model_loaded,
            "message": "ML Hiring Service is running"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/predict", response_model=HiringResponse)
async def predict_hiring(request: HiringRequest):
    """Predice la probabilidad de contratación"""
    try:
        # Importar el predictor simple
        from simple_predictor import SimpleHiringPredictor
        
        model_path = "trained_models/simple_hiring_model.pkl"
        
        if not os.path.exists(model_path):
            logger.warning("Modelo no encontrado, usando predicción por defecto")
            return HiringResponse(
                prediction=1,
                probability=0.75,
                confidence_level="Alta",
                recommendation="Recomendado para entrevista",
                model_used="DefaultPredictor"
            )
        
        # Crear datos de entrada
        application_data = {
            'nombre': request.nombre,
            'años_experiencia': request.anos_experiencia,
            'nivel_educacion': request.nivel_educacion,
            'habilidades': request.habilidades,
            'idiomas': request.idiomas,
            'certificaciones': request.certificaciones or "",
            'puesto_actual': request.puesto_actual or "Desarrollador",
            'industria': request.industria or "Tecnología",
            'titulo': request.titulo,
            'descripcion': request.descripcion,
            'salario': request.salario,
            'ubicacion': request.ubicacion,
            'requisitos': request.requisitos,
            'fecha_postulacion': request.fecha_postulacion or '2024-01-15',
            'fecha_publicacion': request.fecha_publicacion or '2024-01-10'
        }
        
        # Realizar predicción
        predictor = SimpleHiringPredictor(model_path)
        result = predictor.predict(application_data)
        
        return HiringResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            confidence_level=result['confidence_level'],
            recommendation=result['recommendation'],
            model_used=result['model_used']
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en predicción: {str(e)}"
        )

@app.get("/info")
async def get_service_info():
    """Información del servicio"""
    return {
        "name": "ML Hiring Service",
        "description": "Machine Learning microservice for hiring prediction",
        "version": "1.0.0",
        "features": [
            "Hiring probability prediction",
            "Candidate evaluation",
            "REST API"
        ],
        "technologies": [
            "FastAPI",
            "scikit-learn",
            "pandas",
            "numpy"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)