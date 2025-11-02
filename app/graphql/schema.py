"""Esquema GraphQL central de la aplicación.

Incluye consultas para leer las colecciones ERP en MongoDB y operaciones ML.
"""
import strawberry
from typing import List, Optional

from app.graphql.types.erp_types import (
    EmpresaFeature, OfertaFeature, PostulanteFeature,
    EmpresaFilter, OfertaFilter, PostulanteFilter,
)
from app.graphql.types.ml_types import (
    CompatibilityPredictionInput, BatchCompatibilityInput, TopCandidatesInput,
    CompatibilityPrediction, BatchCompatibilityResult, ModelInfo,
    ModelFeatureImportance, PredictionExplanation, TrainingDataSummary,
    ModelPerformanceMetrics
)
from app.graphql.types.feature_types import (
    CandidateFeature, JobOfferFeature, CompanyFeature,
    CandidateFeatureList, JobOfferFeatureList, CompanyFeatureList,
    FeatureQueryInput, FeatureCollectionInfo
)
from app.graphql.resolvers import erp_resolvers
from app.graphql.resolvers import ml_resolvers
from app.graphql.resolvers import feature_resolvers
from app.graphql.mutations.ml_mutations import MLMutation


@strawberry.type
class Query:
    # Consultas ERP existentes
    @strawberry.field(description="Empresas desde MongoDB")
    async def empresas(self, filter: Optional[EmpresaFilter] = None, limit: Optional[int] = 100) -> List[EmpresaFeature]:
        data = await erp_resolvers.get_empresas(filter.__dict__ if filter else None, limit)
        return [EmpresaFeature(**d) for d in data]

    @strawberry.field(description="Ofertas de trabajo desde MongoDB")
    async def ofertas(self, filter: Optional[OfertaFilter] = None, limit: Optional[int] = 100) -> List[OfertaFeature]:
        data = await erp_resolvers.get_ofertas(filter.__dict__ if filter else None, limit)
        return [OfertaFeature(**d) for d in data]

    @strawberry.field(description="Postulantes (features) desde MongoDB")
    async def postulantes(self, filter: Optional[PostulanteFilter] = None, limit: Optional[int] = 100) -> List[PostulanteFeature]:
        data = await erp_resolvers.get_postulantes(filter.__dict__ if filter else None, limit)
        return [PostulanteFeature(**d) for d in data]
    
    # Nuevas consultas ML
    @strawberry.field(description="Predice compatibilidad entre candidato y oferta")
    async def predict_compatibility(self, input: CompatibilityPredictionInput) -> CompatibilityPrediction:
        return await ml_resolvers.predict_compatibility(input)
    
    @strawberry.field(description="Predice compatibilidad para múltiples pares candidato-oferta")
    async def predict_batch_compatibility(self, input: BatchCompatibilityInput) -> BatchCompatibilityResult:
        return await ml_resolvers.predict_batch_compatibility(input)
    
    @strawberry.field(description="Obtiene los mejores candidatos para una oferta")
    async def get_top_candidates_for_offer(self, input: TopCandidatesInput) -> List[CompatibilityPrediction]:
        return await ml_resolvers.get_top_candidates_for_offer(input)
    
    @strawberry.field(description="Información del modelo ML actual")
    async def model_info(self) -> ModelInfo:
        return await ml_resolvers.get_model_info()
    
    @strawberry.field(description="Importancia de features del modelo")
    async def feature_importance(self, top_n: Optional[int] = 20) -> ModelFeatureImportance:
        return await ml_resolvers.get_feature_importance(top_n)
    
    @strawberry.field(description="Explicación detallada de una predicción")
    async def explain_prediction(self, candidate_id: str, offer_id: str) -> PredictionExplanation:
        return await ml_resolvers.explain_prediction(candidate_id, offer_id)
    
    @strawberry.field(description="Resumen de datos de entrenamiento")
    async def training_data_summary(self) -> TrainingDataSummary:
        return await ml_resolvers.get_training_data_summary()
    
    @strawberry.field(description="Métricas de rendimiento del modelo")
    async def model_performance(self) -> ModelPerformanceMetrics:
        return await ml_resolvers.get_model_performance()
    
    @strawberry.field(description="Verifica si el modelo está cargado")
    async def is_model_loaded(self) -> bool:
        return await ml_resolvers.is_model_loaded()
    
    @strawberry.field(description="Estado completo del sistema ML")
    async def model_status(self) -> str:
        """Retorna estado del sistema como JSON string"""
        import json
        status = await ml_resolvers.get_model_status()
        return json.dumps(status, indent=2)
    
    # ==========================================
    # NUEVAS CONSULTAS DE FEATURES MONGODB
    # ==========================================
    
    @strawberry.field(description="Lista de candidatos con características desde MongoDB")
    async def candidates_features(self, query: Optional[FeatureQueryInput] = None) -> CandidateFeatureList:
        return await feature_resolvers.get_candidates_features(query)
    
    @strawberry.field(description="Lista de ofertas laborales con características desde MongoDB")
    async def job_offers_features(self, query: Optional[FeatureQueryInput] = None) -> JobOfferFeatureList:
        return await feature_resolvers.get_job_offers_features(query)
    
    @strawberry.field(description="Lista de empresas con características desde MongoDB")
    async def companies_features(self, query: Optional[FeatureQueryInput] = None) -> CompanyFeatureList:
        return await feature_resolvers.get_companies_features(query)
    
    @strawberry.field(description="Obtiene un candidato específico por ID")
    async def candidate_by_id(self, candidate_id: str) -> Optional[CandidateFeature]:
        return await feature_resolvers.get_candidate_by_id(candidate_id)
    
    @strawberry.field(description="Obtiene una oferta específica por ID")
    async def job_offer_by_id(self, offer_id: str) -> Optional[JobOfferFeature]:
        return await feature_resolvers.get_job_offer_by_id(offer_id)
    
    @strawberry.field(description="Obtiene una empresa específica por ID")
    async def company_by_id(self, company_id: str) -> Optional[CompanyFeature]:
        return await feature_resolvers.get_company_by_id(company_id)
    
    @strawberry.field(description="Información sobre una colección de features")
    async def collection_info(self, collection_name: str) -> FeatureCollectionInfo:
        return await feature_resolvers.get_collection_info(collection_name)
    
    @strawberry.field(description="Candidatos que aplicaron a una oferta específica")
    async def candidates_by_offer(self, offer_id: str, limit: Optional[int] = 10) -> List[CandidateFeature]:
        return await feature_resolvers.get_candidates_by_offer_id(offer_id, limit or 10)
    
    @strawberry.field(description="Ofertas de una empresa específica")
    async def offers_by_company(self, company_id: str, limit: Optional[int] = 10) -> List[JobOfferFeature]:
        return await feature_resolvers.get_offers_by_company_id(company_id, limit or 10)


@strawberry.type
class Mutation:
    """Mutaciones disponibles en el sistema"""
    
    # Incluir mutaciones ML
    ml: MLMutation = strawberry.field(resolver=lambda: MLMutation())


schema = strawberry.Schema(query=Query, mutation=Mutation)
