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
    CustomCompatibilityPredictionInput,
    CompatibilityPrediction, BatchCompatibilityResult, ModelInfo,
    ModelFeatureImportance, PredictionExplanation, TrainingDataSummary,
    ModelPerformanceMetrics,
    # Tipos para modelo semi-supervisado
    SemiSupervisedTrainingInput, SemiSupervisedTrainingResult,
    SemiSupervisedModelInfo, SemiSupervisedDataSummary,
    PostulacionEstadoPredictionInput, PostulacionEstadoPrediction,
    BatchEstadoPredictionInput, BatchEstadoPredictionResult,
    UnlabeledDataInsights, RetrainModelInput, RetrainModelResult
)
from app.graphql.types.feature_types import (
    CandidateFeature, JobOfferFeature, CompanyFeature,
    CandidateFeatureList, JobOfferFeatureList, CompanyFeatureList,
    FeatureQueryInput, FeatureCollectionInfo
)
from app.graphql.types.clustering_types import (
    ClusterAnalysis, ClusterProfile, SimilarCandidates,
    ClusteringQueryInput, SimilarCandidatesInput, ClusterProfileInput
)
from app.graphql.resolvers import erp_resolvers
from app.graphql.resolvers import ml_resolvers
from app.graphql.resolvers import feature_resolvers
from app.graphql.resolvers.clustering_resolvers import clustering_resolver
from app.graphql.resolvers.semi_supervised_resolvers import semi_supervised_resolver
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
    
    # Alias camelCase para compatibilidad
    @strawberry.field(description="Predice compatibilidad entre candidato y oferta (camelCase alias)", name="predictCompatibility")
    async def predict_compatibility_camel(self, input: CompatibilityPredictionInput) -> CompatibilityPrediction:
        return await ml_resolvers.predict_compatibility(input)
    
    @strawberry.field(description="Predice compatibilidad con datos personalizados (sin BD)")
    async def predict_custom_compatibility(self, input: CustomCompatibilityPredictionInput) -> CompatibilityPrediction:
        return await ml_resolvers.predict_custom_compatibility(input)
    
    # Alias camelCase para predicción personalizada
    @strawberry.field(description="Predice compatibilidad con datos personalizados (camelCase alias)", name="predictCustomCompatibility")
    async def predict_custom_compatibility_camel(self, input: CustomCompatibilityPredictionInput) -> CompatibilityPrediction:
        return await ml_resolvers.predict_custom_compatibility(input)
    
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
    
    # ==========================================
    # CONSULTAS DE CLUSTERING NO SUPERVISADO
    # ==========================================
    
    @strawberry.field(description="Análisis completo de clustering de candidatos")
    async def analyze_candidate_clusters(self, input: Optional[ClusteringQueryInput] = None) -> ClusterAnalysis:
        return await clustering_resolver.analyze_candidate_clusters(input)
    
    @strawberry.field(description="Encuentra candidatos similares basado en clustering")
    async def find_similar_candidates(self, input: SimilarCandidatesInput) -> SimilarCandidates:
        return await clustering_resolver.find_similar_candidates(input)
    
    @strawberry.field(description="Obtiene detalles específicos de un cluster")
    async def get_cluster_profile_details(self, input: ClusterProfileInput) -> ClusterProfile:
        return await clustering_resolver.get_cluster_profile_details(input)
    
    # ==========================================
    # CONSULTAS MODELO SEMI-SUPERVISADO
    # ==========================================
    
    @strawberry.field(description="Entrena modelos semi-supervisados para predicción de estados de postulaciones")
    async def train_semi_supervised_models(self, config: Optional[SemiSupervisedTrainingInput] = None) -> SemiSupervisedTrainingResult:
        return await semi_supervised_resolver.train_semi_supervised_models(config)
    
    @strawberry.field(description="Predice el estado de una postulación usando modelo semi-supervisado")
    async def predict_postulacion_estado(self, input_data: PostulacionEstadoPredictionInput, model_type: Optional[str] = None) -> PostulacionEstadoPrediction:
        return await semi_supervised_resolver.predict_postulacion_estado(input_data, model_type)
    
    @strawberry.field(description="Predice estados de múltiples postulaciones en batch")
    async def predict_batch_estados(self, input_data: BatchEstadoPredictionInput) -> BatchEstadoPredictionResult:
        return await semi_supervised_resolver.predict_batch_estados(input_data)
    
    @strawberry.field(description="Obtiene resumen de datos para modelo semi-supervisado")
    async def get_semi_supervised_data_summary(self) -> SemiSupervisedDataSummary:
        return await semi_supervised_resolver.get_semi_supervised_data_summary()
    
    @strawberry.field(description="Obtiene información de modelos semi-supervisados entrenados")
    async def get_trained_models_info(self) -> List[SemiSupervisedModelInfo]:
        return await semi_supervised_resolver.get_trained_models_info()
    
    @strawberry.field(description="Analiza datos no etiquetados y proporciona insights")
    async def analyze_unlabeled_data(self) -> UnlabeledDataInsights:
        return await semi_supervised_resolver.analyze_unlabeled_data()
    
    # Aliases con camelCase para mejor compatibilidad
    @strawberry.field(description="Entrena modelos semi-supervisados (camelCase alias)", name="trainSemiSupervisedModels")
    async def train_semi_supervised_models_camel(self, config: Optional[SemiSupervisedTrainingInput] = None) -> SemiSupervisedTrainingResult:
        return await semi_supervised_resolver.train_semi_supervised_models(config)
    
    @strawberry.field(description="Predice estado de postulación (camelCase alias)", name="predictPostulacionEstado")
    async def predict_postulacion_estado_camel(self, input_data: PostulacionEstadoPredictionInput, model_type: Optional[str] = None) -> PostulacionEstadoPrediction:
        return await semi_supervised_resolver.predict_postulacion_estado(input_data, model_type)
    
    @strawberry.field(description="Predicción batch de estados (camelCase alias)", name="predictBatchEstados")
    async def predict_batch_estados_camel(self, input_data: BatchEstadoPredictionInput) -> BatchEstadoPredictionResult:
        return await semi_supervised_resolver.predict_batch_estados(input_data)
    
    @strawberry.field(description="Resumen de datos semi-supervisados (camelCase alias)", name="getSemiSupervisedDataSummary")
    async def get_semi_supervised_data_summary_camel(self) -> SemiSupervisedDataSummary:
        return await semi_supervised_resolver.get_semi_supervised_data_summary()
    
    @strawberry.field(description="Información de modelos entrenados (camelCase alias)", name="getTrainedModelsInfo")
    async def get_trained_models_info_camel(self) -> List[SemiSupervisedModelInfo]:
        return await semi_supervised_resolver.get_trained_models_info()
    
    @strawberry.field(description="Análisis de datos no etiquetados (camelCase alias)", name="analyzeUnlabeledData")
    async def analyze_unlabeled_data_camel(self) -> UnlabeledDataInsights:
        return await semi_supervised_resolver.analyze_unlabeled_data()


@strawberry.type
class Mutation:
    """Mutaciones disponibles en el sistema"""
    
    # Incluir mutaciones ML
    ml: MLMutation = strawberry.field(resolver=lambda: MLMutation())


schema = strawberry.Schema(query=Query, mutation=Mutation)
