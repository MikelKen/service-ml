from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App configuration
    app_name: str = "service_ml"
    app_version: str = "1.0.0"
    description: str = "FastAPI ML Service with GraphQL"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 3001
    debug: bool = True
    environment: str = "development"
    
    # Database configuration
    db_url_postgres: str = ""
    db_url_mongodb: str = ""
    mongodb_username: str = ""
    mongodb_password: str = ""
    mongodb_host: str = ""
    mongodb_database: str = ""
    
    @property
    def database_url(self) -> str:
        return self.db_url_postgres
    
    @property
    def mongodb_url(self) -> str:
        return self.db_url_mongodb
    
    # ML Configuration
    ml_models_path: str = "trained_models"
    ml_data_path: str = "data"
    ml_default_model: str = "hiring_model.pkl"
    ml_batch_size: int = 100
    ml_max_features: int = 10000
    ml_random_state: int = 42
    
    # Training configuration
    ml_test_size: float = 0.2
    ml_validation_size: float = 0.2
    ml_cross_validation_folds: int = 5
    ml_enable_feature_selection: bool = True
    ml_enable_hyperparameter_tuning: bool = True
    
    # Model performance thresholds
    ml_min_accuracy: float = 0.7
    ml_min_precision: float = 0.6
    ml_min_recall: float = 0.6
    ml_min_f1_score: float = 0.6
    ml_min_roc_auc: float = 0.7
    
    # Data preprocessing
    ml_max_text_length: int = 1000
    ml_min_text_length: int = 10
    ml_tfidf_max_features: int = 5000
    ml_tfidf_ngram_range: tuple = (1, 2)
    
    # CORS configuration
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # GraphQL configuration
    graphql_path: str = "/graphql"
    graphiql_enabled: bool = True
    
    # API configuration
    api_prefix: str = "/api"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Global settings instance
settings = get_settings()