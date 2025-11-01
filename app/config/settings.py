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
    database_url: str = "postgresql://neondb_owner:npg_5PdCLF6NrZni@ep-lucky-darkness-ah2vls5k-pooler.c-3.us-east-1.aws.neon.tech/service-erp-rrhh?sslmode=require&channel_binding=require"
    
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