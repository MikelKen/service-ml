"""
Database queries for Machine Learning data extraction
Handles data extraction from the HR database for ML model training
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from app.database.connection import db
import pandas as pd

logger = logging.getLogger(__name__)

class MLDatabaseQueries:
    """Class to handle ML-specific database queries for training data extraction"""
    
    @staticmethod
    async def get_training_data_for_hiring_prediction() -> List[Dict[str, Any]]:
        """
        Get comprehensive training data for hiring prediction model.
        Combines data from postulaciones, ofertas, entrevistas, evaluaciones, and empresas.
        
        Returns:
            List of dictionaries containing training data with target variable (contacted/hired)
        """
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT 
                        -- Postulacion data (candidate features)
                        p.id as postulacion_id,
                        p.nombre as candidato_nombre,
                        p.anios_experiencia,
                        p.nivel_educacion,
                        p.habilidades,
                        p.idiomas,
                        p.certificaciones,
                        p.puesto_actual,
                        p.fecha_postulacion,
                        p.estado as postulacion_estado,
                        
                        -- Oferta data (job features)
                        o.id as oferta_id,
                        o.titulo as oferta_titulo,
                        o.descripcion as oferta_descripcion,
                        o.salario,
                        o.ubicacion,
                        o.requisitos,
                        o.fecha_publicacion,
                        
                        -- Empresa data (company features)
                        e.id as empresa_id,
                        e.nombre as empresa_nombre,
                        e.rubro as empresa_rubro,
                        
                        -- Interview data (if exists)
                        ent.id as entrevista_id,
                        ent.fecha as entrevista_fecha,
                        ent.duracion_min,
                        ent.objetivos_totales,
                        ent.objetivos_cubiertos,
                        
                        -- Evaluation data (if exists)
                        ev.calificacion_tecnica,
                        ev.calificacion_actitud,
                        ev.calificacion_general,
                        ev.comentarios,
                        
                        -- Target variables (to determine if candidate was contacted/hired)
                        CASE 
                            WHEN ent.id IS NOT NULL THEN 1 
                            ELSE 0 
                        END as fue_contactado,
                        
                        CASE 
                            WHEN ev.id IS NOT NULL THEN 1 
                            ELSE 0 
                        END as fue_evaluado,
                        
                        CASE 
                            WHEN p.estado IN ('contratado', 'hired', 'aceptado') THEN 1
                            WHEN ent.id IS NOT NULL THEN 1
                            ELSE 0 
                        END as target_contactado
                        
                    FROM postulaciones p
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                    LEFT JOIN entrevistas ent ON p.id = ent.postulacion_id
                    LEFT JOIN evaluaciones ev ON ent.id = ev.entrevista_id
                    
                    ORDER BY p.fecha_postulacion DESC
                """
                
                rows = await connection.fetch(query)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting ML training data: {e}")
            return []
    
    @staticmethod
    async def get_training_data_aggregated() -> List[Dict[str, Any]]:
        """
        Get aggregated training data where each postulacion has one row with averaged evaluations.
        This is useful when a candidate has multiple interviews/evaluations.
        """
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT 
                        -- Postulacion data (candidate features)
                        p.id as postulacion_id,
                        p.nombre as candidato_nombre,
                        p.anios_experiencia,
                        p.nivel_educacion,
                        p.habilidades,
                        p.idiomas,
                        p.certificaciones,
                        p.puesto_actual,
                        p.fecha_postulacion,
                        p.estado as postulacion_estado,
                        
                        -- Oferta data (job features)
                        o.id as oferta_id,
                        o.titulo as oferta_titulo,
                        o.descripcion as oferta_descripcion,
                        o.salario,
                        o.ubicacion,
                        o.requisitos,
                        o.fecha_publicacion,
                        
                        -- Empresa data (company features)
                        e.id as empresa_id,
                        e.nombre as empresa_nombre,
                        e.rubro as empresa_rubro,
                        
                        -- Aggregated interview/evaluation data
                        COUNT(DISTINCT ent.id) as num_entrevistas,
                        COUNT(DISTINCT ev.id) as num_evaluaciones,
                        AVG(ent.duracion_min) as avg_duracion_entrevista,
                        AVG(ev.calificacion_tecnica) as avg_calificacion_tecnica,
                        AVG(ev.calificacion_actitud) as avg_calificacion_actitud,
                        AVG(ev.calificacion_general) as avg_calificacion_general,
                        
                        -- Target variable
                        CASE 
                            WHEN COUNT(ent.id) > 0 THEN 1 
                            ELSE 0 
                        END as target_contactado
                        
                    FROM postulaciones p
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                    LEFT JOIN entrevistas ent ON p.id = ent.postulacion_id
                    LEFT JOIN evaluaciones ev ON ent.id = ev.entrevista_id
                    
                    GROUP BY 
                        p.id, p.nombre, p.anios_experiencia, p.nivel_educacion, 
                        p.habilidades, p.idiomas, p.certificaciones, p.puesto_actual,
                        p.fecha_postulacion, p.estado,
                        o.id, o.titulo, o.descripcion, o.salario, o.ubicacion, 
                        o.requisitos, o.fecha_publicacion,
                        e.id, e.nombre, e.rubro
                        
                    ORDER BY p.fecha_postulacion DESC
                """
                
                rows = await connection.fetch(query)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting aggregated ML training data: {e}")
            return []
    
    @staticmethod
    async def get_postulaciones_for_prediction(empresa_id: str = None, 
                                             oferta_id: str = None) -> List[Dict[str, Any]]:
        """
        Get postulaciones data for making predictions on new applications.
        
        Args:
            empresa_id: Filter by company ID (optional)
            oferta_id: Filter by job offer ID (optional)
        """
        try:
            async with db.pool.acquire() as connection:
                base_query = """
                    SELECT 
                        -- Postulacion data
                        p.id as postulacion_id,
                        p.nombre as candidato_nombre,
                        p.anios_experiencia,
                        p.nivel_educacion,
                        p.habilidades,
                        p.idiomas,
                        p.certificaciones,
                        p.puesto_actual,
                        p.fecha_postulacion,
                        p.estado as postulacion_estado,
                        
                        -- Oferta data
                        o.id as oferta_id,
                        o.titulo as oferta_titulo,
                        o.descripcion as oferta_descripcion,
                        o.salario,
                        o.ubicacion,
                        o.requisitos,
                        o.fecha_publicacion,
                        
                        -- Empresa data
                        e.id as empresa_id,
                        e.nombre as empresa_nombre,
                        e.rubro as empresa_rubro
                        
                    FROM postulaciones p
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                """
                
                conditions = []
                params = []
                
                if empresa_id:
                    conditions.append(f"e.id = ${len(params) + 1}::uuid")
                    params.append(empresa_id)
                
                if oferta_id:
                    conditions.append(f"o.id = ${len(params) + 1}::uuid")
                    params.append(oferta_id)
                
                if conditions:
                    base_query += " WHERE " + " AND ".join(conditions)
                
                base_query += " ORDER BY p.fecha_postulacion DESC"
                
                if params:
                    rows = await connection.fetch(base_query, *params)
                else:
                    rows = await connection.fetch(base_query)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting postulaciones for prediction: {e}")
            return []
    
    @staticmethod
    async def get_feature_statistics() -> Dict[str, Any]:
        """
        Get statistical information about the features in the dataset.
        Useful for understanding data distribution and quality.
        """
        try:
            async with db.pool.acquire() as connection:
                stats_query = """
                    SELECT 
                        COUNT(*) as total_postulaciones,
                        COUNT(DISTINCT e.id) as total_empresas,
                        COUNT(DISTINCT o.id) as total_ofertas,
                        AVG(p.anios_experiencia) as avg_experiencia,
                        MIN(p.anios_experiencia) as min_experiencia,
                        MAX(p.anios_experiencia) as max_experiencia,
                        AVG(o.salario) as avg_salario,
                        MIN(o.salario) as min_salario,
                        MAX(o.salario) as max_salario,
                        COUNT(DISTINCT p.nivel_educacion) as distinct_education_levels,
                        COUNT(DISTINCT e.rubro) as distinct_industry_sectors,
                        COUNT(CASE WHEN ent.id IS NOT NULL THEN 1 END) as contacted_candidates,
                        COUNT(CASE WHEN ent.id IS NULL THEN 1 END) as not_contacted_candidates
                    FROM postulaciones p
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                    LEFT JOIN entrevistas ent ON p.id = ent.postulacion_id
                """
                
                stats = await connection.fetchrow(stats_query)
                
                # Get distribution of education levels
                education_query = """
                    SELECT nivel_educacion, COUNT(*) as count
                    FROM postulaciones
                    GROUP BY nivel_educacion
                    ORDER BY count DESC
                """
                education_dist = await connection.fetch(education_query)
                
                # Get distribution of industry sectors
                sector_query = """
                    SELECT e.rubro, COUNT(DISTINCT p.id) as postulaciones_count
                    FROM empresas e
                    JOIN ofertas_trabajo o ON e.id = o.empresa_id
                    JOIN postulaciones p ON o.id = p.oferta_id
                    GROUP BY e.rubro
                    ORDER BY postulaciones_count DESC
                """
                sector_dist = await connection.fetch(sector_query)
                
                return {
                    "general_stats": dict(stats),
                    "education_distribution": [dict(row) for row in education_dist],
                    "sector_distribution": [dict(row) for row in sector_dist]
                }
                
        except Exception as e:
            logger.error(f"Error getting feature statistics: {e}")
            return {}
    
    @staticmethod
    async def export_training_data_to_dataframe() -> pd.DataFrame:
        """
        Export training data directly to a pandas DataFrame for ML processing.
        """
        try:
            training_data = await MLDatabaseQueries.get_training_data_aggregated()
            
            if not training_data:
                logger.warning("No training data found in database")
                return pd.DataFrame()
            
            df = pd.DataFrame(training_data)
            logger.info(f"Exported {len(df)} records to DataFrame for ML training")
            
            return df
            
        except Exception as e:
            logger.error(f"Error exporting training data to DataFrame: {e}")
            return pd.DataFrame()
    
    @staticmethod
    async def get_recent_applications_for_monitoring(days: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent applications for model monitoring and performance evaluation.
        
        Args:
            days: Number of days to look back for recent applications
        """
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT 
                        p.id as postulacion_id,
                        p.nombre as candidato_nombre,
                        p.anios_experiencia,
                        p.nivel_educacion,
                        p.habilidades,
                        p.idiomas,
                        p.certificaciones,
                        p.puesto_actual,
                        p.fecha_postulacion,
                        p.estado as postulacion_estado,
                        o.titulo as oferta_titulo,
                        o.salario,
                        o.ubicacion,
                        e.nombre as empresa_nombre,
                        e.rubro as empresa_rubro,
                        CASE 
                            WHEN ent.id IS NOT NULL THEN 1 
                            ELSE 0 
                        END as fue_contactado
                    FROM postulaciones p
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                    LEFT JOIN entrevistas ent ON p.id = ent.postulacion_id
                    WHERE p.fecha_postulacion >= (CURRENT_DATE - INTERVAL '%s days')
                    ORDER BY p.fecha_postulacion DESC
                """ % days
                
                rows = await connection.fetch(query)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting recent applications: {e}")
            return []

# Create instance
ml_db_queries = MLDatabaseQueries()