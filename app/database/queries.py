"""
Database queries for the HR microservice
Handles all database operations for the entities in the ER model
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from app.database.connection import db

logger = logging.getLogger(__name__)

class DatabaseQueries:
    """Class to handle all database queries for HR entities"""
    
    # ==================== EMPRESA QUERIES ====================
    
    @staticmethod
    async def get_all_empresas() -> List[Dict[str, Any]]:
        """Get all companies"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT id, nombre, correo, rubro
                    FROM empresas
                    ORDER BY nombre
                """
                rows = await connection.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting companies: {e}")
            return []
    
    @staticmethod
    async def get_empresa_by_id(empresa_id: str) -> Optional[Dict[str, Any]]:
        """Get company by ID (UUID)"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT id, nombre, correo, rubro
                    FROM empresas
                    WHERE id = $1::uuid
                """
                row = await connection.fetchrow(query, empresa_id)
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting company {empresa_id}: {e}")
            return None
    
    # ==================== OFERTAS TRABAJO QUERIES ====================
    
    @staticmethod
    async def get_ofertas_by_empresa(empresa_id: str) -> List[Dict[str, Any]]:
        """Get job offers by company"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT o.id, o.titulo, o.descripcion, o.salario, o.ubicacion, 
                           o.requisitos, o.fecha_publicacion, o.empresa_id,
                           e.nombre as empresa_nombre
                    FROM ofertas_trabajo o
                    JOIN empresas e ON o.empresa_id = e.id
                    WHERE o.empresa_id = $1::uuid
                    ORDER BY o.fecha_publicacion DESC
                """
                rows = await connection.fetch(query, empresa_id)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting job offers for company {empresa_id}: {e}")
            return []
    
    @staticmethod
    async def get_all_ofertas() -> List[Dict[str, Any]]:
        """Get all job offers"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT o.id, o.titulo, o.descripcion, o.salario, o.ubicacion, 
                           o.requisitos, o.fecha_publicacion, o.empresa_id,
                           e.nombre as empresa_nombre
                    FROM ofertas_trabajo o
                    JOIN empresas e ON o.empresa_id = e.id
                    ORDER BY o.fecha_publicacion DESC
                """
                rows = await connection.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting job offers: {e}")
            return []
    
    @staticmethod
    async def get_oferta_by_id(oferta_id: str) -> Optional[Dict[str, Any]]:
        """Get job offer by ID (UUID)"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT o.id, o.titulo, o.descripcion, o.salario, o.ubicacion, 
                           o.requisitos, o.fecha_publicacion, o.empresa_id,
                           e.nombre as empresa_nombre
                    FROM ofertas_trabajo o
                    JOIN empresas e ON o.empresa_id = e.id
                    WHERE o.id = $1::uuid
                """
                row = await connection.fetchrow(query, oferta_id)
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting job offer {oferta_id}: {e}")
            return None
    
    # ==================== POSTULACIÓN QUERIES ====================
    
    @staticmethod
    async def get_postulaciones_by_oferta(oferta_id: str) -> List[Dict[str, Any]]:
        """Get applications by job offer"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT p.id, p.nombre, p.anios_experiencia, p.nivel_educacion, 
                           p.habilidades, p.idiomas, p.certificaciones, p.puesto_actual,
                           p.url_cv, p.fecha_postulacion, p.estado, p.oferta_id,
                           o.titulo as oferta_titulo,
                           o.empresa_id,
                           e.nombre as empresa_nombre
                    FROM postulaciones p
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                    WHERE p.oferta_id = $1::uuid
                    ORDER BY p.fecha_postulacion DESC
                """
                rows = await connection.fetch(query, oferta_id)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting applications for offer {oferta_id}: {e}")
            return []
    
    @staticmethod
    async def get_all_postulaciones() -> List[Dict[str, Any]]:
        """Get all applications"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT p.id, p.nombre, p.anios_experiencia, p.nivel_educacion, 
                           p.habilidades, p.idiomas, p.certificaciones, p.puesto_actual,
                           p.url_cv, p.fecha_postulacion, p.estado, p.oferta_id,
                           o.titulo as oferta_titulo,
                           o.empresa_id,
                           e.nombre as empresa_nombre
                    FROM postulaciones p
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                    ORDER BY p.fecha_postulacion DESC
                """
                rows = await connection.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting applications: {e}")
            return []
    
    @staticmethod
    async def get_postulacion_by_id(postulacion_id: str) -> Optional[Dict[str, Any]]:
        """Get application by ID (UUID)"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT p.id, p.nombre, p.anios_experiencia, p.nivel_educacion, 
                           p.habilidades, p.idiomas, p.certificaciones, p.puesto_actual,
                           p.url_cv, p.fecha_postulacion, p.estado, p.oferta_id,
                           o.titulo as oferta_titulo,
                           o.empresa_id,
                           e.nombre as empresa_nombre
                    FROM postulaciones p
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                    WHERE p.id = $1::uuid
                """
                row = await connection.fetchrow(query, postulacion_id)
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting application {postulacion_id}: {e}")
            return None
    
    # ==================== ENTREVISTA QUERIES ====================
    
    @staticmethod
    async def get_entrevistas_by_postulacion(postulacion_id: str) -> List[Dict[str, Any]]:
        """Get interviews by application"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT e.id, e.postulacion_id, e.fecha, e.duracion_min, 
                           e.objetivos_totales, e.objetivos_cubiertos, e.entrevistador,
                           p.nombre as postulante_nombre,
                           p.url_cv,
                           o.titulo as oferta_titulo,
                           emp.nombre as empresa_nombre
                    FROM entrevistas e
                    JOIN postulaciones p ON e.postulacion_id = p.id
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas emp ON o.empresa_id = emp.id
                    WHERE e.postulacion_id = $1::uuid
                    ORDER BY e.fecha DESC
                """
                rows = await connection.fetch(query, postulacion_id)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting interviews for application {postulacion_id}: {e}")
            return []
    
    @staticmethod
    async def get_all_entrevistas() -> List[Dict[str, Any]]:
        """Get all interviews"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT e.id, e.postulacion_id, e.fecha, e.duracion_min, 
                           e.objetivos_totales, e.objetivos_cubiertos, e.entrevistador,
                           p.nombre as postulante_nombre,
                           p.url_cv,
                           o.titulo as oferta_titulo,
                           emp.nombre as empresa_nombre
                    FROM entrevistas e
                    JOIN postulaciones p ON e.postulacion_id = p.id
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas emp ON o.empresa_id = emp.id
                    ORDER BY e.fecha DESC
                """
                rows = await connection.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting interviews: {e}")
            return []
    
    @staticmethod
    async def get_entrevista_by_id(entrevista_id: str) -> Optional[Dict[str, Any]]:
        """Get interview by ID (UUID)"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT e.id, e.postulacion_id, e.fecha, e.duracion_min, 
                           e.objetivos_totales, e.objetivos_cubiertos, e.entrevistador,
                           p.nombre as postulante_nombre,
                           p.url_cv,
                           o.titulo as oferta_titulo,
                           emp.nombre as empresa_nombre
                    FROM entrevistas e
                    JOIN postulaciones p ON e.postulacion_id = p.id
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas emp ON o.empresa_id = emp.id
                    WHERE e.id = $1::uuid
                """
                row = await connection.fetchrow(query, entrevista_id)
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting interview {entrevista_id}: {e}")
            return None
    
    # ==================== EVALUACIÓN QUERIES ====================
    
    @staticmethod
    async def get_evaluaciones_by_entrevista(entrevista_id: str) -> List[Dict[str, Any]]:
        """Get evaluations by interview"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT ev.id, ev.entrevista_id, ev.calificacion_tecnica, 
                           ev.calificacion_actitud, ev.calificacion_general, ev.comentarios,
                           e.fecha as entrevista_fecha,
                           e.entrevistador,
                           p.nombre as postulante_nombre,
                           p.url_cv,
                           o.titulo as oferta_titulo
                    FROM evaluaciones ev
                    JOIN entrevistas e ON ev.entrevista_id = e.id
                    JOIN postulaciones p ON e.postulacion_id = p.id
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    WHERE ev.entrevista_id = $1::uuid
                """
                rows = await connection.fetch(query, entrevista_id)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting evaluations for interview {entrevista_id}: {e}")
            return []
    
    @staticmethod
    async def get_all_evaluaciones() -> List[Dict[str, Any]]:
        """Get all evaluations"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT ev.id, ev.entrevista_id, ev.calificacion_tecnica, 
                           ev.calificacion_actitud, ev.calificacion_general, ev.comentarios,
                           e.fecha as entrevista_fecha,
                           e.entrevistador,
                           p.nombre as postulante_nombre,
                           p.url_cv,
                           o.titulo as oferta_titulo,
                           emp.nombre as empresa_nombre
                    FROM evaluaciones ev
                    JOIN entrevistas e ON ev.entrevista_id = e.id
                    JOIN postulaciones p ON e.postulacion_id = p.id
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    JOIN empresas emp ON o.empresa_id = emp.id
                    ORDER BY e.fecha DESC
                """
                rows = await connection.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting evaluations: {e}")
            return []
    
    # ==================== VISUALIZACIÓN OFERTA QUERIES ====================
    
    @staticmethod
    async def get_visualizaciones_by_oferta(oferta_id: str) -> List[Dict[str, Any]]:
        """Get offer views by job offer"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT v.id, v.oferta_id, v.fecha_visualizacion, v.origen,
                           o.titulo as oferta_titulo,
                           o.empresa_id,
                           e.nombre as empresa_nombre
                    FROM visualizaciones_oferta v
                    JOIN ofertas_trabajo o ON v.oferta_id = o.id
                    JOIN empresas e ON o.empresa_id = e.id
                    WHERE v.oferta_id = $1::uuid
                    ORDER BY v.fecha_visualizacion DESC
                """
                rows = await connection.fetch(query, oferta_id)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting views for offer {oferta_id}: {e}")
            return []
    
    # ==================== ANALYTICS QUERIES ====================
    
    @staticmethod
    async def get_estadisticas_empresa(empresa_id: str) -> Dict[str, Any]:
        """Get company statistics"""
        try:
            async with db.pool.acquire() as connection:
                # Get basic stats
                stats_query = """
                    SELECT 
                        COUNT(DISTINCT o.id) as total_ofertas,
                        COUNT(DISTINCT p.id) as total_postulaciones,
                        COUNT(DISTINCT e.id) as total_entrevistas,
                        COUNT(DISTINCT ev.id) as total_evaluaciones,
                        COUNT(DISTINCT v.id) as total_visualizaciones
                    FROM empresas emp
                    LEFT JOIN ofertas_trabajo o ON emp.id = o.empresa_id
                    LEFT JOIN postulaciones p ON o.id = p.oferta_id
                    LEFT JOIN entrevistas e ON p.id = e.postulacion_id
                    LEFT JOIN evaluaciones ev ON e.id = ev.entrevista_id
                    LEFT JOIN visualizaciones_oferta v ON o.id = v.oferta_id
                    WHERE emp.id = $1::uuid
                """
                stats = await connection.fetchrow(stats_query, empresa_id)
                
                # Get average evaluations
                avg_query = """
                    SELECT 
                        AVG(ev.calificacion_tecnica) as avg_tecnica,
                        AVG(ev.calificacion_actitud) as avg_actitud,
                        AVG(ev.calificacion_general) as avg_general
                    FROM evaluaciones ev
                    JOIN entrevistas e ON ev.entrevista_id = e.id
                    JOIN postulaciones p ON e.postulacion_id = p.id
                    JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    WHERE o.empresa_id = $1::uuid
                """
                averages = await connection.fetchrow(avg_query, empresa_id)
                
                return {
                    "empresa_id": empresa_id,
                    "total_ofertas": stats['total_ofertas'] or 0,
                    "total_postulaciones": stats['total_postulaciones'] or 0,
                    "total_entrevistas": stats['total_entrevistas'] or 0,
                    "total_evaluaciones": stats['total_evaluaciones'] or 0,
                    "total_visualizaciones": stats['total_visualizaciones'] or 0,
                    "promedio_calificacion_tecnica": float(averages['avg_tecnica']) if averages['avg_tecnica'] else 0.0,
                    "promedio_calificacion_actitud": float(averages['avg_actitud']) if averages['avg_actitud'] else 0.0,
                    "promedio_calificacion_general": float(averages['avg_general']) if averages['avg_general'] else 0.0
                }
        except Exception as e:
            logger.error(f"Error getting company stats {empresa_id}: {e}")
            return {}
    
    @staticmethod
    async def get_estadisticas_generales() -> Dict[str, Any]:
        """Get general statistics"""
        try:
            async with db.pool.acquire() as connection:
                query = """
                    SELECT 
                        COUNT(DISTINCT emp.id) as total_empresas,
                        COUNT(DISTINCT o.id) as total_ofertas,
                        COUNT(DISTINCT p.id) as total_postulaciones,
                        COUNT(DISTINCT e.id) as total_entrevistas,
                        COUNT(DISTINCT ev.id) as total_evaluaciones,
                        COUNT(DISTINCT v.id) as total_visualizaciones,
                        AVG(ev.calificacion_tecnica) as avg_tecnica,
                        AVG(ev.calificacion_actitud) as avg_actitud,
                        AVG(ev.calificacion_general) as avg_general
                    FROM empresas emp
                    LEFT JOIN ofertas_trabajo o ON emp.id = o.empresa_id
                    LEFT JOIN postulaciones p ON o.id = p.oferta_id
                    LEFT JOIN entrevistas e ON p.id = e.postulacion_id
                    LEFT JOIN evaluaciones ev ON e.id = ev.entrevista_id
                    LEFT JOIN visualizaciones_oferta v ON o.id = v.oferta_id
                """
                stats = await connection.fetchrow(query)
                
                return {
                    "total_empresas": stats['total_empresas'] or 0,
                    "total_ofertas": stats['total_ofertas'] or 0,
                    "total_postulaciones": stats['total_postulaciones'] or 0,
                    "total_entrevistas": stats['total_entrevistas'] or 0,
                    "total_evaluaciones": stats['total_evaluaciones'] or 0,
                    "total_visualizaciones": stats['total_visualizaciones'] or 0,
                    "promedio_calificacion_tecnica": float(stats['avg_tecnica']) if stats['avg_tecnica'] else 0.0,
                    "promedio_calificacion_actitud": float(stats['avg_actitud']) if stats['avg_actitud'] else 0.0,
                    "promedio_calificacion_general": float(stats['avg_general']) if stats['avg_general'] else 0.0
                }
        except Exception as e:
            logger.error(f"Error getting general stats: {e}")
            return {}

# Create instance
db_queries = DatabaseQueries()