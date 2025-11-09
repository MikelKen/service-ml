import asyncio
import asyncpg
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from app.config.connection import db
import json

logger = logging.getLogger(__name__)

class PostgreSQLExtractor:
    """Extractor de datos de PostgreSQL para el modelo semi-supervisado de postulaciones"""
    
    def __init__(self):
        self.connection = db
        
    async def extract_postulaciones_with_features(self) -> pd.DataFrame:
        """
        Extrae datos de postulaciones con características para el modelo semi-supervisado
        Incluye datos de ofertas, empresas y entrevistas
        """
        try:
            async with await self.connection.get_connection() as conn:
                query = """
                    SELECT 
                        p.id as postulacion_id,
                        p.nombre,
                        p.anios_experiencia,
                        p.nivel_educacion,
                        p.habilidades,
                        p.idiomas,
                        p.certificaciones,
                        p.puesto_actual,
                        p.fecha_postulacion,
                        p.estado,
                        p.telefono,
                        p.email,
                        
                        -- Datos de la oferta
                        o.id as oferta_id,
                        o.titulo as oferta_titulo,
                        o.descripcion as oferta_descripcion,
                        o.salario as oferta_salario,
                        o.ubicacion as oferta_ubicacion,
                        o.requisitos as oferta_requisitos,
                        o.fecha_publicacion as oferta_fecha_publicacion,
                        
                        -- Datos de la empresa
                        e.id as empresa_id,
                        e.nombre as empresa_nombre,
                        e.correo as empresa_correo,
                        e.rubro as empresa_rubro,
                        
                        -- Estadísticas de entrevistas
                        COALESCE(ent_stats.total_entrevistas, 0) as total_entrevistas,
                        COALESCE(ent_stats.promedio_duracion, 0) as promedio_duracion_entrevistas,
                        COALESCE(eval_stats.promedio_calificacion_tecnica, 0) as promedio_calificacion_tecnica,
                        COALESCE(eval_stats.promedio_calificacion_actitud, 0) as promedio_calificacion_actitud,
                        COALESCE(eval_stats.promedio_calificacion_general, 0) as promedio_calificacion_general
                        
                    FROM postulaciones p
                    LEFT JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    LEFT JOIN empresas e ON o.empresa_id = e.id
                    
                    -- Estadísticas de entrevistas
                    LEFT JOIN (
                        SELECT 
                            postulacion_id,
                            COUNT(*) as total_entrevistas,
                            AVG(duracion_min) as promedio_duracion
                        FROM entrevistas
                        GROUP BY postulacion_id
                    ) ent_stats ON p.id = ent_stats.postulacion_id
                    
                    -- Estadísticas de evaluaciones
                    LEFT JOIN (
                        SELECT 
                            e.postulacion_id,
                            AVG(ev.calificacion_tecnica) as promedio_calificacion_tecnica,
                            AVG(ev.calificacion_actitud) as promedio_calificacion_actitud,
                            AVG(ev.calificacion_general) as promedio_calificacion_general
                        FROM entrevistas e
                        LEFT JOIN evaluaciones ev ON e.id = ev.entrevista_id
                        GROUP BY e.postulacion_id
                    ) eval_stats ON p.id = eval_stats.postulacion_id
                    
                    ORDER BY p.fecha_postulacion DESC
                """
                
                logger.info("Ejecutando consulta para extraer postulaciones con características")
                rows = await conn.fetch(query)
                
                # Convertir a DataFrame
                data = [dict(row) for row in rows]
                df = pd.DataFrame(data)
                
                logger.info(f"Extraídas {len(df)} postulaciones con características")
                
                # Procesar tipos de datos
                if not df.empty:
                    df = self._process_data_types(df)
                
                return df
                
        except Exception as e:
            logger.error(f"Error extrayendo postulaciones: {str(e)}")
            raise e
    
    async def extract_estado_distribution(self) -> Dict[str, int]:
        """Extrae la distribución de estados de las postulaciones"""
        try:
            async with await self.connection.get_connection() as conn:
                query = """
                    SELECT estado, COUNT(*) as count
                    FROM postulaciones
                    WHERE estado IS NOT NULL
                    GROUP BY estado
                    ORDER BY count DESC
                """
                
                rows = await conn.fetch(query)
                distribution = {row['estado']: row['count'] for row in rows}
                
                logger.info(f"Distribución de estados: {distribution}")
                return distribution
                
        except Exception as e:
            logger.error(f"Error extrayendo distribución de estados: {str(e)}")
            raise e
    
    async def extract_missing_estado_postulaciones(self) -> pd.DataFrame:
        """Extrae postulaciones sin estado (para predicción semi-supervisada)"""
        try:
            async with await self.connection.get_connection() as conn:
                query = """
                    SELECT 
                        p.id as postulacion_id,
                        p.nombre,
                        p.anios_experiencia,
                        p.nivel_educacion,
                        p.habilidades,
                        p.idiomas,
                        p.certificaciones,
                        p.puesto_actual,
                        p.fecha_postulacion,
                        p.telefono,
                        p.email,
                        
                        -- Datos de la oferta
                        o.titulo as oferta_titulo,
                        o.descripcion as oferta_descripcion,
                        o.salario as oferta_salario,
                        o.ubicacion as oferta_ubicacion,
                        o.requisitos as oferta_requisitos,
                        
                        -- Datos de la empresa
                        e.nombre as empresa_nombre,
                        e.rubro as empresa_rubro
                        
                    FROM postulaciones p
                    LEFT JOIN ofertas_trabajo o ON p.oferta_id = o.id
                    LEFT JOIN empresas e ON o.empresa_id = e.id
                    WHERE p.estado IS NULL OR p.estado = ''
                    ORDER BY p.fecha_postulacion DESC
                """
                
                rows = await conn.fetch(query)
                data = [dict(row) for row in rows]
                df = pd.DataFrame(data)
                
                logger.info(f"Encontradas {len(df)} postulaciones sin estado")
                
                if not df.empty:
                    df = self._process_data_types(df)
                
                return df
                
        except Exception as e:
            logger.error(f"Error extrayendo postulaciones sin estado: {str(e)}")
            raise e
    
    def _process_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa los tipos de datos del DataFrame"""
        try:
            # Convertir fechas
            if 'fecha_postulacion' in df.columns:
                df['fecha_postulacion'] = pd.to_datetime(df['fecha_postulacion'], errors='coerce')
            
            if 'oferta_fecha_publicacion' in df.columns:
                df['oferta_fecha_publicacion'] = pd.to_datetime(df['oferta_fecha_publicacion'], errors='coerce')
            
            # Convertir valores numéricos
            numeric_columns = [
                'anios_experiencia', 'oferta_salario', 'total_entrevistas',
                'promedio_duracion_entrevistas', 'promedio_calificacion_tecnica',
                'promedio_calificacion_actitud', 'promedio_calificacion_general'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Rellenar valores nulos en texto
            text_columns = [
                'habilidades', 'idiomas', 'certificaciones', 'puesto_actual',
                'oferta_descripcion', 'oferta_requisitos', 'empresa_rubro'
            ]
            
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('')
            
            return df
            
        except Exception as e:
            logger.error(f"Error procesando tipos de datos: {str(e)}")
            return df
    
    async def get_table_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de las tablas relacionadas"""
        try:
            async with await self.connection.get_connection() as conn:
                stats = {}
                
                # Estadísticas de postulaciones
                postulaciones_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN estado IS NOT NULL AND estado != '' THEN 1 END) as con_estado,
                        COUNT(CASE WHEN estado IS NULL OR estado = '' THEN 1 END) as sin_estado
                    FROM postulaciones
                """)
                stats['postulaciones'] = dict(postulaciones_stats)
                
                # Estadísticas de ofertas
                ofertas_stats = await conn.fetchrow("""
                    SELECT COUNT(*) as total FROM ofertas_trabajo
                """)
                stats['ofertas'] = dict(ofertas_stats)
                
                # Estadísticas de empresas
                empresas_stats = await conn.fetchrow("""
                    SELECT COUNT(*) as total FROM empresas
                """)
                stats['empresas'] = dict(empresas_stats)
                
                # Estadísticas de entrevistas
                entrevistas_stats = await conn.fetchrow("""
                    SELECT COUNT(*) as total FROM entrevistas
                """)
                stats['entrevistas'] = dict(entrevistas_stats)
                
                logger.info(f"Estadísticas de tablas: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            raise e

# Instancia global del extractor
postgres_extractor = PostgreSQLExtractor()