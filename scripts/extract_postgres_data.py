#!/usr/bin/env python3
"""
🔄 EXTRACCIÓN DE DATOS DESDE POSTGRESQL
Script para extraer datos de postulaciones, ofertas, empresas y entrevistas
desde PostgreSQL para el modelo semi-supervisado
"""

import asyncio
import asyncpg
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import sys

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLDataExtractor:
    """Extractor de datos desde PostgreSQL para modelo semi-supervisado"""
    
    def __init__(self):
        self.connection_pool = None
        self.database_url = settings.database_url or os.getenv("DB_URL_POSTGRES")
        
        if not self.database_url:
            raise ValueError("No se encontró URL de base de datos PostgreSQL")
    
    async def connect(self):
        """Establecer conexión con PostgreSQL"""
        try:
            logger.info("🔌 Conectando a PostgreSQL...")
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=5,
                command_timeout=60
            )
            logger.info("✅ Conexión establecida con PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ Error conectando a PostgreSQL: {e}")
            return False
    
    async def disconnect(self):
        """Cerrar conexión con PostgreSQL"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("🔌 Conexión cerrada con PostgreSQL")
    
    async def extract_postulaciones_data(self) -> pd.DataFrame:
        """Extraer datos completos de postulaciones con información relacionada"""
        logger.info("📊 Extrayendo datos de postulaciones...")
        
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
            p.url_cv,
            CASE 
                WHEN p.fecha_postulacion ~ '^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$' 
                THEN p.fecha_postulacion::date
                ELSE CURRENT_DATE
            END as fecha_postulacion,
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
            CASE 
                WHEN o.fecha_publicacion ~ '^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$' 
                THEN o.fecha_publicacion::date
                ELSE CURRENT_DATE
            END as fecha_publicacion,
            -- Datos de la empresa
            e.id as empresa_id,
            e.nombre as empresa_nombre,
            e.correo as empresa_correo,
            e.rubro as empresa_rubro
        FROM postulaciones p
        INNER JOIN ofertas_trabajo o ON p.oferta_id = o.id
        INNER JOIN empresas e ON o.empresa_id = e.id
        ORDER BY p.fecha_postulacion DESC
        """
        
        async with self.connection_pool.acquire() as connection:
            rows = await connection.fetch(query)
            
        if not rows:
            logger.warning("⚠️ No se encontraron postulaciones")
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        logger.info(f"📈 Extraídas {len(df)} postulaciones")
        
        return df
    
    async def extract_entrevistas_data(self) -> pd.DataFrame:
        """Extraer datos de entrevistas y evaluaciones"""
        logger.info("🎤 Extrayendo datos de entrevistas...")
        
        query = """
        SELECT 
            e.id as entrevista_id,
            CASE 
                WHEN e.fecha ~ '^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$' 
                THEN e.fecha::date
                ELSE CURRENT_DATE
            END as entrevista_fecha,
            e.duracion_min,
            e.objetivos_totales,
            e.objetivos_cubiertos,
            e.entrevistador,
            e.postulacion_id,
            -- Evaluaciones
            ev.id as evaluacion_id,
            ev.calificacion_tecnica,
            ev.calificacion_actitud,
            ev.calificacion_general,
            ev.comentarios as evaluacion_comentarios
        FROM entrevistas e
        LEFT JOIN evaluaciones ev ON e.id = ev.entrevista_id
        ORDER BY e.fecha DESC
        """
        
        async with self.connection_pool.acquire() as connection:
            rows = await connection.fetch(query)
        
        if not rows:
            logger.warning("⚠️ No se encontraron entrevistas")
            return pd.DataFrame()
        
        df = pd.DataFrame([dict(row) for row in rows])
        logger.info(f"🎤 Extraídas {len(df)} entrevistas/evaluaciones")
        
        return df
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de los datos"""
        logger.info("📊 Obteniendo estadísticas de datos...")
        
        stats_queries = {
            'total_postulaciones': "SELECT COUNT(*) FROM postulaciones",
            'total_ofertas': "SELECT COUNT(*) FROM ofertas_trabajo",
            'total_empresas': "SELECT COUNT(*) FROM empresas",
            'total_entrevistas': "SELECT COUNT(*) FROM entrevistas",
            'estados_postulaciones': """
                SELECT estado, COUNT(*) as cantidad 
                FROM postulaciones 
                GROUP BY estado 
                ORDER BY cantidad DESC
            """,
            'ofertas_por_empresa': """
                SELECT e.nombre, COUNT(o.id) as num_ofertas
                FROM empresas e
                LEFT JOIN ofertas_trabajo o ON e.id = o.empresa_id
                GROUP BY e.id, e.nombre
                ORDER BY num_ofertas DESC
                LIMIT 10
            """,
            'postulaciones_por_mes': """
                SELECT 
                    DATE_TRUNC('month', 
                        CASE 
                            WHEN fecha_postulacion ~ '^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$' 
                            THEN fecha_postulacion::date
                            ELSE NULL
                        END
                    ) as mes,
                    COUNT(*) as cantidad
                FROM postulaciones 
                WHERE fecha_postulacion IS NOT NULL
                    AND fecha_postulacion ~ '^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$'
                GROUP BY mes
                HAVING DATE_TRUNC('month', 
                    CASE 
                        WHEN fecha_postulacion ~ '^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$' 
                        THEN fecha_postulacion::date
                        ELSE NULL
                    END
                ) IS NOT NULL
                ORDER BY mes DESC
                LIMIT 12
            """
        }
        
        stats = {}
        
        async with self.connection_pool.acquire() as connection:
            # Estadísticas simples
            for key in ['total_postulaciones', 'total_ofertas', 'total_empresas', 'total_entrevistas']:
                result = await connection.fetchval(stats_queries[key])
                stats[key] = result
            
            # Estadísticas complejas
            for key in ['estados_postulaciones', 'ofertas_por_empresa', 'postulaciones_por_mes']:
                rows = await connection.fetch(stats_queries[key])
                stats[key] = [dict(row) for row in rows]
        
        return stats
    
    async def extract_all_data(self) -> Dict[str, pd.DataFrame]:
        """Extraer todos los datos necesarios"""
        logger.info("🔍 Iniciando extracción completa de datos...")
        
        try:
            # Conectar
            if not await self.connect():
                raise Exception("No se pudo conectar a PostgreSQL")
            
            # Extraer datos
            postulaciones_df = await self.extract_postulaciones_data()
            entrevistas_df = await self.extract_entrevistas_data()
            stats = await self.get_data_statistics()
            
            # Mostrar estadísticas
            self._log_statistics(stats)
            
            return {
                'postulaciones': postulaciones_df,
                'entrevistas': entrevistas_df,
                'statistics': stats
            }
            
        finally:
            await self.disconnect()
    
    def _log_statistics(self, stats: Dict[str, Any]):
        """Mostrar estadísticas en logs"""
        logger.info("📊 ESTADÍSTICAS DE DATOS:")
        logger.info(f"  📝 Total Postulaciones: {stats.get('total_postulaciones', 0)}")
        logger.info(f"  💼 Total Ofertas: {stats.get('total_ofertas', 0)}")
        logger.info(f"  🏢 Total Empresas: {stats.get('total_empresas', 0)}")
        logger.info(f"  🎤 Total Entrevistas: {stats.get('total_entrevistas', 0)}")
        
        # Estados de postulaciones
        logger.info("  📊 Estados de Postulaciones:")
        for estado in stats.get('estados_postulaciones', []):
            logger.info(f"    - {estado['estado']}: {estado['cantidad']}")
    
    def save_data_to_files(self, data: Dict[str, pd.DataFrame], output_dir: str = "extracted_data"):
        """Guardar datos extraídos en archivos CSV"""
        logger.info(f"💾 Guardando datos en {output_dir}...")
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar DataFrames
        for name, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"  ✅ Guardado: {filepath} ({len(df)} filas)")
        
        # Guardar estadísticas
        if 'statistics' in data:
            stats_path = os.path.join(output_dir, "statistics.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                # Convertir DataFrames a dict para JSON
                stats_for_json = {}
                for key, value in data['statistics'].items():
                    if isinstance(value, list):
                        stats_for_json[key] = value
                    else:
                        stats_for_json[key] = value
                
                json.dump(stats_for_json, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"  ✅ Guardado: {stats_path}")
        
        logger.info("✅ Todos los datos guardados correctamente")


async def main():
    """Función principal"""
    extractor = PostgreSQLDataExtractor()
    
    try:
        # Extraer datos
        data = await extractor.extract_all_data()
        
        # Guardar en archivos
        extractor.save_data_to_files(data)
        
        logger.info("🎉 Extracción de datos completada exitosamente")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Error en extracción de datos: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())