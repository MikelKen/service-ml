"""
Extractor de datos desde PostgreSQL para análisis y entrenamiento semi-supervisado
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
import asyncpg
from datetime import datetime

from app.config.connection import db
from app.config.settings import settings

logger = logging.getLogger(__name__)


class PostgreSQLExtractor:
    """Extrae datos desde PostgreSQL ERP para análisis de postulaciones"""
    
    def __init__(self):
        self.connection_pool = None
    
    async def connect(self):
        """Conecta a PostgreSQL"""
        try:
            await db.connect()
            self.connection_pool = db.pool
            logger.info("Conectado a PostgreSQL")
        except Exception as e:
            logger.error(f"Error conectando a PostgreSQL: {e}")
            raise
    
    async def extract_postulaciones(self) -> pd.DataFrame:
        """Extrae todas las postulaciones con información completa"""
        if not self.connection_pool:
            await self.connect()
        
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
            p.url_cv,
            p.fecha_postulacion,
            p.estado,
            -- Información de la oferta
            o.id as oferta_id,
            o.titulo as oferta_titulo,
            o.descripcion as oferta_descripcion,
            o.salario,
            o.ubicacion,
            o.requisitos,
            o.fecha_publicacion,
            -- Información de la empresa
            e.id as empresa_id,
            e.nombre as empresa_nombre,
            e.correo as empresa_correo,
            e.rubro as empresa_rubro
        FROM postulaciones p
        INNER JOIN ofertas_trabajo o ON p.oferta_id = o.id
        INNER JOIN empresas e ON o.empresa_id = e.id
        ORDER BY p.fecha_postulacion DESC
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(query)
                
            df = pd.DataFrame([dict(row) for row in rows])
            logger.info(f"Extraídas {len(df)} postulaciones desde PostgreSQL")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extrayendo postulaciones: {e}")
            return pd.DataFrame()
    
    async def extract_entrevistas(self) -> pd.DataFrame:
        """Extrae información de entrevistas"""
        if not self.connection_pool:
            await self.connect()
        
        query = """
        SELECT 
            ent.id as entrevista_id,
            ent.fecha,
            ent.duracion_min,
            ent.objetivos_totales,
            ent.objetivos_cubiertos,
            ent.entrevistador,
            ent.postulacion_id,
            -- Información de evaluaciones
            COUNT(ev.id) as num_evaluaciones,
            AVG(ev.calificacion_tecnica) as promedio_tecnica,
            AVG(ev.calificacion_actitud) as promedio_actitud,
            AVG(ev.calificacion_general) as promedio_general
        FROM entrevistas ent
        LEFT JOIN evaluaciones ev ON ent.id = ev.entrevista_id
        GROUP BY ent.id, ent.fecha, ent.duracion_min, ent.objetivos_totales, 
                 ent.objetivos_cubiertos, ent.entrevistador, ent.postulacion_id
        ORDER BY ent.fecha DESC
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(query)
            
            df = pd.DataFrame([dict(row) for row in rows])
            logger.info(f"Extraídas {len(df)} entrevistas desde PostgreSQL")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extrayendo entrevistas: {e}")
            return pd.DataFrame()
    
    async def extract_visualizaciones(self) -> pd.DataFrame:
        """Extrae información de visualizaciones de ofertas"""
        if not self.connection_pool:
            await self.connect()
        
        query = """
        SELECT 
            v.id as visualizacion_id,
            v.fecha_visualizacion,
            v.origen,
            v.oferta_id,
            COUNT(*) OVER (PARTITION BY v.oferta_id) as total_visualizaciones
        FROM visualizaciones_oferta v
        ORDER BY v.fecha_visualizacion DESC
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(query)
            
            df = pd.DataFrame([dict(row) for row in rows])
            logger.info(f"Extraídas {len(df)} visualizaciones desde PostgreSQL")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extrayendo visualizaciones: {e}")
            return pd.DataFrame()
    
    async def get_estado_distribution(self) -> pd.DataFrame:
        """Obtiene la distribución de estados de postulaciones"""
        if not self.connection_pool:
            await self.connect()
        
        query = """
        SELECT estado, COUNT(*) as cantidad
        FROM postulaciones
        GROUP BY estado
        ORDER BY cantidad DESC
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(query)
            
            df = pd.DataFrame([dict(row) for row in rows])
            logger.info(f"Distribución de estados: {df.to_dict('records')}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo distribución de estados: {e}")
            return pd.DataFrame()
    
    async def extract_complete_dataset(self) -> pd.DataFrame:
        """Extrae el dataset completo combinando todas las tablas"""
        logger.info("Extrayendo dataset completo...")
        
        # Extraer datos de todas las tablas
        postulaciones_df = await self.extract_postulaciones()
        entrevistas_df = await self.extract_entrevistas()
        visualizaciones_df = await self.extract_visualizaciones()
        
        if postulaciones_df.empty:
            logger.error("No se pudieron extraer postulaciones")
            return pd.DataFrame()
        
        # Combinar con entrevistas
        if not entrevistas_df.empty:
            postulaciones_df = postulaciones_df.merge(
                entrevistas_df, 
                on='postulacion_id', 
                how='left'
            )
            logger.info("Datos de entrevistas combinados")
        
        # Agregar información de visualizaciones por oferta
        if not visualizaciones_df.empty:
            viz_summary = visualizaciones_df.groupby('oferta_id').agg({
                'total_visualizaciones': 'first',
                'fecha_visualizacion': 'count'
            }).rename(columns={'fecha_visualizacion': 'num_visualizaciones'}).reset_index()
            
            postulaciones_df = postulaciones_df.merge(
                viz_summary,
                on='oferta_id',
                how='left'
            )
            logger.info("Datos de visualizaciones combinados")
        
        # Rellenar valores faltantes
        postulaciones_df['total_visualizaciones'] = postulaciones_df['total_visualizaciones'].fillna(0)
        postulaciones_df['num_visualizaciones'] = postulaciones_df['num_visualizaciones'].fillna(0)
        postulaciones_df['num_evaluaciones'] = postulaciones_df['num_evaluaciones'].fillna(0)
        postulaciones_df['promedio_tecnica'] = postulaciones_df['promedio_tecnica'].fillna(0)
        postulaciones_df['promedio_actitud'] = postulaciones_df['promedio_actitud'].fillna(0)
        postulaciones_df['promedio_general'] = postulaciones_df['promedio_general'].fillna(0)
        
        logger.info(f"Dataset completo creado: {len(postulaciones_df)} registros, {len(postulaciones_df.columns)} columnas")
        
        return postulaciones_df
    
    async def save_to_mongo(self, df: pd.DataFrame, collection_name: str):
        """Guarda el dataset en MongoDB"""
        from app.config.connection import mongodb
        
        try:
            # Conectar a MongoDB si no está conectado
            if not mongodb.database:
                await mongodb.connect()
            
            collection = mongodb.get_collection(collection_name)
            
            # Convertir DataFrame a lista de diccionarios
            records = df.to_dict('records')
            
            # Procesar fechas y UUIDs
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif key.endswith('_id') and value:
                        record[key] = str(value)  # Convertir UUID a string
                    elif 'fecha' in key.lower() and value:
                        if isinstance(value, str):
                            record[key] = value
                        else:
                            record[key] = str(value)
            
            # Limpiar colección existente
            await collection.delete_many({})
            
            # Insertar nuevos datos
            if records:
                await collection.insert_many(records)
                logger.info(f"Guardados {len(records)} registros en MongoDB colección '{collection_name}'")
            else:
                logger.warning(f"No hay registros para guardar en '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error guardando en MongoDB: {e}")
            raise


# Instancia global
postgres_extractor = PostgreSQLExtractor()


async def extract_and_save_all_data():
    """Función conveniente para extraer todos los datos y guardarlos en MongoDB"""
    try:
        # Extraer dataset completo
        df = await postgres_extractor.extract_complete_dataset()
        
        if not df.empty:
            # Guardar en MongoDB
            await postgres_extractor.save_to_mongo(df, "postulaciones_completas")
            
            # También crear una colección solo con datos para training
            training_df = df[df['estado'].notna()].copy()  # Solo registros con estado conocido
            await postgres_extractor.save_to_mongo(training_df, "postulaciones_labeled")
            
            logger.info("Datos extraídos y guardados exitosamente")
            return df
        else:
            logger.error("No se pudieron extraer datos")
            return None
            
    except Exception as e:
        logger.error(f"Error en extracción completa: {e}")
        raise


async def get_estado_stats():
    """Función para obtener estadísticas de estados"""
    try:
        stats = await postgres_extractor.get_estado_distribution()
        return stats
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return None


if __name__ == "__main__":
    async def main():
        try:
            # Obtener estadísticas de estados
            print("=== ESTADÍSTICAS DE ESTADOS ===")
            stats = await get_estado_stats()
            if stats is not None:
                print(stats.to_string(index=False))
            
            # Extraer y guardar todos los datos
            print("\n=== EXTRAYENDO DATOS COMPLETOS ===")
            df = await extract_and_save_all_data()
            
            if df is not None:
                print(f"\nDataset extraído: {len(df)} registros")
                print(f"Columnas: {list(df.columns)}")
                print(f"\nDistribución de estados:")
                print(df['estado'].value_counts())
                
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())