"""
Servicio de sincronización automática PostgreSQL -> MongoDB

Este servicio ejecuta en background y sincroniza automáticamente
los nuevos registros de PostgreSQL a MongoDB.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import uuid

from pymongo import UpdateOne
from app.config.connection import db, mongodb

logger = logging.getLogger("sync_service")

class AutoSyncService:
    def __init__(self, sync_interval_seconds: int = 60):
        self.sync_interval = sync_interval_seconds
        self.is_running = False
        self.last_sync_timestamps = {
            'empresas': None,
            'ofertas_trabajo': None, 
            'postulaciones': None
        }
        self._task = None

    def _normalize_value(self, value: Any) -> Any:
        """Normaliza tipos de datos para MongoDB"""
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, uuid.UUID):
            return str(value)
        elif value is None:
            return None
        return value

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza un registro completo"""
        return {k: self._normalize_value(v) for k, v in record.items()}

    async def get_new_empresas(self, since: datetime = None) -> List[Dict[str, Any]]:
        """Obtiene empresas nuevas/modificadas desde 'since'"""
        query = """
        SELECT 
            id AS empresa_id,
            nombre,
            rubro,
            correo,
            EXTRACT(EPOCH FROM COALESCE(updated_at, created_at))::bigint as last_modified_ts
        FROM empresas
        """
        
        params = []
        if since:
            query += " WHERE COALESCE(updated_at, created_at) > $1"
            params.append(since)
        
        query += " ORDER BY COALESCE(updated_at, created_at) ASC"
        
        async with await db.get_connection() as conn:
            rows = await conn.fetch(query, *params)
            return [self._normalize_record(dict(row)) for row in rows]

    async def get_new_ofertas(self, since: datetime = None) -> List[Dict[str, Any]]:
        """Obtiene ofertas nuevas/modificadas desde 'since'"""
        query = """
        SELECT 
            id AS oferta_id,
            titulo,
            salario,
            ubicacion,
            requisitos,
            empresa_id,
            EXTRACT(EPOCH FROM COALESCE(updated_at, created_at))::bigint as last_modified_ts
        FROM ofertas_trabajo
        """
        
        params = []
        if since:
            query += " WHERE COALESCE(updated_at, created_at) > $1"
            params.append(since)
            
        query += " ORDER BY COALESCE(updated_at, created_at) ASC"
        
        async with await db.get_connection() as conn:
            rows = await conn.fetch(query, *params)
            return [self._normalize_record(dict(row)) for row in rows]

    async def get_new_postulaciones(self, since: datetime = None) -> List[Dict[str, Any]]:
        """Obtiene postulaciones nuevas/modificadas desde 'since'"""
        query = """
        SELECT 
            id AS postulante_id,
            anios_experiencia,
            nivel_educacion,
            habilidades,
            idiomas,
            certificaciones,
            puesto_actual,
            oferta_id,
            EXTRACT(EPOCH FROM COALESCE(updated_at, created_at))::bigint as last_modified_ts
        FROM postulaciones
        """
        
        params = []
        if since:
            query += " WHERE COALESCE(updated_at, created_at) > $1"
            params.append(since)
            
        query += " ORDER BY COALESCE(updated_at, created_at) ASC"
        
        async with await db.get_connection() as conn:
            rows = await conn.fetch(query, *params)
            return [self._normalize_record(dict(row)) for row in rows]

    async def sync_collection(self, collection_name: str, id_field: str, records: List[Dict[str, Any]]) -> int:
        """Sincroniza registros en una colección MongoDB"""
        if not records:
            return 0

        await mongodb.connect()
        coll = mongodb.get_collection(collection_name)
        
        ops = []
        for record in records:
            # Normalizar ID a string y agregar timestamps
            if record.get(id_field):
                record[id_field] = str(record[id_field])
            
            record.setdefault("updated_at", datetime.now().isoformat() + "Z")
            record.setdefault("created_at", datetime.now().isoformat() + "Z")
            
            # Usar el ID como _id de MongoDB para upserts eficientes
            ops.append(
                UpdateOne(
                    {"_id": record[id_field]}, 
                    {"$set": record}, 
                    upsert=True
                )
            )

        if ops:
            result = await coll.bulk_write(ops, ordered=False)
            synced = (result.upserted_count or 0) + (result.modified_count or 0)
            logger.info(f"Sincronizados {synced} registros en {collection_name}")
            return synced
        
        return 0

    async def sync_all_collections(self) -> Dict[str, int]:
        """Ejecuta sincronización completa de todas las colecciones"""
        stats = {"empresas": 0, "ofertas": 0, "postulaciones": 0}
        
        try:
            # Sincronizar empresas
            empresas = await self.get_new_empresas(self.last_sync_timestamps.get('empresas'))
            stats["empresas"] = await self.sync_collection("companies_features", "empresa_id", empresas)
            
            # Sincronizar ofertas
            ofertas = await self.get_new_ofertas(self.last_sync_timestamps.get('ofertas_trabajo'))
            stats["ofertas"] = await self.sync_collection("job_offers_features", "oferta_id", ofertas)
            
            # Sincronizar postulaciones
            postulaciones = await self.get_new_postulaciones(self.last_sync_timestamps.get('postulaciones'))
            stats["postulaciones"] = await self.sync_collection("candidates_features", "postulante_id", postulaciones)
            
            # Actualizar timestamps de última sincronización
            now = datetime.now()
            self.last_sync_timestamps['empresas'] = now
            self.last_sync_timestamps['ofertas_trabajo'] = now
            self.last_sync_timestamps['postulaciones'] = now
            
            total_synced = sum(stats.values())
            if total_synced > 0:
                logger.info(f"Sincronización completada: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error en sincronización: {e}")
            return stats

    async def _sync_loop(self):
        """Loop principal de sincronización"""
        logger.info(f"Iniciando servicio de sincronización automática (intervalo: {self.sync_interval}s)")
        
        while self.is_running:
            try:
                await self.sync_all_collections()
                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                logger.info("Servicio de sincronización cancelado")
                break
            except Exception as e:
                logger.error(f"Error en loop de sincronización: {e}")
                await asyncio.sleep(self.sync_interval)

    async def start(self):
        """Inicia el servicio de sincronización en background"""
        if self.is_running:
            logger.warning("Servicio de sincronización ya está ejecutándose")
            return
            
        self.is_running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info("✅ Servicio de sincronización iniciado")

    async def stop(self):
        """Detiene el servicio de sincronización"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Servicio de sincronización detenido")

    async def force_sync(self) -> Dict[str, int]:
        """Ejecuta sincronización inmediata (sin esperar el intervalo)"""
        logger.info("Ejecutando sincronización manual...")
        return await self.sync_all_collections()

# Instancia global del servicio
auto_sync_service = AutoSyncService(sync_interval_seconds=60)  # Sincronizar cada minuto

async def get_sync_service():
    """Dependency injection para el servicio de sincronización"""
    return auto_sync_service