"""
Script de migración: Postgres -> MongoDB para features de ERP

Este script realiza los siguientes pasos:
- Crea (si no existen) las colecciones en MongoDB con índices útiles
- Ejecuta 3 consultas SQL en PostgreSQL para obtener features de:
  1) Postulantes
  2) Ofertas de trabajo
  3) Empresas
- Inserta/actualiza los documentos en MongoDB usando upsert

Uso recomendado:
  python -m scripts.migrate_erp_to_mongo
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
from decimal import Decimal
import uuid

from pymongo import UpdateOne, ASCENDING

from app.config.connection import db, mongodb


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("migration")


# =====================
# SQL QUERIES (3)
# =====================
SQL_POSTULANTES = """
SELECT 
  p.id                         AS postulante_id,
  p.anios_experiencia          AS anios_experiencia,
  p.nivel_educacion            AS nivel_educacion,
  p.habilidades                AS habilidades,
  p.idiomas                    AS idiomas,
  p.certificaciones            AS certificaciones,
  p.puesto_actual              AS puesto_actual,
  p.oferta_id                  AS oferta_id
FROM postulaciones p;
"""

SQL_OFERTAS = """
SELECT 
  o.id           AS oferta_id,
  o.titulo       AS titulo,
  o.salario      AS salario,
  o.ubicacion    AS ubicacion,
  o.requisitos   AS requisitos,
  o.empresa_id   AS empresa_id
FROM ofertas_trabajo o;
"""

SQL_EMPRESAS = """
SELECT 
  e.id       AS empresa_id,
  e.nombre   AS nombre,
  e.rubro    AS rubro
FROM empresas e;
"""


# =====================
# Helpers Mongo
# =====================
COLL_POSTULANTES = "candidates_features"
COLL_OFERTAS = "job_offers_features"
COLL_EMPRESAS = "companies_features"


async def ensure_collections_and_indexes():
    """Crea colecciones (si no existen) e índices en MongoDB."""
    await mongodb.connect()
    db_mongo = mongodb.get_database()

    # Crear explícitamente colecciones si no existen
    existing = await db_mongo.list_collection_names()
    for name in (COLL_EMPRESAS, COLL_OFERTAS, COLL_POSTULANTES):
        if name not in existing:
            await db_mongo.create_collection(name)
            logger.info(f"Creada colección Mongo: {name}")

    # Índices
    companies = mongodb.get_collection(COLL_EMPRESAS)
    offers = mongodb.get_collection(COLL_OFERTAS)
    candidates = mongodb.get_collection(COLL_POSTULANTES)

    await companies.create_index([("empresa_id", ASCENDING)], unique=True, name="uid_empresa")
    await offers.create_index([("oferta_id", ASCENDING)], unique=True, name="uid_oferta")
    await offers.create_index([("empresa_id", ASCENDING)], name="idx_oferta_empresa")
    await candidates.create_index([("postulante_id", ASCENDING)], unique=True, name="uid_postulante")
    await candidates.create_index([("oferta_id", ASCENDING)], name="idx_postulante_oferta")


def _now_iso() -> str:
    return datetime.now().isoformat() + "Z"


def _row_to_dict(row: Any) -> Dict[str, Any]:
    """Convierte un registro asyncpg a dict plano, normalizando tipos."""
    result = dict(row)
    
    # Convertir tipos problemáticos para MongoDB
    for key, value in result.items():
        if isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, uuid.UUID):
            result[key] = str(value)
        elif value is None:
            result[key] = None
    
    return result


async def fetch_postulantes(conn) -> List[Dict[str, Any]]:
    rows = await conn.fetch(SQL_POSTULANTES)
    return [_row_to_dict(r) for r in rows]


async def fetch_ofertas(conn) -> List[Dict[str, Any]]:
    rows = await conn.fetch(SQL_OFERTAS)
    return [_row_to_dict(r) for r in rows]


async def fetch_empresas(conn) -> List[Dict[str, Any]]:
    rows = await conn.fetch(SQL_EMPRESAS)
    return [_row_to_dict(r) for r in rows]


async def upsert_many(collection_name: str, id_field: str, docs: List[Dict[str, Any]]):
    """Realiza bulk upsert en Mongo utilizando id_field como clave única."""
    if not docs:
        logger.info(f"No hay documentos para {collection_name}")
        return 0

    coll = mongodb.get_collection(collection_name)
    ops = []
    for d in docs:
        # Normalizar IDs a string
        if d.get(id_field) is not None:
            d[id_field] = str(d[id_field])
        # timestamps
        d.setdefault("updated_at", _now_iso())
        d.setdefault("created_at", _now_iso())
        # Documento guardado con _id también como el id lógico
        ops.append(
            UpdateOne({"_id": d[id_field]}, {"$set": d}, upsert=True)
        )

    result = await coll.bulk_write(ops, ordered=False)
    upserts = (result.upserted_count or 0) + (result.modified_count or 0)
    logger.info(f"Upserts en {collection_name}: {upserts}")
    return upserts


async def migrate():
    logger.info("Iniciando migración Postgres -> Mongo…")

    # Conexiones
    pg_ok = await db.connect()
    mg_ok = await mongodb.connect()
    if not (pg_ok and mg_ok):
        raise RuntimeError("No se pudo conectar a Postgres o Mongo. Verifique sus credenciales en .env")

    # Colecciones/índices
    await ensure_collections_and_indexes()

    # Obtener datos desde Postgres
    async with await db.get_connection() as conn:
        empresas = await fetch_empresas(conn)
        ofertas = await fetch_ofertas(conn)
        postulantes = await fetch_postulantes(conn)

    logger.info(f"Empresas: {len(empresas)}, Ofertas: {len(ofertas)}, Postulantes: {len(postulantes)}")

    # Guardar en Mongo (bulk upsert)
    await upsert_many(COLL_EMPRESAS, "empresa_id", empresas)
    await upsert_many(COLL_OFERTAS, "oferta_id", ofertas)
    await upsert_many(COLL_POSTULANTES, "postulante_id", postulantes)

    logger.info("Migración completada ✅")


def main():
    asyncio.run(migrate())


if __name__ == "__main__":
    main()
