"""
Creador de colecciones en MongoDB para el proyecto.

Ejecuta este script para crear las colecciones e índices necesarios
sin realizar la migración de datos desde Postgres.

Uso:
  python -m scripts.create_mongo_collections
"""

import asyncio
import logging

# Importar ASCENDING de forma más robusta
try:
    from pymongo import ASCENDING
except ImportError:
    # Fallback si pymongo no está disponible
    ASCENDING = 1

from app.config.connection import mongodb


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("create_mongo_collections")

COLL_POSTULANTES = "candidates_features"
COLL_OFERTAS = "job_offers_features"
COLL_EMPRESAS = "companies_features"


async def ensure_collections_and_indexes():
    await mongodb.connect()
    db = mongodb.get_database()

    existing = await db.list_collection_names()
    for name in (COLL_EMPRESAS, COLL_OFERTAS, COLL_POSTULANTES):
        if name not in existing:
            await db.create_collection(name)
            logger.info(f"Creada colección: {name}")
        else:
            logger.info(f"Colección ya existe: {name}")

    companies = mongodb.get_collection(COLL_EMPRESAS)
    offers = mongodb.get_collection(COLL_OFERTAS)
    candidates = mongodb.get_collection(COLL_POSTULANTES)

    # Índices
    await companies.create_index([("empresa_id", ASCENDING)], unique=True, name="uid_empresa")
    await offers.create_index([("oferta_id", ASCENDING)], unique=True, name="uid_oferta")
    await offers.create_index([("empresa_id", ASCENDING)], name="idx_oferta_empresa")
    await candidates.create_index([("postulante_id", ASCENDING)], unique=True, name="uid_postulante")
    await candidates.create_index([("oferta_id", ASCENDING)], name="idx_postulante_oferta")

    logger.info("Índices creados/asegurados")


def main():
    asyncio.run(ensure_collections_and_indexes())
    logger.info("Colecciones listas ✅")


if __name__ == "__main__":
    main()
