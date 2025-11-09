#!/usr/bin/env python3
"""
🔄 MIGRACIÓN DE DATOS POSTGRESQL A MONGODB
Script completo para migrar y transformar datos desde PostgreSQL a MongoDB
optimizado para machine learning semi-supervisado
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import os
import sys
import re
import unicodedata

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.extract_postgres_data import PostgreSQLDataExtractor
from scripts.create_mongo_collections_ml import MongoMLCollectionDesigner
from app.config.mongodb_connection import get_mongodb_sync, mongodb_connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgresToMongoMigrator:
    """Migrador de datos desde PostgreSQL a MongoDB con transformaciones ML"""
    
    def __init__(self):
        self.postgres_extractor = PostgreSQLDataExtractor()
        self.mongo_designer = MongoMLCollectionDesigner()
        self.db = None
        
        # Transformadores para features
        self.tfidf_skills = TfidfVectorizer(max_features=100, ngram_range=(1, 2), 
                                          stop_words=['y', 'e', 'o', 'de', 'la', 'el'])
        self.tfidf_requirements = TfidfVectorizer(max_features=100, ngram_range=(1, 2), 
                                                stop_words=['y', 'e', 'o', 'de', 'la', 'el'])
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Mapeo de estados para target variable
        self.estado_mapping = {
            'aceptado': 1,
            'contratado': 1,
            'entrevista': 1,
            'rechazado': 0,
            'descartado': 0,
            'en revisión': -1,  # No etiquetado
            'pendiente': -1,    # No etiquetado
            'en proceso': -1    # No etiquetado
        }
        
        # Mapeo de niveles educativos
        self.education_mapping = {
            'primaria': 1,
            'secundaria': 2,
            'técnico': 3,
            'universitario': 4,
            'licenciatura': 4,
            'ingeniería': 5,
            'maestría': 6,
            'doctorado': 7,
            'phd': 7
        }
    
    def connect(self):
        """Conectar a MongoDB"""
        try:
            self.mongo_designer.connect()
            self.db = get_mongodb_sync()
            logger.info("✅ Conectado a MongoDB para migración")
        except Exception as e:
            logger.error(f"❌ Error conectando a MongoDB: {e}")
            raise
    
    def disconnect(self):
        """Desconectar de MongoDB"""
        try:
            self.mongo_designer.disconnect()
        except Exception as e:
            logger.error(f"❌ Error desconectando: {e}")
    
    def normalize_text(self, text: str) -> str:
        """Normaliza texto removiendo acentos y limpiando"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower().strip()
        
        # Remover acentos
        text = unicodedata.normalize('NFKD', text)
        text = "".join([c for c in text if not unicodedata.combining(c)])
        
        # Limpiar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def parse_list_field(self, field_str: str, separator: str = ',') -> List[str]:
        """Convierte string separado por comas en lista limpia"""
        if pd.isna(field_str) or field_str == '':
            return []
        
        items = [item.strip() for item in str(field_str).split(separator)]
        items = [self.normalize_text(item) for item in items if item.strip()]
        
        return items
    
    def calculate_profile_completeness(self, candidate_data: Dict) -> float:
        """Calcula la completitud del perfil del candidato (0-1)"""
        required_fields = ['nombre', 'email', 'habilidades', 'anos_experiencia', 'nivel_educacion']
        optional_fields = ['telefono', 'idiomas', 'certificaciones', 'puesto_actual', 'url_cv']
        
        required_score = sum(1 for field in required_fields 
                           if candidate_data.get(field) and str(candidate_data[field]).strip())
        optional_score = sum(1 for field in optional_fields 
                           if candidate_data.get(field) and str(candidate_data[field]).strip())
        
        # Peso: 70% campos requeridos, 30% campos opcionales
        completeness = (required_score / len(required_fields)) * 0.7 + \
                      (optional_score / len(optional_fields)) * 0.3
        
        return round(completeness, 3)
    
    def extract_job_level(self, title: str, description: str) -> str:
        """Extrae el nivel del trabajo (junior, mid, senior)"""
        text = f"{title} {description}".lower()
        
        if any(keyword in text for keyword in ['junior', 'entry', 'trainee', 'inicial']):
            return 'junior'
        elif any(keyword in text for keyword in ['senior', 'lead', 'principal', 'expert']):
            return 'senior'
        else:
            return 'mid'
    
    def extract_work_modality(self, description: str, location: str) -> str:
        """Extrae la modalidad de trabajo"""
        text = f"{description} {location}".lower()
        
        if any(keyword in text for keyword in ['remoto', 'remote', 'home office', 'teletrabajo']):
            return 'remoto'
        elif any(keyword in text for keyword in ['híbrido', 'hybrid', 'mixto']):
            return 'hibrido'
        else:
            return 'presencial'
    
    def process_candidates(self, postulaciones_df: pd.DataFrame) -> List[Dict]:
        """Procesa candidatos únicos desde las postulaciones"""
        logger.info("👥 Procesando candidatos...")
        
        # Obtener candidatos únicos por email
        candidates_df = postulaciones_df.drop_duplicates(subset=['email']).copy()
        
        candidates = []
        skills_texts = []
        
        # Primera pasada: recopilar textos para vectorización
        for _, row in candidates_df.iterrows():
            skills_text = self.normalize_text(str(row.get('habilidades', '')))
            skills_texts.append(skills_text)
        
        # Vectorizar habilidades
        if skills_texts and any(text.strip() for text in skills_texts):
            skills_vectors = self.tfidf_skills.fit_transform(skills_texts)
        else:
            skills_vectors = np.zeros((len(skills_texts), 100))
        
        # Segunda pasada: crear documentos de candidatos
        for idx, (_, row) in enumerate(candidates_df.iterrows()):
            try:
                habilidades_list = self.parse_list_field(row.get('habilidades', ''))
                idiomas_list = self.parse_list_field(row.get('idiomas', ''))
                certificaciones_list = self.parse_list_field(row.get('certificaciones', ''))
                
                # Normalizar nivel educativo
                nivel_edu = self.normalize_text(str(row.get('nivel_educacion', '')))
                nivel_edu_encoded = self.education_mapping.get(nivel_edu, 0)
                
                # Calcular experiencia normalizada (0-1, máximo 40 años)
                anos_exp = int(row.get('anios_experiencia', 0))
                exp_normalizada = min(anos_exp / 40.0, 1.0)
                
                candidate_data = {
                    'candidate_id': str(row.get('postulacion_id', '')),  # Usar ID de postulación como referencia
                    'nombre': str(row.get('candidato_nombre', '')),
                    'email': str(row.get('email', '')),
                    'telefono': str(row.get('telefono', '')),
                    'anos_experiencia': anos_exp,
                    'nivel_educacion': nivel_edu,
                    'habilidades': habilidades_list,
                    'habilidades_raw': str(row.get('habilidades', '')),
                    'idiomas': idiomas_list,
                    'idiomas_raw': str(row.get('idiomas', '')),
                    'certificaciones': certificaciones_list,
                    'certificaciones_raw': str(row.get('certificaciones', '')),
                    'puesto_actual': str(row.get('puesto_actual', '')),
                    'url_cv': str(row.get('url_cv', '')),
                    
                    # Features procesadas
                    'features': {
                        'experiencia_normalizada': exp_normalizada,
                        'nivel_educacion_encoded': nivel_edu_encoded,
                        'num_habilidades': len(habilidades_list),
                        'num_idiomas': len(idiomas_list),
                        'num_certificaciones': len(certificaciones_list),
                        'skills_vector': skills_vectors[idx].toarray().flatten().tolist() if hasattr(skills_vectors, 'toarray') else [0.0] * 100,
                        'profile_completeness': 0.0  # Se calculará después
                    },
                    
                    # Labels ML (se determinarán después)
                    'ml_label': 'unlabeled',
                    'ml_confidence': 0.0,
                    'label_source': 'initial',
                    
                    # Metadatos
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'version': 1
                }
                
                # Calcular completitud del perfil
                candidate_data['features']['profile_completeness'] = self.calculate_profile_completeness(candidate_data)
                
                candidates.append(candidate_data)
                
            except Exception as e:
                logger.error(f"❌ Error procesando candidato {row.get('email', 'unknown')}: {e}")
        
        logger.info(f"✅ Procesados {len(candidates)} candidatos únicos")
        return candidates
    
    def process_job_offers(self, postulaciones_df: pd.DataFrame) -> List[Dict]:
        """Procesa ofertas de trabajo únicas"""
        logger.info("💼 Procesando ofertas de trabajo...")
        
        # Obtener ofertas únicas
        offers_df = postulaciones_df.drop_duplicates(subset=['oferta_id']).copy()
        
        offers = []
        requirements_texts = []
        
        # Primera pasada: recopilar textos para vectorización
        for _, row in offers_df.iterrows():
            req_text = self.normalize_text(str(row.get('oferta_requisitos', '')))
            requirements_texts.append(req_text)
        
        # Vectorizar requisitos
        if requirements_texts and any(text.strip() for text in requirements_texts):
            req_vectors = self.tfidf_requirements.fit_transform(requirements_texts)
        else:
            req_vectors = np.zeros((len(requirements_texts), 100))
        
        # Segunda pasada: crear documentos de ofertas
        for idx, (_, row) in enumerate(offers_df.iterrows()):
            try:
                # Procesar fecha de publicación
                fecha_pub = row.get('fecha_publicacion')
                if pd.isna(fecha_pub):
                    fecha_pub_dt = datetime.now(timezone.utc)
                else:
                    try:
                        fecha_pub_dt = pd.to_datetime(fecha_pub).tz_localize(timezone.utc)
                    except:
                        fecha_pub_dt = datetime.now(timezone.utc)
                
                # Calcular días desde publicación
                dias_desde_pub = (datetime.now(timezone.utc) - fecha_pub_dt).days
                
                # Normalizar salario (0-1, máximo 100,000)
                salario = float(row.get('oferta_salario', 0)) if not pd.isna(row.get('oferta_salario')) else 0
                salario_norm = min(salario / 100000.0, 1.0) if salario > 0 else 0.0
                
                # Extraer características del trabajo
                titulo = str(row.get('oferta_titulo', ''))
                descripcion = str(row.get('oferta_descripcion', ''))
                
                offer_data = {
                    'offer_id': str(row.get('oferta_id', '')),
                    'titulo': titulo,
                    'descripcion': descripcion,
                    'salario': salario,
                    'ubicacion': str(row.get('oferta_ubicacion', '')),
                    'requisitos': str(row.get('oferta_requisitos', '')),
                    'fecha_publicacion': fecha_pub_dt,
                    'activa': dias_desde_pub <= 90,  # Activa si fue publicada en los últimos 90 días
                    
                    # Información de empresa
                    'empresa_id': str(row.get('empresa_id', '')),
                    'empresa_nombre': str(row.get('empresa_nombre', '')),
                    'empresa_rubro': str(row.get('empresa_rubro', '')),
                    
                    # Features procesadas
                    'features': {
                        'requisitos_vector': req_vectors[idx].toarray().flatten().tolist() if hasattr(req_vectors, 'toarray') else [0.0] * 100,
                        'salario_normalizado': salario_norm,
                        'dias_desde_publicacion': dias_desde_pub,
                        'nivel_requisitos': self.extract_job_level(titulo, descripcion),
                        'tipo_contrato': 'indefinido',  # Podría inferirse del texto
                        'modalidad_trabajo': self.extract_work_modality(descripcion, str(row.get('oferta_ubicacion', '')))
                    },
                    
                    # Metadatos
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'version': 1
                }
                
                offers.append(offer_data)
                
            except Exception as e:
                logger.error(f"❌ Error procesando oferta {row.get('oferta_id', 'unknown')}: {e}")
        
        logger.info(f"✅ Procesadas {len(offers)} ofertas únicas")
        return offers
    
    def calculate_compatibility_features(self, candidate: Dict, offer: Dict) -> Dict:
        """Calcula features de compatibilidad entre candidato y oferta"""
        
        # Similitud de habilidades (usando vectores TF-IDF)
        candidate_skills_vector = np.array(candidate['features']['skills_vector'])
        offer_req_vector = np.array(offer['features']['requisitos_vector'])
        
        # Similitud coseno
        if np.linalg.norm(candidate_skills_vector) > 0 and np.linalg.norm(offer_req_vector) > 0:
            skill_similarity = np.dot(candidate_skills_vector, offer_req_vector) / \
                             (np.linalg.norm(candidate_skills_vector) * np.linalg.norm(offer_req_vector))
        else:
            skill_similarity = 0.0
        
        # Compatibilidad de experiencia
        exp_match = 1.0  # Simplificado por ahora
        
        # Compatibilidad educativa
        edu_match = 1.0 if candidate['features']['nivel_educacion_encoded'] >= 3 else 0.5
        
        # Coincidencia de ubicación (simplificado)
        location_match = True  # Por ahora asumir que sí
        
        # Expectativa salarial (simplificado)
        salary_match = 0.8  # Por ahora valor fijo
        
        # Score general de compatibilidad
        overall_compatibility = (
            skill_similarity * 0.4 +
            exp_match * 0.3 +
            edu_match * 0.2 +
            salary_match * 0.1
        )
        
        return {
            'skill_match_score': round(skill_similarity, 3),
            'experience_match': round(exp_match, 3),
            'education_match': round(edu_match, 3),
            'location_match': location_match,
            'salary_expectation_match': round(salary_match, 3),
            'overall_compatibility': round(overall_compatibility, 3)
        }
    
    def process_applications(self, postulaciones_df: pd.DataFrame, candidates: List[Dict], offers: List[Dict]) -> List[Dict]:
        """Procesa postulaciones con features de compatibilidad"""
        logger.info("📝 Procesando postulaciones...")
        
        # Crear mapas para búsqueda rápida
        candidates_map = {c['candidate_id']: c for c in candidates}
        offers_map = {o['offer_id']: o for o in offers}
        
        applications = []
        
        for _, row in postulaciones_df.iterrows():
            try:
                candidate_id = str(row.get('postulacion_id', ''))
                offer_id = str(row.get('oferta_id', ''))
                
                # Buscar candidato y oferta
                candidate = candidates_map.get(candidate_id)
                offer = offers_map.get(offer_id)
                
                if not candidate or not offer:
                    logger.warning(f"⚠️ Candidato u oferta no encontrada para postulación {candidate_id}-{offer_id}")
                    continue
                
                # Procesar fecha de postulación
                fecha_post = row.get('fecha_postulacion')
                if pd.isna(fecha_post):
                    fecha_post_dt = datetime.now(timezone.utc)
                else:
                    try:
                        fecha_post_dt = pd.to_datetime(fecha_post).tz_localize(timezone.utc)
                    except:
                        fecha_post_dt = datetime.now(timezone.utc)
                
                # Determinar etiqueta ML
                estado = self.normalize_text(str(row.get('estado', '')))
                ml_target = self.estado_mapping.get(estado, -1)
                is_labeled = ml_target != -1
                
                # Calcular features de compatibilidad
                compatibility_features = self.calculate_compatibility_features(candidate, offer)
                
                application_data = {
                    'application_id': f"{candidate_id}_{offer_id}",
                    'candidate_id': candidate_id,
                    'offer_id': offer_id,
                    'fecha_postulacion': fecha_post_dt,
                    'estado': estado,
                    
                    # Features de compatibilidad
                    'compatibility_features': compatibility_features,
                    
                    # Labels y predicciones ML
                    'ml_target': ml_target,
                    'ml_prediction': -1,  # Se llenará con el modelo
                    'ml_probability': 0.0,
                    'ml_confidence': 'unknown',
                    
                    # Semi-supervised learning
                    'is_labeled': is_labeled,
                    'label_quality': 'high' if is_labeled else 'unlabeled',
                    
                    # Metadatos
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'processed_at': datetime.now(timezone.utc),
                    'version': 1
                }
                
                applications.append(application_data)
                
            except Exception as e:
                logger.error(f"❌ Error procesando postulación: {e}")
        
        logger.info(f"✅ Procesadas {len(applications)} postulaciones")
        
        # Estadísticas de etiquetas
        labeled_count = sum(1 for app in applications if app['is_labeled'])
        unlabeled_count = len(applications) - labeled_count
        
        logger.info(f"📊 Datos etiquetados: {labeled_count}, No etiquetados: {unlabeled_count}")
        logger.info(f"📊 Ratio etiquetado: {labeled_count/len(applications)*100:.1f}%")
        
        return applications
    
    def process_interviews(self, entrevistas_df: pd.DataFrame) -> List[Dict]:
        """Procesa entrevistas y evaluaciones"""
        if entrevistas_df.empty:
            logger.info("ℹ️ No hay datos de entrevistas para procesar")
            return []
        
        logger.info("🎤 Procesando entrevistas...")
        
        interviews = []
        
        # Agrupar evaluaciones por entrevista
        interviews_grouped = entrevistas_df.groupby('entrevista_id')
        
        for interview_id, group in interviews_grouped:
            try:
                # Datos de la primera fila (datos de entrevista)
                first_row = group.iloc[0]
                
                # Procesar fecha
                fecha = first_row.get('entrevista_fecha')
                if pd.isna(fecha):
                    fecha_dt = datetime.now(timezone.utc)
                else:
                    try:
                        fecha_dt = pd.to_datetime(fecha).tz_localize(timezone.utc)
                    except:
                        fecha_dt = datetime.now(timezone.utc)
                
                # Recopilar evaluaciones
                evaluaciones = []
                calificaciones = []
                
                for _, row in group.iterrows():
                    if not pd.isna(row.get('evaluacion_id')):
                        eval_data = {
                            'evaluacion_id': str(row.get('evaluacion_id', '')),
                            'calificacion_tecnica': float(row.get('calificacion_tecnica', 0)) if not pd.isna(row.get('calificacion_tecnica')) else 0.0,
                            'calificacion_actitud': float(row.get('calificacion_actitud', 0)) if not pd.isna(row.get('calificacion_actitud')) else 0.0,
                            'calificacion_general': float(row.get('calificacion_general', 0)) if not pd.isna(row.get('calificacion_general')) else 0.0,
                            'comentarios': str(row.get('evaluacion_comentarios', ''))
                        }
                        evaluaciones.append(eval_data)
                        calificaciones.extend([
                            eval_data['calificacion_tecnica'],
                            eval_data['calificacion_actitud'],
                            eval_data['calificacion_general']
                        ])
                
                # Calcular métricas
                promedio_calificaciones = np.mean([c for c in calificaciones if c > 0]) if calificaciones else 0.0
                
                # Calcular cobertura de objetivos (simplificado)
                objetivos_totales = str(first_row.get('objetivos_totales', ''))
                objetivos_cubiertos = str(first_row.get('objetivos_cubiertos', ''))
                cobertura_objetivos = 0.8 if objetivos_cubiertos and objetivos_totales else 0.0
                
                # Normalizar duración (0-1, máximo 240 minutos)
                duracion = int(first_row.get('duracion_min', 0)) if not pd.isna(first_row.get('duracion_min')) else 0
                duracion_norm = min(duracion / 240.0, 1.0) if duracion > 0 else 0.0
                
                interview_data = {
                    'interview_id': str(interview_id),
                    'application_id': str(first_row.get('postulacion_id', '')),
                    'fecha': fecha_dt,
                    'duracion_min': duracion,
                    'objetivos_totales': objetivos_totales,
                    'objetivos_cubiertos': objetivos_cubiertos,
                    'entrevistador': str(first_row.get('entrevistador', '')),
                    
                    # Evaluaciones
                    'evaluaciones': evaluaciones,
                    
                    # Features derivadas
                    'features': {
                        'promedio_calificaciones': round(promedio_calificaciones, 3),
                        'cobertura_objetivos': round(cobertura_objetivos, 3),
                        'duracion_normalizada': round(duracion_norm, 3),
                        'sentiment_comentarios': 0.5  # Por ahora neutral
                    },
                    
                    # Metadatos
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc)
                }
                
                interviews.append(interview_data)
                
            except Exception as e:
                logger.error(f"❌ Error procesando entrevista {interview_id}: {e}")
        
        logger.info(f"✅ Procesadas {len(interviews)} entrevistas")
        return interviews
    
    def insert_documents(self, collection_name: str, documents: List[Dict], batch_size: int = 1000):
        """Insertar documentos en MongoDB por lotes"""
        if not documents:
            logger.info(f"ℹ️ No hay documentos para insertar en {collection_name}")
            return
        
        logger.info(f"📥 Insertando {len(documents)} documentos en {collection_name}...")
        
        collection = self.db[collection_name]
        
        # Limpiar colección existente
        result = collection.delete_many({})
        if result.deleted_count > 0:
            logger.info(f"🗑️ Eliminados {result.deleted_count} documentos existentes")
        
        # Insertar por lotes
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                collection.insert_many(batch, ordered=False)
                logger.info(f"  ✅ Insertado lote {i//batch_size + 1}: {len(batch)} documentos")
            except Exception as e:
                logger.error(f"❌ Error insertando lote {i//batch_size + 1}: {e}")
        
        # Verificar inserción
        final_count = collection.count_documents({})
        logger.info(f"✅ Total en {collection_name}: {final_count} documentos")
    
    async def migrate_data(self):
        """Ejecutar migración completa"""
        logger.info("🚀 Iniciando migración de datos PostgreSQL → MongoDB")
        
        try:
            # 1. Conectar a MongoDB
            self.connect()
            
            # 2. Crear colecciones si no existen
            self.mongo_designer.create_collections()
            
            # 3. Extraer datos de PostgreSQL
            logger.info("📊 Extrayendo datos de PostgreSQL...")
            postgres_data = await self.postgres_extractor.extract_all_data()
            
            postulaciones_df = postgres_data['postulaciones']
            entrevistas_df = postgres_data['entrevistas']
            
            if postulaciones_df.empty:
                logger.error("❌ No se encontraron datos de postulaciones")
                return
            
            # 4. Procesar y transformar datos
            candidates = self.process_candidates(postulaciones_df)
            offers = self.process_job_offers(postulaciones_df)
            applications = self.process_applications(postulaciones_df, candidates, offers)
            interviews = self.process_interviews(entrevistas_df)
            
            # 5. Insertar en MongoDB
            self.insert_documents('ml_candidates', candidates)
            self.insert_documents('ml_job_offers', offers)
            self.insert_documents('ml_applications', applications)
            self.insert_documents('ml_interviews', interviews)
            
            # 6. Crear registro de migración
            migration_record = {
                'migration_id': f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'migration_date': datetime.now(timezone.utc),
                'source': 'postgresql',
                'target': 'mongodb',
                'status': 'completed',
                'statistics': {
                    'candidates_migrated': len(candidates),
                    'offers_migrated': len(offers),
                    'applications_migrated': len(applications),
                    'interviews_migrated': len(interviews),
                    'labeled_applications': sum(1 for app in applications if app['is_labeled']),
                    'unlabeled_applications': sum(1 for app in applications if not app['is_labeled'])
                }
            }
            
            # Insertar registro de migración
            migration_collection = self.db.get_collection('migration_history')
            migration_collection.insert_one(migration_record)
            
            logger.info("🎉 Migración completada exitosamente")
            logger.info(f"📊 Resumen:")
            logger.info(f"  👥 Candidatos: {len(candidates)}")
            logger.info(f"  💼 Ofertas: {len(offers)}")
            logger.info(f"  📝 Postulaciones: {len(applications)}")
            logger.info(f"  🎤 Entrevistas: {len(interviews)}")
            logger.info(f"  🏷️ Datos etiquetados: {migration_record['statistics']['labeled_applications']}")
            logger.info(f"  ❓ Datos no etiquetados: {migration_record['statistics']['unlabeled_applications']}")
            
        except Exception as e:
            logger.error(f"❌ Error en migración: {e}")
            raise
        
        finally:
            self.disconnect()


async def main():
    """Función principal"""
    migrator = PostgresToMongoMigrator()
    
    try:
        await migrator.migrate_data()
        logger.info("✅ Migración completada exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error en migración: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())