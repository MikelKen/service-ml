"""
Resolvers de GraphQL para operaciones de Machine Learning
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import time

from app.graphql.types.ml_types import (
    CompatibilityPredictionInput, BatchCompatibilityInput, TopCandidatesInput,
    CustomCompatibilityPredictionInput, CustomCandidateData, CustomJobOfferData,
    CompatibilityPrediction, BatchCompatibilityResult, ModelTrainingResult,
    ModelInfo, FeatureImportance, ModelFeatureImportance, PredictionExplanation,
    TrainingDataSummary, ModelPerformanceMetrics, TrainingConfigInput,
    TrainingMetrics, DataSummary, ModelMetrics, PredictionFactors, MetadataInfo
)

from app.ml.models.predictor import compatibility_predictor
from app.ml.training.model_trainer import compatibility_trainer, train_compatibility_model
from app.ml.data.data_extractor import data_extractor

logger = logging.getLogger(__name__)


async def predict_compatibility(input_data: CompatibilityPredictionInput) -> CompatibilityPrediction:
    """Predice compatibilidad entre un candidato y una oferta"""
    
    try:
        # Realizar predicción
        result = compatibility_predictor.predict_compatibility(
            input_data.candidate_id, 
            input_data.offer_id
        )
        
        return CompatibilityPrediction(
            candidate_id=result['candidate_id'],
            offer_id=result['offer_id'],
            probability=result['probability'],
            prediction=result['prediction'],
            confidence=result['confidence'],
            model_used=result.get('model_used'),
            prediction_date=result.get('prediction_date'),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error en predicción de compatibilidad: {e}")
        return CompatibilityPrediction(
            candidate_id=input_data.candidate_id,
            offer_id=input_data.offer_id,
            probability=0.0,
            prediction=False,
            confidence='Error',
            error=str(e)
        )


async def predict_custom_compatibility(input_data: CustomCompatibilityPredictionInput) -> CompatibilityPrediction:
    """Predice compatibilidad con datos personalizados (no desde BD)"""
    
    try:
        import pandas as pd
        from datetime import datetime
        from app.ml.preprocessing.mongo_preprocessor import mongo_preprocessor
        
        # Convertir datos de entrada al formato esperado
        candidate_data = {
            'candidate_id': 'custom_candidate',
            'years_experience': input_data.candidate_data.anios_experiencia,
            'education_level': input_data.candidate_data.nivel_educacion,
            'skills': input_data.candidate_data.habilidades,
            'languages': input_data.candidate_data.idiomas or '',
            'certifications': input_data.candidate_data.certificaciones or '',
            'current_position': input_data.candidate_data.puesto_actual or '',
            
            'offer_id': 'custom_offer',
            'job_title': input_data.offer_data.titulo,
            'salary': input_data.offer_data.salario,
            'location': input_data.offer_data.ubicacion,
            'requirements': input_data.offer_data.requisitos,
            'company_id': 'custom_company',
            
            'created_at': datetime.now().isoformat()
        }
        
        # Convertir a DataFrame
        df = pd.DataFrame([candidate_data])
        
        # Preprocessar datos
        df_processed = mongo_preprocessor.preprocess_data(df, fit_transformers=False)
        
        # Excluir columnas de ID
        exclude_columns = ['candidate_id', 'offer_id', 'created_at']
        feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
        X = df_processed[feature_columns]
        
        # Realizar predicción
        probability = compatibility_predictor.model.predict_proba(X)[0, 1]
        prediction = compatibility_predictor.model.predict(X)[0]
        
        # Determinar nivel de confianza
        confidence = compatibility_predictor._calculate_confidence(probability)
        
        # Generar análisis descriptivo detallado
        analysis = _generate_detailed_analysis(
            probability, prediction, confidence,
            input_data.candidate_data, input_data.offer_data
        )
        
        # Usar probabilidad ajustada si está disponible
        final_probability = analysis.get('adjusted_probability', probability)
        
        return CompatibilityPrediction(
            candidate_id='custom_candidate',
            offer_id='custom_offer',
            probability=float(final_probability),
            prediction=bool(final_probability >= 0.5),
            confidence=confidence,
            model_used=compatibility_predictor.model_name,
            prediction_date=datetime.now().isoformat(),
            
            # Información descriptiva adicional
            probability_percentage=f"{final_probability*100:.2f}%",
            compatibility_level=analysis['compatibility_level'],
            recommendation=analysis['recommendation'],
            decision_factors=analysis['decision_factors'],
            
            # Análisis detallado
            strengths=analysis['strengths'],
            weaknesses=analysis['weaknesses'],
            suggestions=analysis['suggestions'],
            
            # Información técnica
            confidence_score=float(final_probability),
            summary=analysis['summary'],
            detailed_analysis=analysis['detailed_analysis']
        )
        
    except Exception as e:
        logger.error(f"Error en predicción personalizada: {e}")
        return CompatibilityPrediction(
            candidate_id='custom_candidate',
            offer_id='custom_offer',
            probability=0.0,
            prediction=False,
            confidence='Error',
            error=str(e),
            summary="Error en el análisis de compatibilidad",
            detailed_analysis=f"No se pudo completar el análisis debido a: {str(e)}"
        )


def _generate_detailed_analysis(probability, prediction, confidence, candidate_data, offer_data):
    """Genera análisis descriptivo detallado con lógica mejorada para candidatos junior"""
    
    # Análisis básico del candidato
    years_exp = candidate_data.anios_experiencia
    education = candidate_data.nivel_educacion.lower()
    skills = candidate_data.habilidades.lower()
    certifications = candidate_data.certificaciones
    languages = candidate_data.idiomas.lower() if candidate_data.idiomas else ''
    current_position = candidate_data.puesto_actual.lower() if candidate_data.puesto_actual else ''
    
    # Análisis de la oferta
    job_title = offer_data.titulo.lower()
    requirements = offer_data.requisitos.lower()
    salary = offer_data.salario
    
    # Factores de análisis mejorados
    is_junior = years_exp <= 2
    has_technical_education = any(word in education for word in ['sistem', 'informatic', 'computac', 'software', 'ingenier'])
    has_certifications = certifications and certifications.lower() not in ['', 'sin certificacion', 'ninguna']
    has_relevant_skills = any(skill in skills for skill in ['python', 'javascript', 'java', 'react', 'node', 'django', 'spring'])
    is_junior_position = 'junior' in job_title or 'junior' in requirements
    has_language_advantage = 'inglés' in languages or 'english' in languages
    
    # Lógica mejorada para TODOS los candidatos (no solo junior)
    adjusted_probability = probability
    applied_bonuses = []
    
    # === BONIFICACIONES GENERALES ===
    
    # 1. Educación técnica alineada
    if has_technical_education:
        bonus = 0.25  # +25% por educación técnica relevante
        adjusted_probability += bonus
        applied_bonuses.append(f"Educación técnica relevante (+{bonus*100:.0f}%)")
    
    # 2. Skills altamente relevantes
    if has_relevant_skills:
        # Contar skills relevantes
        relevant_skills_count = sum(1 for skill in ['python', 'javascript', 'java', 'react', 'node', 'django', 'spring', 'angular', 'php', 'laravel', '.net', 'c#'] if skill in skills)
        if relevant_skills_count >= 3:
            bonus = 0.30  # +30% por múltiples skills relevantes
            adjusted_probability += bonus
            applied_bonuses.append(f"Stack tecnológico alineado (+{bonus*100:.0f}%)")
        elif relevant_skills_count >= 1:
            bonus = 0.15  # +15% por algunas skills relevantes
            adjusted_probability += bonus
            applied_bonuses.append(f"Tecnologías relevantes (+{bonus*100:.0f}%)")
    
    # 3. Experiencia apropiada para el nivel
    experience_bonus = 0
    if 'senior' in job_title and years_exp >= 5:
        experience_bonus = 0.25  # +25% senior con experiencia senior
        applied_bonuses.append(f"Experiencia senior apropiada (+{experience_bonus*100:.0f}%)")
    elif 'junior' in job_title and years_exp <= 3:
        experience_bonus = 0.20  # +20% junior con experiencia apropiada
        applied_bonuses.append(f"Experiencia junior apropiada (+{experience_bonus*100:.0f}%)")
    elif not 'senior' in job_title and not 'junior' in job_title and 2 <= years_exp <= 7:
        experience_bonus = 0.15  # +15% experiencia media apropiada
        applied_bonuses.append(f"Experiencia media apropiada (+{experience_bonus*100:.0f}%)")
    
    adjusted_probability += experience_bonus
    
    # 4. Certificaciones relevantes
    if has_certifications:
        # Bonus extra si las certificaciones son muy relevantes
        cert_keywords = ['aws', 'azure', 'react', 'angular', 'java', 'python', 'docker', 'kubernetes', 'oracle', 'microsoft']
        relevant_certs = sum(1 for keyword in cert_keywords if keyword in certifications.lower())
        if relevant_certs >= 2:
            bonus = 0.20  # +20% por certificaciones múltiples relevantes
            applied_bonuses.append(f"Certificaciones profesionales múltiples (+{bonus*100:.0f}%)")
        else:
            bonus = 0.10  # +10% por certificaciones
            applied_bonuses.append(f"Certificaciones profesionales (+{bonus*100:.0f}%)")
        adjusted_probability += bonus
    
    # 5. Match perfecto de tecnologías (bonus especial)
    job_tech_keywords = []
    if 'full stack' in job_title: job_tech_keywords.extend(['javascript', 'react', 'node', 'python', 'java'])
    if 'backend' in job_title: job_tech_keywords.extend(['python', 'java', 'node', 'django', 'spring'])
    if 'frontend' in job_title: job_tech_keywords.extend(['react', 'angular', 'javascript', 'html', 'css'])
    if 'php' in job_title or 'php' in requirements: job_tech_keywords.extend(['php', 'laravel'])
    if 'java' in requirements: job_tech_keywords.extend(['java', 'spring'])
    if 'python' in requirements: job_tech_keywords.extend(['python', 'django'])
    
    if job_tech_keywords:
        matching_techs = sum(1 for tech in job_tech_keywords if tech in skills)
        if matching_techs >= 3:
            bonus = 0.35  # +35% por match perfecto de tecnologías
            adjusted_probability += bonus
            applied_bonuses.append(f"Match perfecto de tecnologías (+{bonus*100:.0f}%)")
        elif matching_techs >= 2:
            bonus = 0.20  # +20% por buen match de tecnologías
            adjusted_probability += bonus
            applied_bonuses.append(f"Buen match tecnológico (+{bonus*100:.0f}%)")
    
    # 6. Idiomas
    if has_language_advantage:
        bonus = 0.08  # +8% por ventaja idiomática
        adjusted_probability += bonus
        applied_bonuses.append(f"Ventaja idiomática (+{bonus*100:.0f}%)")
    
    # === BONIFICACIONES ESPECÍFICAS PARA JUNIOR ===
    if is_junior and is_junior_position:
        bonus = 0.15  # +15% extra por match de nivel junior
        adjusted_probability += bonus
        applied_bonuses.append(f"Match de nivel junior (+{bonus*100:.0f}%)")
    
    # === PENALIZACIONES ===
    
    # Penalización por falta de experiencia técnica
    if years_exp == 0 and not is_junior_position:
        penalty = 0.30  # -30% por falta de experiencia para puesto no junior
        adjusted_probability -= penalty
        applied_bonuses.append(f"Sin experiencia para puesto senior (-{penalty*100:.0f}%)")
    
    # Penalización por educación no técnica (solo para seniors)
    if not has_technical_education and not is_junior and 'desarrollador' in job_title:
        penalty = 0.15  # -15% por educación no técnica en puesto senior
        adjusted_probability -= penalty
        applied_bonuses.append(f"Educación no técnica (-{penalty*100:.0f}%)")
    
    # Limitar probabilidad ajustada entre 0.05 y 0.98
    adjusted_probability = max(0.05, min(adjusted_probability, 0.98))
    
    # Usar probabilidad ajustada para candidatos junior
    # Usar siempre la probabilidad ajustada
    final_probability = adjusted_probability
    
    # Determinar nivel de compatibilidad con probabilidad final mejorada
    if final_probability >= 0.80:
        compatibility_level = "🟢 EXCELENTE COMPATIBILIDAD"
        level_desc = "Match perfecto"
        recommendation = f"🎯 CANDIDATO IDEAL: Con {final_probability*100:.1f}% de compatibilidad, es un match perfecto. Proceder inmediatamente con contratación."
    elif final_probability >= 0.65:
        compatibility_level = "🟢 ALTA COMPATIBILIDAD"
        level_desc = "Excelente match"
        recommendation = f"✅ ALTAMENTE RECOMENDADO: Con {final_probability*100:.1f}% de compatibilidad, proceder con entrevista final."
    elif final_probability >= 0.50:
        compatibility_level = "🟡 COMPATIBILIDAD MODERADA"
        level_desc = "Buen potencial"
        recommendation = f"⚡ BUEN CANDIDATO: Con {final_probability*100:.1f}% de compatibilidad, continuar con evaluación técnica."
    elif final_probability >= 0.35:
        compatibility_level = "🟠 COMPATIBILIDAD BAJA-MEDIA"
        level_desc = "Requiere evaluación"
        recommendation = f"⚠️ EVALUACIÓN REQUERIDA: {final_probability*100:.1f}% de compatibilidad sugiere revisar requisitos específicos."
    else:
        compatibility_level = "🔴 COMPATIBILIDAD BAJA"
        level_desc = "No recomendado"
        recommendation = f"❌ NO RECOMENDADO: {final_probability*100:.1f}% de compatibilidad indica desajuste significativo."
    
    # Agregar información sobre bonificaciones aplicadas
    if applied_bonuses:
        bonus_text = " Bonificaciones aplicadas: " + ", ".join(applied_bonuses)
        recommendation += bonus_text
    
    # Analizar fortalezas del candidato (mejorado)
    strengths = []
    
    # Experiencia con análisis más matizado
    if years_exp >= 5:
        strengths.append(f"💼 Experiencia sólida: {years_exp} años en el campo")
    elif years_exp >= 2:
        strengths.append(f"💼 Experiencia moderada: {years_exp} años de desarrollo profesional")
    elif years_exp >= 1:
        strengths.append(f"🌱 Experiencia inicial: {years_exp} año(s) con potencial de crecimiento")
    
    # Educación técnica
    if has_technical_education:
        strengths.append("🎓 Formación técnica relevante para el puesto")
    
    # Habilidades técnicas
    technical_skills = []
    tech_keywords = ['python', 'javascript', 'java', 'react', 'node', 'django', 'spring', 'html', 'css', 'git', 'sql']
    for skill in tech_keywords:
        if skill in skills:
            technical_skills.append(skill.capitalize())
    
    if technical_skills:
        skills_text = ', '.join(technical_skills[:4])  # Mostrar las primeras 4
        if len(technical_skills) > 4:
            skills_text += f" y {len(technical_skills)-4} más"
        strengths.append(f"🛠️ Tecnologías relevantes: {skills_text}")
    
    # Certificaciones
    if has_certifications:
        cert_preview = certifications[:50] + "..." if len(certifications) > 50 else certifications
        strengths.append(f"🏆 Certificaciones: {cert_preview}")
    
    # Idiomas
    if has_language_advantage:
        strengths.append("🌍 Manejo de inglés (ventaja competitiva)")
    
    if not strengths:
        strengths.append("💡 Candidato con potencial de desarrollo")
    
    # Identificar debilidades/desafíos (mejorado)
    weaknesses = []
    
    # Educación vs puesto
    if not has_technical_education and not is_junior:
        weaknesses.append("📚 Educación no técnica para posición especializada")
    
    # Experiencia vs requisitos
    if years_exp < 2 and not is_junior_position:
        weaknesses.append("⏱️ Experiencia limitada para los requisitos del puesto")
    elif years_exp < 1:
        weaknesses.append("🔰 Sin experiencia profesional documentada")
    
    # Skills alignment
    if not has_relevant_skills:
        weaknesses.append("🎯 Skills técnicos no completamente alineados con requisitos")
    
    # Certificaciones para junior
    if is_junior and not has_certifications:
        weaknesses.append("📜 Sin certificaciones que demuestren conocimientos actualizados")
    
    if not weaknesses:
        weaknesses.append("🔍 Perfil sólido sin debilidades significativas")
    
    # Generar sugerencias mejoradas
    suggestions = []
    
    if not has_relevant_skills:
        if 'backend' in job_title or 'python' in requirements:
            suggestions.append("📈 Desarrollar skills en tecnologías backend (Python, APIs, bases de datos)")
        elif 'frontend' in job_title:
            suggestions.append("📈 Desarrollar skills en tecnologías frontend (React, JavaScript, CSS)")
        elif 'fullstack' in job_title or 'full stack' in job_title:
            suggestions.append("📈 Desarrollar skills en tecnologías web (HTML, CSS, JavaScript, frameworks)")
    
    if not has_certifications:
        if 'python' in requirements or 'python' in job_title:
            suggestions.append("🎓 Considerar certificaciones en Python y frameworks relacionados")
        elif 'javascript' in requirements:
            suggestions.append("🎓 Considerar certificaciones en JavaScript y desarrollo web")
        else:
            suggestions.append("🎓 Obtener certificaciones profesionales relevantes al puesto")
    
    if years_exp < 2:
        suggestions.append("💼 Buscar experiencia práctica en proyectos reales o contribuciones open source")
    
    if not has_language_advantage and 'inglés' in requirements:
        suggestions.append("🌍 Mejorar nivel de inglés para ampliar oportunidades")
    
    suggestions.append("🛠️ Ampliar portfolio de tecnologías y proyectos personales")
    
    # Factores de decisión mejorados
    decision_factors = f"""📊 FACTORES CLAVE DE LA PREDICCIÓN:
• Experiencia: {years_exp} años ({'✅ Adecuada' if years_exp >= 1 else '⚠️ Limitada'})
• Educación: {'✅ Técnica' if has_technical_education else '⚠️ No técnica'}
• Skills: {'✅ Relevantes' if has_relevant_skills else '⚠️ Limitadas'}
• Certificaciones: {'✅ Presentes' if has_certifications else '⚠️ Ausentes'}
• Nivel requerido: {'✅ Match' if (is_junior and is_junior_position) or (not is_junior and not is_junior_position) else '⚠️ Desajuste'}"""
    
    # Resumen ejecutivo
    summary = f"""🎯 RESUMEN EJECUTIVO:
Candidato con {years_exp} años de experiencia como {candidate_data.puesto_actual or 'desarrollador'}, 
formación en {candidate_data.nivel_educacion}, presenta {final_probability*100:.1f}% de compatibilidad 
para el puesto de {offer_data.titulo}. {level_desc} basado en análisis de ML."""
    
    # Análisis detallado
    detailed_analysis = f"""📋 ANÁLISIS DETALLADO DE COMPATIBILIDAD:

🔍 PERFIL DEL CANDIDATO:
• Experiencia: {years_exp} años como {candidate_data.puesto_actual or 'desarrollador'}
• Educación: {candidate_data.nivel_educacion}
• Tecnologías: {candidate_data.habilidades[:100]}{'...' if len(candidate_data.habilidades) > 100 else ''}
• Idiomas: {candidate_data.idiomas or 'No especificado'}

💼 PERFIL DE LA OFERTA:
• Posición: {offer_data.titulo}
• Salario: ${offer_data.salario:,.2f}
• Ubicación: {offer_data.ubicacion}
• Requisitos: {offer_data.requisitos[:100]}{'...' if len(offer_data.requisitos) > 100 else ''}

🎯 RESULTADO DE COMPATIBILIDAD:
• Probabilidad base: {probability*100:.2f}%
• Probabilidad ajustada: {final_probability*100:.2f}% ({compatibility_level})
• Predicción: {'✅ Compatible' if final_probability >= 0.5 else '❌ No compatible'}
• Confianza del modelo: {confidence}
• Modelo utilizado: Gradient Boosting

📈 NIVEL DE RECOMENDACIÓN:
{recommendation}

🔧 FACTORES DETERMINANTES:
{decision_factors}"""
    
    return {
        'compatibility_level': compatibility_level,
        'recommendation': recommendation,
        'decision_factors': decision_factors,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'suggestions': suggestions,
        'summary': summary,
        'detailed_analysis': detailed_analysis,
        'adjusted_probability': final_probability  # Retornar probabilidad ajustada
    }


async def predict_batch_compatibility(input_data: BatchCompatibilityInput) -> BatchCompatibilityResult:
    """Predice compatibilidad para múltiples pares candidato-oferta"""
    
    start_time = time.time()
    predictions = []
    success_count = 0
    error_count = 0
    
    try:
        for pair in input_data.pairs:
            try:
                result = compatibility_predictor.predict_compatibility(
                    pair.candidate_id, 
                    pair.offer_id
                )
                
                prediction = CompatibilityPrediction(
                    candidate_id=result['candidate_id'],
                    offer_id=result['offer_id'],
                    probability=result['probability'],
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    model_used=result.get('model_used'),
                    prediction_date=result.get('prediction_date'),
                    error=result.get('error')
                )
                
                predictions.append(prediction)
                
                if result.get('error'):
                    error_count += 1
                else:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error procesando par {pair.candidate_id}-{pair.offer_id}: {e}")
                predictions.append(CompatibilityPrediction(
                    candidate_id=pair.candidate_id,
                    offer_id=pair.offer_id,
                    probability=0.0,
                    prediction=False,
                    confidence='Error',
                    error=str(e)
                ))
                error_count += 1
        
        processing_time = time.time() - start_time
        
        return BatchCompatibilityResult(
            predictions=predictions,
            total_processed=len(input_data.pairs),
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        return BatchCompatibilityResult(
            predictions=[],
            total_processed=0,
            success_count=0,
            error_count=len(input_data.pairs),
            processing_time=time.time() - start_time
        )


async def get_top_candidates_for_offer(input_data: TopCandidatesInput) -> List[CompatibilityPrediction]:
    """Obtiene los mejores candidatos para una oferta específica"""
    
    try:
        results = compatibility_predictor.predict_candidates_for_offer(
            input_data.offer_id, 
            input_data.top_n
        )
        
        predictions = []
        for result in results:
            prediction = CompatibilityPrediction(
                candidate_id=result['candidate_id'],
                offer_id=result['offer_id'],
                probability=result['probability'],
                prediction=result['prediction'],
                confidence=result['confidence'],
                ranking=result.get('ranking')
            )
            predictions.append(prediction)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error obteniendo top candidatos para oferta {input_data.offer_id}: {e}")
        return []


async def train_model(config: Optional[TrainingConfigInput] = None) -> ModelTrainingResult:
    """Entrena un nuevo modelo de compatibilidad"""
    
    start_time = time.time()
    
    try:
        logger.info("Iniciando entrenamiento de modelo desde GraphQL")
        
        # Ejecutar entrenamiento
        training_result = await asyncio.to_thread(train_compatibility_model)
        
        training_time = time.time() - start_time
        
        # Convertir métricas a tipos específicos
        metrics = None
        if training_result.get('best_metrics'):
            metrics_dict = training_result['best_metrics']
            metrics = TrainingMetrics(
                accuracy=metrics_dict.get('accuracy'),
                precision=metrics_dict.get('precision'),
                recall=metrics_dict.get('recall'),
                f1_score=metrics_dict.get('f1_score'),
                roc_auc=metrics_dict.get('roc_auc')
            )
        
        # Convertir resumen de datos
        data_summary = None
        if training_result.get('data_summary'):
            summary_dict = training_result['data_summary']
            data_summary = DataSummary(
                total_samples=summary_dict.get('total_samples'),
                positive_samples=summary_dict.get('positive_samples'),
                negative_samples=summary_dict.get('negative_samples'),
                features_count=summary_dict.get('features_count')
            )
        
        return ModelTrainingResult(
            success=True,
            message="Modelo entrenado exitosamente",
            best_model=training_result.get('best_model'),
            metrics=metrics,
            training_time=training_time,
            data_summary=data_summary
        )
        
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        return ModelTrainingResult(
            success=False,
            message=f"Error en entrenamiento: {str(e)}",
            training_time=time.time() - start_time
        )


async def get_model_info() -> ModelInfo:
    """Obtiene información del modelo actual"""
    
    try:
        info = compatibility_predictor.get_model_info()
        
        # Convertir métricas a tipo específico
        metrics = None
        if info.get('metrics'):
            metrics_dict = info['metrics']
            metrics = ModelMetrics(
                accuracy=metrics_dict.get('accuracy'),
                precision=metrics_dict.get('precision'),
                recall=metrics_dict.get('recall'),
                f1_score=metrics_dict.get('f1_score')
            )
        
        return ModelInfo(
            model_name=info.get('model_name'),
            model_type=info.get('model_type'),
            is_loaded=info.get('is_loaded', False),
            metrics=metrics,
            feature_importance_count=info.get('feature_importance_count'),
            top_features=info.get('top_features')
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo información del modelo: {e}")
        return ModelInfo(
            is_loaded=False
        )


async def get_feature_importance(top_n: Optional[int] = 20) -> ModelFeatureImportance:
    """Obtiene la importancia de features del modelo"""
    
    try:
        importance_dict = compatibility_predictor.get_feature_importance(top_n or 20)
        
        features = [
            FeatureImportance(feature_name=name, importance=importance)
            for name, importance in importance_dict.items()
        ]
        
        return ModelFeatureImportance(
            features=features,
            total_features=len(features)
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo importancia de features: {e}")
        return ModelFeatureImportance(
            features=[],
            total_features=0
        )


async def explain_prediction(candidate_id: str, offer_id: str) -> PredictionExplanation:
    """Proporciona explicación detallada de una predicción"""
    
    try:
        explanation = compatibility_predictor.explain_prediction(candidate_id, offer_id)
        
        # Convertir prediction
        pred_data = explanation.get('prediction', {})
        prediction = CompatibilityPrediction(
            candidate_id=pred_data.get('candidate_id', candidate_id),
            offer_id=pred_data.get('offer_id', offer_id),
            probability=pred_data.get('probability', 0.0),
            prediction=pred_data.get('prediction', False),
            confidence=pred_data.get('confidence', 'Unknown')
        )
        
        # Convertir feature importance
        feature_imp_dict = explanation.get('feature_importance', {})
        feature_importance = [
            FeatureImportance(feature_name=name, importance=importance)
            for name, importance in feature_imp_dict.items()
        ]
        
        # Convertir key factors
        factors_dict = explanation.get('key_factors', {})
        key_factors = PredictionFactors(
            experience_match=factors_dict.get('experience_match'),
            skills_overlap=factors_dict.get('skills_overlap'),
            education_fit=factors_dict.get('education_fit'),
            location_match=factors_dict.get('location_match')
        )
        
        return PredictionExplanation(
            prediction=prediction,
            key_factors=key_factors,
            feature_importance=feature_importance,
            recommendation=explanation.get('recommendation', 'No hay recomendación disponible')
        )
        
    except Exception as e:
        logger.error(f"Error explicando predicción: {e}")
        # Retornar explicación básica
        prediction = CompatibilityPrediction(
            candidate_id=candidate_id,
            offer_id=offer_id,
            probability=0.0,
            prediction=False,
            confidence='Error',
            error=str(e)
        )
        
        return PredictionExplanation(
            prediction=prediction,
            key_factors=PredictionFactors(),
            feature_importance=[],
            recommendation=f"Error generando explicación: {str(e)}"
        )


async def get_training_data_summary() -> TrainingDataSummary:
    """Obtiene resumen de los datos de entrenamiento disponibles"""
    
    try:
        # Extraer datos para obtener estadísticas
        training_data = data_extractor.create_training_dataset()
        
        if training_data.empty:
            return TrainingDataSummary(
                total_records=0,
                positive_samples=0,
                negative_samples=0,
                features_count=0
            )
        
        target_counts = training_data['target'].value_counts().to_dict()
        
        return TrainingDataSummary(
            total_records=len(training_data),
            positive_samples=target_counts.get(1, 0),
            negative_samples=target_counts.get(0, 0),
            features_count=len(training_data.columns) - 1  # Excluir target
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo resumen de datos: {e}")
        return TrainingDataSummary(
            total_records=0,
            positive_samples=0,
            negative_samples=0,
            features_count=0
        )


async def get_model_performance() -> ModelPerformanceMetrics:
    """Obtiene métricas de rendimiento del modelo actual"""
    
    try:
        info = compatibility_predictor.get_model_info()
        metrics = info.get('metrics', {})
        
        return ModelPerformanceMetrics(
            accuracy=metrics.get('accuracy'),
            precision=metrics.get('precision'),
            recall=metrics.get('recall'),
            f1_score=metrics.get('f1_score'),
            roc_auc=metrics.get('roc_auc'),
            confusion_matrix=metrics.get('confusion_matrix')
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas de rendimiento: {e}")
        return ModelPerformanceMetrics()


# Funciones auxiliares para verificar estado
async def is_model_loaded() -> bool:
    """Verifica si hay un modelo cargado"""
    return compatibility_predictor.is_loaded


async def get_model_status() -> Dict[str, Any]:
    """Obtiene estado completo del sistema ML"""
    
    try:
        info = compatibility_predictor.get_model_info()
        
        # Verificar datos disponibles
        candidates_count = 0
        offers_count = 0
        companies_count = 0
        
        try:
            candidates_df = data_extractor.extract_candidates()
            offers_df = data_extractor.extract_job_offers()
            companies_df = data_extractor.extract_companies()
            
            candidates_count = len(candidates_df)
            offers_count = len(offers_df)
            companies_count = len(companies_df)
            
        except Exception as e:
            logger.warning(f"Error verificando datos: {e}")
        
        return {
            'model_loaded': info.get('is_loaded', False),
            'model_name': info.get('model_name'),
            'model_type': info.get('model_type'),
            'preprocessor_fitted': info.get('preprocessor_fitted', False),
            'data_available': {
                'candidates': candidates_count,
                'offers': offers_count,
                'companies': companies_count
            },
            'system_ready': (
                info.get('is_loaded', False) and 
                info.get('preprocessor_fitted', False) and 
                candidates_count > 0 and 
                offers_count > 0
            )
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        return {
            'model_loaded': False,
            'system_ready': False,
            'error': str(e)
        }