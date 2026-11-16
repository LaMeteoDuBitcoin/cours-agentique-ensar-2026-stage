"""Outils déterministes pour le matching entre offres de stage et syllabus."""

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from src.core.ingestion import get_vector_store

_STOPWORDS = {
    "avec", "dans", "pour", "sur", "les", "des", "une", "aux", "est", "sont",
    "être", "poste", "stage", "mission", "missions", "profil", "recherche",
    "équipe", "equipe", "travail", "bonne", "plus", "ainsi", "cette", "vous", "nous",
    "leur", "leurs", "tous", "toutes", "tout", "toute", "trop", "très", "peu"
}


def _normalise_token(token: str) -> str:
    """Supprime les accents et convertit un mot en minuscules."""
    without_accents = unicodedata.normalize("NFKD", token)
    return "".join(char for char in without_accents if not unicodedata.combining(char)).lower()


# La fonction conserve les termes alphanumériques, supprime les accents et 
# ignore les mots génériques présents dans ``_STOPWORDS``.
def extract_technical_keywords(text: str) -> list[str]:
    """Extrait les mots-clés uniques et normalisés d'une offre ou d'un syllabus.

    La fonction conserve les termes alphanumériques, supprime les accents et
    ignore les mots génériques présents dans ``_STOPWORDS``.
    """
    tokens = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9+#.-]{1,}", text)
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalised = _normalise_token(token.strip(".-"))
        if len(normalised) < 3 or normalised in _STOPWORDS or normalised in seen:
            continue
        seen.add(normalised)
        keywords.append(normalised)
    return keywords


# Le résultat est compris entre 0 et 1. Deux ensembles vides produisent 0
# afin d'éviter une division par zéro et de signaler l'absence de données.
def compute_jaccard_similarity(offer_skills: set[str], course_skills: set[str]) -> float:
    """Calcule la similarité de Jaccard entre les compétences de l'offre et du cours.

    Le résultat est compris entre 0 et 1. Deux ensembles vides produisent 0
    afin d'éviter une division par zéro et de signaler l'absence de données.
    """
    union = offer_skills | course_skills
    return len(offer_skills & course_skills) / len(union) if union else 0.0

# Regroupe les documents récupérés par code d'unité d'enseignement
def _group_documents_by_course(documents: Iterable[Any]) -> dict[str, list[Any]]:
    """Regroupe les documents récupérés par code d'unité d'enseignement."""
    grouped: dict[str, list[Any]] = {}
    for document in documents:
        code = str(document.metadata.get("code_ue", "")).strip().upper()
        if code and code != "INCONNU":
            grouped.setdefault(code, []).append(document)
    return grouped


# Les documents sont récupérés depuis le vector store, regroupés par code UE,
# puis comparés aux mots-clés de l'offre. Le vector store peut être injecté
# pour faciliter les tests et éviter une nouvelle initialisation.
def find_matching_courses(
    offer_text: str,
    vector_store: Any | None = None,
    k: int = 8,
) -> dict[str, Any]:
    """Recherche les UE pertinentes et calcule un score explicable pour chacune.

    Les documents sont récupérés depuis le vector store, regroupés par code UE,
    puis comparés aux mots-clés de l'offre. Le vector store peut être injecté
    pour faciliter les tests et éviter une nouvelle initialisation.
    """
    if not offer_text.strip():
        return {
            "status": "invalid_input",
            "message": "L'offre de stage ne peut pas être vide.",
            "matches": [],
        }

    try:
        store = vector_store or get_vector_store(reindex_if_empty=False)
        documents = store.as_retriever(search_kwargs={"k": k}).invoke(offer_text)
    except Exception as error:
        return {
            "status": "error",
            "message": f"Erreur lors de la recherche dans les syllabus : {error}",
            "matches": [],
        }

    offer_skills = set(extract_technical_keywords(offer_text))
    matches: list[dict[str, Any]] = []
    for code, course_documents in _group_documents_by_course(documents).items():
        course_text = "\n".join(str(document.page_content) for document in course_documents)
        course_skills = set(extract_technical_keywords(course_text))
        shared_skills = sorted(offer_skills & course_skills)
        matches.append(
            {
                "code_ue": code,
                "score": round(compute_jaccard_similarity(offer_skills, course_skills) * 100, 2),
                "competences_communes": shared_skills,
                "metadata": course_documents[0].metadata,
                "context": course_text,
            }
        )

    matches.sort(key=lambda match: (-match["score"], match["code_ue"]))
    return {
        "status": "success",
        "offer_keywords": sorted(offer_skills),
        "matches": matches,
        "documents_analyzed": len(documents),
    }

# Transforme les résultats du matching en contexte lisible pour le LLM
def format_matching_context(result: dict[str, Any]) -> str:
    """Transforme les résultats du matching en contexte lisible pour le LLM.

    En cas d'erreur ou d'absence de résultat, un message explicite est renvoyé
    afin que l'agent puisse répondre sans inventer de correspondance.
    """
    if result.get("status") != "success":
        return str(result.get("message", "Aucun résultat disponible."))
    if not result["matches"]:
        return "Aucune UE identifiable dans les extraits récupérés."

    sections = []
    for match in result["matches"]:
        sections.append(
            f"[{match['code_ue']}] score de similarité de {match['score']}%; "
            f"compétences communes={', '.join(match['competences_communes']) or 'aucune'}\n"
            f"{match['context']}"
        )
    return "\n\n".join(sections)
