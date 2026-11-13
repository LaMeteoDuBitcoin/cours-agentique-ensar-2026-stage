"""
Agent de Correspondance Stage-Compétences : Correspondance stage-compétences.
Sert de modèle architectural pour les étudiants.
"""

import os
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.base_agent import BaseAgentPlugin
from src.core.ingestion import get_vector_store
from src.core.state import AgentState

from .tools import find_matching_courses, format_matching_context


class InternshipMatcherPlugin(BaseAgentPlugin):
    @property
    def name(self) -> str:
        return "internship_matcher"

    @property
    def description(self) -> str:
        return (
            "Spécialiste du matching entre offres de stage/métiers et compétences académiques. "
            "Analyse une offre de stage ou une fiche de poste, identifie les UEs du cursus "
            "qui y répondent et calcule un score d'adéquation technique."
        )

    def create_node(self) -> Callable[[AgentState], dict[str, Any]]:
        # Initialisation du LLM local et du retriever vectoriel
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")

        llm = ChatOllama(base_url=ollama_url, model=model_name, temperature=0.0, num_predict=400)

        # Lazy loading du vector store pour éviter de ralentir le démarrage si non requis
        vector_store = None

        def node(state: AgentState) -> dict[str, Any]:
            nonlocal vector_store
            if vector_store is None:
                vector_store = get_vector_store(reindex_if_empty=False)

            # Dernier message utilisateur
            last_message = state["messages"][-1]
            user_query = str(last_message.content)

            # Retrieval et calcul déterministe des correspondances.
            matching_result = find_matching_courses(user_query, vector_store=vector_store, k=8)
            context_text = format_matching_context(matching_result)

            # Génération de la réponse avec les résultats calculés en Python.
            system_prompt = f"""Tu es un agent spécialisé dans le matching entre offres de stage/métiers et compétences académiques.
Tu disposes d'extraits de syllabus de cours (UE) et tu dois identifier les UEs qui correspondent aux compétences demandées pour un poste de stage.
Tu dois citer les codes UE présents dans les résultats et reprendre les scores calculés par l'outil, sans les recalculer ni en inventer.
Explique brièvement les compétences communes et précise qu'un score est un indicateur documentaire, pas une garantie d'embauche.
Voici le contexte documentaire que tu peux utiliser pour répondre à la requête de l'utilisateur.
======================
EXTRAIT DE SYLLABUS :
{context_text}
======================

Si le résultat indique une erreur ou aucune correspondance, explique-le honnêtement et ne propose pas de code UE absent des résultats.
"""

            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
            )

            return {
                "messages": [AIMessage(content=f"[Agent {self.name}]\n{response.content}")],
                "current_agent": self.name,
            }

        return node
