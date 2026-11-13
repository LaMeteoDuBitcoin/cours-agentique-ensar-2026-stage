from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.agents.stage.agent import InternshipMatcherPlugin
from src.agents.stage.tools import (
    compute_jaccard_similarity,
    extract_technical_keywords,
    find_matching_courses,
)
from src.core.base_agent import BaseAgentPlugin
from src.core.state import AgentState


def test_extract_technical_keywords_normalises_and_filters_text():
    keywords = extract_technical_keywords("Développer avec Python, Pandas et SQL dans une équipe.")

    assert keywords == ["developper", "python", "pandas", "sql"]


def test_compute_jaccard_similarity_handles_empty_and_partial_sets():
    assert compute_jaccard_similarity(set(), set()) == 0.0
    assert compute_jaccard_similarity({"python", "sql"}, {"python", "pandas"}) == 1 / 3


def test_find_matching_courses_groups_documents_and_scores_matches():
    first = MagicMock(
        metadata={"code_ue": "SD_321", "section_title": "Compétences"},
        page_content="Python et apprentissage automatique avec Scikit-Learn.",
    )
    second = MagicMock(
        metadata={"code_ue": "SD_321", "section_title": "Description"},
        page_content="Projet de machine learning sur des données réelles.",
    )
    retriever = MagicMock()
    retriever.invoke.return_value = [first, second]
    store = MagicMock()
    store.as_retriever.return_value = retriever

    result = find_matching_courses("Stage Python machine learning", vector_store=store)

    assert result["status"] == "success"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["code_ue"] == "SD_321"
    assert "python" in result["matches"][0]["competences_communes"]


@patch("src.agents.stage.agent.ChatOllama")
@patch("src.agents.stage.agent.get_vector_store")
def test_internship_matcher_node_uses_deterministic_matching(
    mock_get_vector_store: MagicMock, mock_chat_ollama: MagicMock
):
    document = MagicMock(
        metadata={"code_ue": "SSI_107", "section_title": "Compétences"},
        page_content="Sécurité réseau, Linux et analyse de trames.",
    )
    retriever = MagicMock()
    retriever.invoke.return_value = [document]
    vector_store = MagicMock()
    vector_store.as_retriever.return_value = retriever
    mock_get_vector_store.return_value = vector_store

    response = MagicMock()
    response.content = "UE SSI_107 pertinente."
    mock_chat_ollama.return_value.invoke.return_value = response

    agent = InternshipMatcherPlugin()
    assert isinstance(agent, BaseAgentPlugin)
    result = agent.create_node()(
        AgentState(
            messages=[HumanMessage(content="Stage en sécurité réseau Linux")],
            next_agent=None,
            current_agent=None,
            context_data={},
        )
    )

    assert result["current_agent"] == "internship_matcher"
    assert "SSI_107" in result["messages"][0].content
    assert "SSI_107" in mock_chat_ollama.return_value.invoke.call_args.args[0][0].content
