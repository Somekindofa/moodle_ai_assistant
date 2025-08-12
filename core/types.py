"""Core types and models for the Moodle AI Assistant."""

from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.documents.base import Document


class ConversationState(TypedDict):
    """State management for conversation flow."""

    question: str
    context: List[Document]
    answer: str
    history: List[Dict[str, Any]]
