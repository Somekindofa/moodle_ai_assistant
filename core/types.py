"""Core types and models for the Moodle AI Assistant."""

from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents.base import Document
from langgraph.graph import MessagesState


class ConversationState(MessagesState):
    """State management for conversation flow.
    
    This class manages the context and history of the conversation.
    It's a subclassed `MessagesState` with additional `context` and `video_metadata` management capabilities.
    """

    ##TODO Refactor to suggest an appropriate reducer to update context
    ## context: Annotated[List[Document], reducer]
    context: List[Document]
    video_metadata: Optional[Dict[str, Any]]
    hypothetical_document: Optional[str]  # HyDE-generated hypothetical document
    enhanced_query: Optional[str]  # [LEGACY] Enhanced query from query enhancement node
