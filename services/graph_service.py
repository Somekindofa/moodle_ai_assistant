"""Graph service for building and managing conversation workflows."""

import logging
from re import I
from typing import List, Callable, Dict, Any, Optional, TypedDict, Union

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph

from core.types import ConversationState
from services.rag_service import RAGService


logger = logging.getLogger(__name__)


class ConversationGraphService:
    """Service for building and managing conversation graphs."""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.state_graph = StateGraph(ConversationState)
        self.edges = set()

    def build_conversation_graph(
        self, functions: Optional[List[str]] = None
    ) -> "ConversationGraphService":
        """Build conversation graph with specified RAG functions."""
        if functions is None:
            functions = ["retrieve", "generate"]

        # Create sequence of (name, callable) tuples
        sequence = []
        for func_name in functions:
            if not hasattr(self.rag_service, func_name):
                raise ValueError(f"Function '`{func_name}`' not implemented in RAG service.\nPlease implement '`{func_name}`' method in `RAGService` class.")

            func = getattr(self.rag_service, func_name)
            sequence.append((func_name, func))

        # Build the graph structure
        self.state_graph.add_sequence(sequence)
        self.state_graph.add_edge(START, functions[0])

        # Track edges
        self.edges.add((START, functions[0]))
        for i in range(len(functions) - 1):
            source_name = functions[i]
            target_name = functions[i + 1]
            self.edges.add((source_name, target_name))

        logger.info(f"Conversation graph built with functions: {functions}")
        return self

    def build_routed_graph(self) -> "ConversationGraphService":
        """Build the production graph with an upfront router node.

        Topology:
            START → route_query ──┬─(rag)──→ multi_query → retrieve_combined → rerank → generate → END
                                  └─(llm_only)──────────────────────────────→ direct_generate → END
        """
        rs = self.rag_service

        self.state_graph.add_node("route_query",      rs.route_query)
        self.state_graph.add_node("multi_query",       rs.multi_query)
        self.state_graph.add_node("retrieve_combined", rs.retrieve_combined)
        self.state_graph.add_node("rerank",            rs.rerank)
        self.state_graph.add_node("generate",          rs.generate)
        self.state_graph.add_node("direct_generate",   rs.direct_generate)

        # Entry point
        self.state_graph.add_edge(START, "route_query")
        self.edges.add((START, "route_query"))

        # Conditional branch on state["route"]
        self.state_graph.add_conditional_edges(
            "route_query",
            lambda state: state.get("route", "rag"),
            {
                "rag":      "multi_query",
                "llm_only": "direct_generate",
            },
        )

        # RAG path
        self.state_graph.add_edge("multi_query",       "retrieve_combined")
        self.state_graph.add_edge("retrieve_combined", "rerank")
        self.state_graph.add_edge("rerank",            "generate")

        logger.info("Routed graph built: route_query → {rag: multi_query→…→generate | llm_only: direct_generate}")
        return self

    def compile_graph(self, checkpointer=None) -> CompiledStateGraph:
        """Compile the state graph."""
        if not checkpointer:
            checkpointer = MemorySaver()
        if not any(src == START for src, _ in self.edges):
            raise ValueError("Graph has no START edges. Cannot compile.")

        try:
            compiled_graph = self.state_graph.compile(checkpointer=checkpointer)
            logger.info("Conversation graph compiled successfully")
            return compiled_graph
        except Exception as e:
            logger.error(f"Error compiling conversation graph: {e}")
            raise
