"""Graph service for building and managing conversation workflows."""

import logging
from typing import List, Callable, Dict, Any, Optional

from langchain_core.runnables import RunnableLambda
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
        self.nodes = set([START])
        self.edges = set()

    def _create_runnable(self, func: Callable, name: str = "") -> RunnableLambda:
        """Create a runnable from a function."""
        if not name:
            name = f"{func.__name__}_runnable"

        # Ensure unique names by checking existing nodes
        original_name = name
        counter = 1
        while name in self.nodes:
            name = f"{original_name}_{counter}"
            counter += 1

        runnable = RunnableLambda(lambda state: func(state), name=name)
        self.nodes.add(str(runnable.name))
        return runnable

    def build_conversation_graph(
        self, functions: Optional[List[str]] = None
    ) -> "ConversationGraphService":
        """Build conversation graph with specified RAG functions."""
        if functions is None:
            functions = ["retrieve", "generate"]

        # Create runnables from RAG service methods
        runnables = []
        for func_name in functions:
            if not hasattr(self.rag_service, func_name):
                raise ValueError(f"Function '{func_name}' not found in RAG service")

            func = getattr(self.rag_service, func_name)
            runnable = self._create_runnable(func)
            runnables.append(runnable)

        # Build the graph structure
        self.state_graph.add_sequence(runnables)
        self.state_graph.add_edge(START, runnables[0].name)

        # Track edges
        self.edges.add((START, runnables[0].name))
        for i in range(len(runnables) - 1):
            source_name = runnables[i].name
            target_name = runnables[i + 1].name
            self.edges.add((source_name, target_name))

        logger.info(f"Conversation graph built with functions: {functions}")
        return self

    def compile_graph(self) -> CompiledStateGraph:
        """Compile the state graph."""
        if not any(src == START for src, _ in self.edges):
            raise ValueError("Graph has no START edges. Cannot compile.")

        compiled_graph = self.state_graph.compile()
        logger.info("Conversation graph compiled successfully")
        return compiled_graph
