import os
import warnings
import asyncio
import logging
import gradio as gr
import pandas as pd
import tkinter as tk
from dotenv import dotenv_values, load_dotenv
from functools import partial
from tkinter import filedialog
from typing_extensions import (
    List,
    Literal,
    TypedDict,
    Dict,
    Union,
    Type,
    Generator,
    AsyncGenerator,
    Callable,
    Any,
)


from langsmith import Client

from langchain import hub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda, Runnable
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, TextSplitter
from langchain_text_splitters.base import TextSplitter
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage
from langchain_core.documents.transformers import BaseDocumentTransformer

from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamMode


logger_ = logging.getLogger(__name__)
# Create console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s   %(levelname)s   %(name)s:   %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger_.addHandler(console_handler)
logger_.setLevel(logging.INFO)

MODEL_PROVIDERS = Literal[
    "openai",
    "anthropic",
    "azure_openai",
    "azure_ai",
    "google_vertexai",
    "google_genai",
    "bedrock",
    "bedrock_converse",
    "cohere",
    "fireworks",
    "together",
    "mistralai",
    "huggingface",
    "groq",
    "ollama",
    "google_anthropic_vertex",
    "deepseek",
    "ibm",
    "nvidia",
    "xai",
    "perplexity",
]


class State(TypedDict):
    question: str  # User query
    context: List[Document]
    answer: str
    history: List[Dict[str, Any]]


class EnvLoader:
    """Helper class to load and validate API keys from .env file."""

    def __init__(self, required_keys=None):
        self.config = {}
        self.required_keys = required_keys or ["FIREWORKS_API_KEY", "LANGCHAIN_API_KEY"]
        self.load()
        self.validate()

    def load(self):
        """Load environment variables from .env file."""
        load_dotenv()
        self.config = dotenv_values()
        return self

    def validate(self):
        """Validate that all required API keys are present."""
        missing_keys = []
        for key in self.required_keys:
            value = self.config.get(key)
            if value:
                logger_.info(f"Successfully loaded {key}: {value[:4]}...{value[-4:]}")
            else:
                logger_.warning(f"{key} not found in environment variables")
                missing_keys.append(key)

        if missing_keys:
            logger_.warning(
                f"Missing required environment variables: {', '.join(missing_keys)}"
            )

        return self

    def get_key_config(self, key, default=None):
        """Get a specific config value."""
        return self.config.get(key, default)

    def get_config(self):
        """Get the entire config."""
        return self.config


class LangInit:
    """Helper class to instantiate and manage a Langchain Routine.
    This class provides a unified interface for initializing and configuring
    components needed for a Langchain-based application, including the client,
    prompt templates, embeddings, vector stores, and language models.

    Attributes:
        client: A Langchain Client instance for API interactions.
        prompt_template: The prompt template pulled from Langchain Hub.
        embeddings: Text embeddings model (not initialized by default).
        vector_store: Vector database for storing embeddings (not initialized by default).
        model: Language model for generating responses (not initialized by default).
        loader: Document loader for text processing (not initialized by default).

    Examples:
        >>> lang_init = LangInit()
        >>> lang_init.lc_client_init(env_config={"LANGCHAIN_API_KEY": "your_api_key"})
        >>> lang_init.pull_prompt(prompt_url="rlm/rag-prompt", include_model=True)
        >>> lang_init.chat_model_init(model_url="accounts/fireworks/models/llama-v3p1-70b-instruct",
        ...                           model_provider="fireworks")
        >>> lang_init.set_loader(PyPDFLoader)
    """

    def __init__(self):
        self.client = self.lc_client_init()
        self.prompt_template = self.pull_prompt()
        self.embeddings = None
        self.vector_store = None

    def lc_client_init(self, env_config=dotenv_values()):
        """Initialize langchain client with provided environment configuration."""
        try:
            _LANGCHAIN_API_KEY = env_config.get("LANGCHAIN_API_KEY")
            if not _LANGCHAIN_API_KEY:
                logger_.warning(
                    "No Langchain API key found. Please check your environment variables."
                )
                return self

            logger_.info("Loaded Langchain API key.")
            self.client = Client(api_key=_LANGCHAIN_API_KEY)
            logger_.info("Instantiated Langchain Client with API key")
            return self
        except Exception as e:
            logger_.error(f"Failed to initialize LangChain client: {str(e)}")
            return self

    def pull_prompt(self, prompt_url="rlm/rag-prompt", include_model=True):
        """Pull prompt from LangChain Hub."""
        if not self.client:
            logger_.warning("No LangChain client available. Initialize client first.")
            return None

        try:
            self.prompt_template = hub.pull(prompt_url, include_model=include_model)
            return self.prompt_template
        except Exception as e:
            logger_.error(f"Failed to pull prompt from {prompt_url}: {str(e)}")
            return None


class RAG:
    """Setup for Retrieval Augmented Generation."""

    def __init__(
        self,
        collection_name="example_collection",
        persist_directory="./chroma_langchain_db",
        prompt=None,
    ):
        """Initialize RAG setup with default configuration."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
            )
            self.llm = self.llm_init()
            self.prompt = prompt
            logger_.info(f"Initialized RAG setup with collection '{collection_name}'")
            if not prompt:
                logger_.warning(
                    "Prompt not loaded. Please provide `prompt` argument to class builder."
                )
            else:
                logger_.info(f"Prompt loaded")
        except Exception as e:
            logger_.error(f"Failed to initialize RAG setup: {str(e)}")

    def llm_init(
        self,
        model_url="accounts/fireworks/models/llama-v3p1-70b-instruct",
        model_provider="fireworks",
    ):
        """Initialize chat model from specified provider."""
        try:
            return init_chat_model(model_url, model_provider=model_provider)
        except Exception as e:
            logger_.error(f"Failed to initialize chat model: {str(e)}")
            return None

    def get_cwd(self):
        """Get current working directory."""
        return os.getcwd()

    def add_documents(self, documents):
        """Add documents to the vector store."""
        try:
            self.vector_store.add_documents(documents)
            logger_.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger_.error(f"Failed to add documents: {str(e)}")

    # Add this to your RAG class
    def remove_documents(self, file_paths: Union[List[str], Literal["all"]] = "all"):
        """Remove documents from the vector store based on file path.

        Args:
            file_paths: List of file paths to remove. If None, clear the entire collection.
        """
        try:
            if file_paths == "all":
                # Clear the entire collection
                self.vector_store.reset_collection()
                logger_.info("Cleared entire vector store collection. New instance created")
                return self.vector_store

            # For specific files, we need their document IDs
            # This requires tracking document metadata during insertion
            ids_to_remove = []
            for file_path in file_paths:
                # Get IDs of documents with this file path in metadata
                results = self.vector_store.get(where={"source": file_path})
                if results and "ids" in results:
                    ids_to_remove.extend(results["ids"])

            if ids_to_remove:
                self.vector_store.delete(ids=ids_to_remove)
                logger_.info(
                    f"Removed {len(ids_to_remove)} documents from vector store"
                )
                return self.vector_store
            else:
                logger_.info(
                    f"No documents found to remove for the specified file paths"
                )
                return self.vector_store

        except Exception as e:
            raise ValueError()

    def similarity_search(self, query, k=4):
        """Perform similarity search with the given query."""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger_.error(f"Error during similarity search: {str(e)}")
            return []

    def retrieve(self, state: State) -> Dict[str, Any]:
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state) -> Dict[str, Any]:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        if self.prompt:
            message = self.prompt.invoke(
                {"question": state["question"], "context": docs_content}
            )

        else:
            raise ValueError("Please provide a prompt.")

        if self.llm:
            response = self.llm.invoke(message)
            current_history = state.get("history", [])
            updated_history = current_history + [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": response.content},
            ]
            return {"answer": response.content, "history": updated_history}
        else:
            raise ValueError(
                "No LLM found. Please make sure that the RAG LLM is correctly instantiated."
            )


class GraphBuilder:
    """A utility class for building and managing state graphs in the RAG framework.
    This class simplifies the creation of state graphs by tracking nodes and edges,
    validating connections, and providing a fluent API for graph construction.
    Attributes:
        state_graph (StateGraph): The underlying state graph being built
        edges (set): Set of tuples representing graph edges (source, target)
        nodes (set): Set of node identifiers in the graph
    Methods:
        `_to_runnable`: Converts a function to a runnable node with tracking
        `add_sequence`: Add a sequence of connected nodes to the graph
        `add_start_edge`: Connect the ```START``` node to a target node
        `add_edge`: Add an edge between existing nodes
        `_validate`: Ensure the graph is properly connected
        `build_graph`: Build a graph from RAG class functions
        `compile_graph`: Validate and compile the final state graph
    Example:
        >>> # Define your state type
        >>> class RAGState(TypedDict):
        >>>     query: str
        >>>     documents: List[Document]
        >>>     response: str
        >>>
        >>> # Initialize graph builder
        >>> builder = GraphBuilder(RAGState)
        >>>
        >>> # Define your RAG functions
        >>> def retrieve(state: RAGState) -> RAGState:
        >>>     # Retrieval logic
        >>>     return state
        >>> def generate(state: RAGState) -> RAGState:
        >>>     # Generation logic
        >>>     return state
        >>>
        >>> # Create the graph
        >>> graph = builder.add_sequence(['retrieve', 'generate'])
        >>>                 .add_start_edge('retrieve')
        >>>                 .compile_graph()

        >>> # Execute the graph
        >>> result = graph.invoke({"question": "What is RAG?"}, stream_mode="messages")
    """

    def __init__(self, state):
        self.state_graph = StateGraph(state)
        self.edges = set()
        self.nodes = set([START])

    def _to_runnable(
        self, func, name: str = ""
    ) -> RunnableLambda[Callable[[StateGraph], StateGraph], str]:
        if name == "":
            name = func.__name__ + "_runnable"
        runnable = RunnableLambda(lambda state_: func(state_), name=name)
        self.nodes.add(str(runnable.name))  # Track this node
        return runnable

    def add_sequence(self, sequence: List):
        """Add a sequence of nodes with edges connecting them in order."""
        if not sequence:
            raise ValueError("Cannot add empty sequence")

        # Add all nodes to our tracking
        for node in sequence:
            self.nodes.add(node)

        # Add edges between consecutive nodes
        for i in range(len(sequence) - 1):
            source, target = sequence[i], sequence[i + 1]
            self.edges.add((source, target))

        # Add the sequence to the state graph
        self.state_graph.add_sequence(sequence)
        return self

    def add_start_edge(self, target):
        """Connect the START node to a target node."""
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' has not been added to the graph")

        self.edges.add((START, target))
        self.state_graph.add_edge(START, target)
        return self

    def add_edge(self, source, target):
        """Add an edge between two nodes."""
        # Validate source and target exist
        for node, name in [(source, "Source"), (target, "Target")]:
            if node not in self.nodes:
                raise ValueError(
                    f"{name} node '{node}' has not been added to the graph"
                )

        # Check if graph has any START connections
        if not any(src == START for src, _ in self.edges):
            raise ValueError("Graph has no START edges. Call add_start_edge first.")

        # Add the edge to our tracking and the state graph
        self.edges.add((source, target))
        self.state_graph.add_edge(source, target)
        return self

    def _validate(self):
        """Validate that the graph is properly connected."""
        if not any(src == START for src, _ in self.edges):
            raise ValueError("Graph has no START edges. Call add_start_edge first.")
        return True

    def build_graph(self, state, *, rag, functions=None):
        """Build a graph from a list of RAG class function names.

        Args:
            state: The state type for the graph
            functions: List of function names from RAG class to convert to runnables.
                       If None, defaults to ["retrieve", "generate"]

        Returns:
            A compiled StateGraph
        """
        if functions is None:
            functions = ["retrieve", "generate"]
        runnables = []
        for func_name in functions:
            if not hasattr(rag, func_name):
                raise ValueError(f"Function '{func_name}' not found in RAG class")
            func = getattr(rag, func_name)
            runnable = self._to_runnable(func=func)
            runnables.append(runnable)

        # Add the sequence of runnables
        if runnables:
            self.state_graph.add_sequence(runnables)
            self.add_start_edge(runnables[0].name)
            logger_.info(f"Built graph with functions: {functions} in that order")
            return self
        else:
            raise ValueError("No functions provided to build graph")

    def compile_graph(self) -> CompiledStateGraph:
        """Validate and compile the graph into a ```CompiledStateGraph``` object."""
        self._validate()
        return self.state_graph.compile()


with gr.Blocks(css="css/custom.css") as demo:
    gr.Markdown("# Moodle AI Assistant")

    state = State
    env = EnvLoader()
    lang = LangInit()
    rag = RAG(prompt=lang.prompt_template)
    builder = GraphBuilder(state)
    builder_sg = builder.build_graph(state=state, rag=rag)
    graph = builder_sg.compile_graph()

    all_splits = []
    documents = []

    def _load_to_dataframe(docs: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame({
            "ID": docs["ids"],
            "Title": [doc.get("title", None) for doc in docs['metadatas']],
            "Source": [doc.get("source", None) for doc in docs['metadatas']]
        })
        return df

    def load_and_split(files, splitter_cls=CharacterTextSplitter) -> pd.DataFrame:
        """Process selected files and add them to the RAG knowledge base."""
        if not files:
            raise ValueError("No files selected. Please select files to load.")
        splitter = splitter_cls()
        all_splits = []
        for file_path in files:
            try:
                if file_path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path=file_path)
                elif file_path.lower().endswith((".txt", ".md")):
                    loader = TextLoader(file_path=file_path)
                else:
                    continue  # Skip unsupported file types

                doc_split = loader.load_and_split(text_splitter=splitter)
                all_splits.extend(doc_split)
            except Exception as e:
                logger_.error(f"Error processing file {file_path}: {str(e)}")

        if all_splits:
            rag.add_documents(documents=all_splits)
            logger_.info(
                f"Successfully loaded {len(all_splits)} document chunks into knowledge base."
            )

            return _load_to_dataframe(rag.vector_store.get())
        logger_.error("Could not split the documents.")
        return pd.DataFrame()

    async def generate_answer(
        user_query: str,
        history: List[Dict[str, str]],
        *,
        stream_mode: StreamMode = "messages",
    ) -> AsyncGenerator:
        history_lc = []
        for msg in history:
            if msg["role"] == "user":
                history_lc.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_lc.append(AIMessage(content=msg["content"]))
        history_lc.append(HumanMessage(content=user_query))
        acc_answer = ""
        async for chunk, _ in graph.astream(
            {"question": user_query, "history": history}, stream_mode=stream_mode
        ):
            if hasattr(chunk, "content"):
                chunk_content = chunk.content  # type: ignore
            else:
                chunk_content = str(chunk)
            acc_answer += chunk_content
            yield acc_answer

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Files")
            file_explorer = gr.FileExplorer(root_dir=rag.get_cwd())

        with gr.Column(scale=5):
            chat_interface = gr.ChatInterface(
                fn=generate_answer,
                type="messages",
                chatbot=gr.Chatbot(type="messages"),
                textbox=gr.Textbox(placeholder="Ask something...", container=True),
                submit_btn="Submit",
                stop_btn="Stop",
                show_progress="hidden",
            )
    knowledge_df = gr.Dataframe(
        headers=["id", "title", "source"],
        interactive=False
    )
    file_explorer.change(
        fn=load_and_split,
        inputs=file_explorer,
        outputs=knowledge_df,
        show_progress="minimal",
    )

    refresh_df = gr.Button("Refresh Knowledge Base", variant="primary")
    refresh_df.click(
        fn=lambda: rag.remove_documents("all"),
        outputs=None
    ).then()

if __name__ == "__main__":
    demo.launch()
