import os
import warnings
import asyncio
import logging
import gradio as gr
import tkinter as tk
from dotenv import dotenv_values, load_dotenv
from functools import partial
from tkinter import filedialog
from typing_extensions import List, Literal, TypedDict, Dict, Union, Type, Generator

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
from langchain_text_splitters.base import TextSplitter
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage
from langchain_core.documents.transformers import BaseDocumentTransformer
from langsmith import Client
from langgraph.graph import StateGraph, START



logger_ = logging.getLogger(__name__)

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
        self.model = None
        self.loader = None

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

    def chat_model_init(
        self,
        model_url="accounts/fireworks/models/llama-v3p1-70b-instruct",
        model_provider="fireworks",
    ):
        """Initialize chat model from specified provider."""
        try:
            self.model = init_chat_model(model_url, model_provider=model_provider)
            return self
        except Exception as e:
            logger_.error(f"Failed to initialize chat model: {str(e)}")
            return self

    def set_loader(self, loader_class):
        """Store a document loader class to be used for file processing.

        This doesn't instantiate the loader yet, but stores the class
        for later instantiation with file paths.
        """
        self.loader = loader_class
        return self.loader


class State(TypedDict):
    question: str  # User query
    context: List[Document]
    answer: str


class RAG:
    """Setup for Retrieval Augmented Generation."""

    def __init__(
        self,
        collection_name="example_collection",
        persist_directory="./chroma_langchain_db",
        llm=None,
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
            self.history = []
            self.llm = llm
            logger_.info(f"Initialized RAG setup with collection '{collection_name}'")
        except Exception as e:
            logger_.error(f"Failed to initialize RAG setup: {str(e)}")
            raise

    def get_cwd(self):
        """Get current working directory."""
        return os.getcwd()

    def get_history(self):
        """Get conversation history."""
        return self.history

    def add_documents(self, documents):
        """Add documents to the vector store."""
        try:
            self.vector_store.add_documents(documents)
            logger_.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger_.error(f"Failed to add documents: {str(e)}")

    def similarity_search(self, query, k=4):
        """Perform similarity search with the given query."""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger_.error(f"Error during similarity search: {str(e)}")
            return []

    def retrieve(self, state):
        print(f"\nState in RETRIEVE is: {state}\n")
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def set_llm(self, llm):
        """Set or update the language model."""
        self.llm = llm
        logger_.info("LLM updated successfully")

    def generate(self, state, *, prompt=None):
        print(f"\nState in GENERATE is: {state}\n")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        if prompt:
            message = prompt.invoke(
                {"question": state["question"], "context": docs_content}
            )
        else:
            # Fallback if no prompt is available
            message = f"Question: {state['question']}\n\nContext: {docs_content}"

        if self.llm:
            response = self.llm.invoke(message)
            return {"answer": response.content}
        else:
            return {"answer": "LLM not initialized properly."}


class GraphBuilder:
    def __init__(self, state):
        self.state_graph = StateGraph(state)
        self.edges = set()
        self.nodes = set([START])

    def to_runnable(self, func, name: str = ""):
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

    def validate(self):
        """Validate that the graph is properly connected."""
        if not any(src == START for src, _ in self.edges):
            raise ValueError("Graph has no START edges. Call add_start_edge first.")
        return True

    def build_graph(self, state_graph, functions=None):
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
            runnable = state_graph.to_runnable(func=func)
            runnables.append(runnable)

        # Add the sequence of runnables
        if runnables:
            state_graph.add_sequence(runnables)
            state_graph.add_start_edge(runnables[0].name)
            logger_.info(f"\nBuilt graph with functions: {functions}\nin that order")
            return state_graph.compile_graph()
        else:
            raise ValueError("No functions provided to build graph")

    def compile_graph(self):
        """Validate and compile the graph."""
        self.validate()
        return self.state_graph.compile()


with gr.Blocks(css="css/custom.css") as demo:
    gr.Markdown("# Moodle AI Assistant")
    
    state = State
    env = EnvLoader()
    lang = LangInit()
    rag = RAG()

    loader = lang.set_loader(PyPDFLoader)
    prompt = lang.pull_prompt()
    rag.set_llm(lang.chat_model_init().model)

    
    all_splits = []
    documents = []

    def load_files_to_knowledge(selected_files):
        """Process selected files and add them to the RAG knowledge base."""
        if not selected_files:
            return "No files selected. Please select files to load."

        all_splits = []
        for file_path in selected_files:
            try:
                if file_path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path=file_path)
                elif file_path.lower().endswith((".txt", ".md")):
                    loader = TextLoader(file_path=file_path)
                else:
                    continue  # Skip unsupported file types

                splits = loader.load()
                all_splits.extend(splits)
            except Exception as e:
                logger_.error(f"Error processing file {file_path}: {str(e)}")

        if all_splits:
            rag.add_documents(documents=all_splits)
            logger_.info(
                f"Successfully loaded {len(all_splits)} document chunks into knowledge base."
            )
            return f"Successfully loaded {len(all_splits)} document chunks into knowledge base."
        else:
            return "No valid documents were processed. Please check file types and try again."

    async def generate_answer(user_query:str, history:List[Dict[str, str]], *, stream_mode="messages") -> Generator:
        history_lc = []
        for msg in history:
            if msg["role"] == "user":
                history_lc.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_lc.append(AIMessage(content=msg["content"]))
            history_lc.append(HumanMessage(content=user_query))
    
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Files")
            file_explorer = gr.FileExplorer(root_dir=rag.get_cwd())
            load_knowledge = gr.Button("Load to knowledge")
            knowledge_status = gr.Textbox(label="Status", interactive=False)
            load_knowledge.click(
                fn=load_files_to_knowledge,
                inputs=file_explorer,
                outputs=knowledge_status,
            )

        with gr.Column(scale=5):
            chat_interface = gr.ChatInterface(
                fn=echo,
                type="messages",
                chatbot=gr.Chatbot(type="messages"),
                textbox=gr.Textbox(placeholder="Ask something...", container=True),
                submit_btn="Submit",
                stop_btn="Stop",
                show_progress="hidden",
            )


if __name__ == "__main__":
    demo.launch()
