import gradio as gr
import os

import asyncio
import logging

from dotenv import dotenv_values, load_dotenv
from langchain import hub
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
from typing_extensions import List, Literal, TypedDict, Dict, Union, Type
import warnings

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
    """Helper class to instantiate a Langchain Routine"""

    def __init__(self):
        self.client = self.lc_client_init()
        self.prompt_template = self.pull_prompt()
        self.embeddings = None
        self.vector_store = None
        self.model = None
        self.splitter = None

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
            self.prompt_template = hub.pull(
                prompt_url, include_model=include_model
            )
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

    def load_splitter(self, text_splitter, **kwargs):
        """Load text splitter with specified configuration."""
        try:
            self.splitter = text_splitter(**kwargs)
            return self
        except Exception as e:
            logger_.error(f"Failed to load text splitter: {str(e)}")
            return self


class RAG:
    """Setup for Retrieval Augmented Generation."""

    class State(TypedDict):
        question: str  # User query
        context: List[Document]
        answer: str

    def __init__(
        self,
        collection_name="example_collection",
        persist_directory="./chroma_langchain_db",
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

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State, *, prompt=None):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        if prompt:
            message = prompt.invoke(
                {"question": state["question"], "context": docs_content}
            )
        else:
            # Fallback if no prompt is available
            message = f"Question: {state['question']}\n\nContext: {docs_content}"

        if llm:
            response = llm.invoke(message)
            return {"answer": response.content}
        else:
            return {"answer": "LLM not initialized properly."}


# Initialize LangInit and get prompt
env = EnvLoader()
lang = LangInit()
rag = RAG()
prompt = lang.pull_prompt()
llm = lang.chat_model_init().model


def echo(message, history):
    return message


with gr.Blocks(css="css/custom.css") as demo:
    gr.Markdown("# Moodle AI Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Files")
            file_explorer = gr.FileExplorer(rag.get_cwd())

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
    all_splits = []
    documents_dir = "./documents"
    for file in os.listdir(path=documents_dir):
        file_path = os.path.join(documents_dir, file)
        print(f"Loading file {file_path}...")
        loader = PyPDFLoader(file_path=file_path)
        splits = loader.load()
        all_splits.extend(splits)
    _ = rag.add_documents(documents=all_splits)
    graph_builder = StateGraph(rag.State).add_sequence([rag.retrieve, rag.generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    async def run():
        async for message, metadata in graph.astream(
            {
                "question": "How can one better transmit glassblowing knowledge to novices?"
            },
            stream_mode="messages",
        ):
            print(message.content)  # type: ignore

    asyncio.run(run())
    # demo.launch()
