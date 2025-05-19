import os
import warnings
import asyncio
import logging
import gradio as gr
from dotenv import dotenv_values, load_dotenv
from functools import partial

from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
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
import tkinter as tk
from tkinter import filedialog

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

    def load_splitter(self, text_splitter_cls):
        """Load text splitter with specified configuration."""
        try:
            self.splitter = text_splitter_cls
            return self
        except Exception as e:
            logger_.error(f"Failed to load text splitter: {str(e)}")
            return self


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


def echo(message, history):
    return message


# with gr.Blocks(css="css/custom.css") as demo:
#     gr.Markdown("# Moodle AI Assistant")

#     def load_files_to_knowledge(selected_files):
#         """Process selected files and add them to the RAG knowledge base."""
#         if not selected_files:
#             return "No files selected. Please select files to load."

#         all_splits = []
#         for file_path in selected_files:
#             try:
#                 if file_path.lower().endswith(".pdf"):
#                     loader = PyPDFLoader(file_path=file_path)
#                 elif file_path.lower().endswith((".txt", ".md")):
#                     loader = TextLoader(file_path=file_path)
#                 else:
#                     continue  # Skip unsupported file types

#                 splits = loader.load()
#                 all_splits.extend(splits)
#             except Exception as e:
#                 logger_.error(f"Error processing file {file_path}: {str(e)}")

#         if all_splits:
#             rag.add_documents(documents=all_splits)
#             logger_.info(
#                 "Successfully loaded {len(all_splits)} document chunks into knowledge base."
#             )
#         else:
#             return "No valid documents were processed. Please check file types and try again."

#     with gr.Row():
#         with gr.Column(scale=1):
#             gr.Markdown("### Files")
#             file_explorer = gr.FileExplorer(rag.get_cwd())
#             load_knowledge = gr.Button("Load to knowledge")
#             knowledge_status = gr.Textbox(label="Status", interactive=False)
#             load_knowledge.click(
#                 fn=load_files_to_knowledge,
#                 inputs=file_explorer,
#                 outputs=knowledge_status,
#             )

#         with gr.Column(scale=5):
#             chat_interface = gr.ChatInterface(
#                 fn=echo,
#                 type="messages",
#                 chatbot=gr.Chatbot(type="messages"),
#                 textbox=gr.Textbox(placeholder="Ask something...", container=True),
#                 submit_btn="Submit",
#                 stop_btn="Stop",
#                 show_progress="hidden",
#             )


class GraphBuilder:
    def __init__(self, state):
        self.state_graph = StateGraph(state)
    
    def to_runnable(self, func, name:str=""):
        if name=="":
            name = func.__name__ + "_runnable"
        return RunnableLambda(lambda state_: func(state_), name=name)

    def add_sequence(self, sequence:List):
        self.state_graph.add_sequence(sequence)

    def add_edge(self, edge_name):
        self.state_graph.add_edge(START, edge_name)
    
    def compile(self):
        return self.state_graph.compile()


if __name__ == "__main__":
    # Initialize LangInit and get prompt
    env = EnvLoader()
    lang = LangInit()
    rag = RAG()

    lang.load_splitter(PyPDFLoader)
    prompt = lang.pull_prompt()
    rag.set_llm(lang.chat_model_init().model)

    state = State
    all_splits = []
    documents = []
    # Set up Tkinter dialog to select files
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    documents = list(
        filedialog.askopenfilenames(
            title="Select files to add to knowledge base",
            filetypes=[
                ("Document files", "*.pdf;*.txt;*.md"),
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("All files", "*.*"),
            ],
        )
    )
    root.destroy()

    if not documents:
        print("No files selected. Please run the program again to select files.")
    else:
        print(f"Selected {len(documents)} files to process.")
        for file in documents:
            file_path = file
            if lang.splitter:
                if file_path:
                    loader = lang.splitter(file_path=file_path)
                    splits = loader.load()
                    all_splits.extend(splits)

    rag.add_documents(documents=all_splits)

    state_graph = GraphBuilder(state)
    retrieve_runnable = state_graph.to_runnable(func=rag.retrieve)
    generate_runnable = state_graph.to_runnable(func=rag.generate)
    state_graph.add_sequence([retrieve_runnable, generate_runnable])
    state_graph.add_edge(edge_name="retrieve_runnable") # this adds an edge called "retrieve" to START (pls generalize the function 'add_edge')
    graph = state_graph.compile()

    async def run():
        async for message, metadata in graph.astream(
            {"question": "What are the main steps to build a RAG?"},
            stream_mode="messages",
        ):
            print(message.content)  # type: ignore

    asyncio.run(run())
    # demo.launch()
