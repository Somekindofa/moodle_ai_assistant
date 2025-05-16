import gradio as gr
import os

import asyncio
import logging

from dotenv import dotenv_values, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.text import TextLoader
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
        'openai',
        'anthropic',
        'azure_openai',
        'azure_ai',
        'google_vertexai',
        'google_genai',
        'bedrock',
        'bedrock_converse',
        'cohere',
        'fireworks',
        'together',
        'mistralai',
        'huggingface',
        'groq',
        'ollama',
        'google_anthropic_vertex',
        'deepseek',
        'ibm',
        'nvidia',
        'xai',
        'perplexity'
    ]

class EnvLoader:
    """Helper class to load and validate API keys from .env file."""
    
    def __init__(self, required_keys=None):
        self.config = {}
        self.required_keys = required_keys or ["FIREWORKS_API_KEY", "LANGCHAIN_API_KEY"]
        
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
                print(f"Successfully loaded {key}: {value[:4]}...{value[-4:]}")
            else:
                print(f"{key} not found in environment variables")
                missing_keys.append(key)
        
        if missing_keys:
            warnings.warn(f"Missing required environment variables: {', '.join(missing_keys)}")
            
        return self
    
    def get(self, key, default=None):
        """Get a specific config value."""
        return self.config.get(key, default)

class LangInit:
    """Helper class to instantiate a Langchain Routine"""
    
    def __init__(self):
        self.client = None
        self.prompt_template = None
        self.embeddings = None
        self.vector_store = None
        self.model = None
        self.splitter = None
        self.logger = logging.getLogger(__name__)
    
    def lc_client_init(self, env_config):
        """Initialize langchain client with provided environment configuration."""
        try:
            _LANGCHAIN_API_KEY = env_config.get("LANGCHAIN_API_KEY")
            if not _LANGCHAIN_API_KEY:
                self.logger.warning("No Langchain API key found. Please check your environment variables.")
                return self
                
            self.logger.info("Loaded Langchain API key.")
            self.client = Client(api_key=_LANGCHAIN_API_KEY)
            self.logger.info("Instantiated Langchain Client with API key")
            return self
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain client: {str(e)}")
            return self
    
    def pull_prompt(self, prompt_url="rlm/rag-prompt", include_model=True):
        """Pull prompt from LangChain Hub."""
        if not self.client:
            self.logger.warning("No LangChain client available. Initialize client first.")
            return None
            
        try:
            prompt = self.client.pull_prompt(prompt_url, include_model=include_model)
            return prompt
        except Exception as e:
            self.logger.error(f"Failed to pull prompt from {prompt_url}: {str(e)}")
            return None
        
    def chat_model_init(
        self,
        model_url="accounts/fireworks/models/llama-v3p1-70b-instruct",
        model_provider="fireworks"
    ):
        """Initialize chat model from specified provider."""
        try:
            self.model = init_chat_model(model_url, model_provider=model_provider)
            return self
        except Exception as e:
            self.logger.error(f"Failed to initialize chat model: {str(e)}")
            return self
    
    def load_splitter(self, text_splitter, **kwargs):
        """Load text splitter with specified configuration."""
        try:
            self.splitter = text_splitter(**kwargs)
            return self
        except Exception as e:
            self.logger.error(f"Failed to load text splitter: {str(e)}")
            return self
        
class RAGSetup:
    """Setup for Retrieval Augmented Generation."""
    
    def __init__(self, collection_name="example_collection", persist_directory="./chroma_langchain_db"):
        """Initialize RAG setup with default configuration."""
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
            )
            self.history = []
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Initialized RAG setup with collection '{collection_name}'")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG setup: {str(e)}")
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
            self.logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            
    def similarity_search(self, query, k=4):
        """Perform similarity search with the given query."""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            self.logger.error(f"Error during similarity search: {str(e)}")
            return []


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Initialize LangInit and get prompt
lang_init = LangInit()
env_loader = EnvLoader().load().validate()
lang_init.lc_client_init(env_loader.config)
prompt = lang_init.pull_prompt()
llm = lang_init.chat_model_init().model

# Define application steps
def retrieve(state: State):
    rag_setup = RAGSetup()  # Create an instance of RAGSetup to access vector_store
    retrieved_docs = rag_setup.vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    if prompt:
        message = prompt.invoke({"question": state["question"], "context": docs_content})
    else:
        # Fallback if no prompt is available
        message = f"Question: {state['question']}\n\nContext: {docs_content}"
        
    if llm:
        response = llm.invoke(message)
        return {"answer": response.content}
    else:
        return {"answer": "LLM not initialized properly."}


def load_chunk_text(
    #TOFIX
    text_path: str = "./elicitation.txt", chunker: SemanticChunker = sem_chunker
) -> List[Document]:
    with open(text_path, "r") as f:
        text = f.read()
    docs = chunker.create_documents(texts=[text])
    return docs


def echo(message, history):
    return message


with gr.Blocks(css="css/custom.css") as demo:
    gr.Markdown("# Moodle AI Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Files")
            #TOFIX
            file_explorer = gr.FileExplorer(current_dir)

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

# demo.launch()
all_splits = load_chunk_text()
#TOFIX
_ = vector_store.add_documents(documents=all_splits)
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

async def run():
    async for message, metadata in graph.astream(
        {"question": "How can one better transmit glassblowing knowledge to novices?"},
        stream_mode="messages",
    ):
        print(message.content) # type: ignore

asyncio.run(run())