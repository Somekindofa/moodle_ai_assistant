import gradio as gr
import os
from dotenv import dotenv_values, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.text import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents.base import Document
from langsmith import Client
from typing_extensions import List, TypedDict


load_dotenv()
config = dotenv_values()
FIREWORKS_API_KEY = config.get("FIREWORKS_API_KEY")
LANGSMITH_API_KEY = config.get("LANGSMITH_API_KEY")
client = Client(api_key=LANGSMITH_API_KEY)
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)
llm = init_chat_model(
    "accounts/fireworks/models/llama-v3p1-70b-instruct", model_provider="fireworks"
)
current_dir = os.getcwd()
history = gr.State([])
sem_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


def load_chunk_text(text_path:str="./elicitation.txt") -> List[Document]:
    with open(text_path, "r") as f:
        text = f.read()
    docs = sem_chunker.create_documents(texts=[text])
    return docs


def echo(message, history):
    return message


with gr.Blocks(css="css/custom.css") as demo:
    gr.Markdown("# Moodle AI Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Files")
            file_explorer = gr.FileExplorer(current_dir)
            
        with gr.Column(scale=5):
            chat_interface = gr.ChatInterface(  
                fn=echo,
                type="messages",
                chatbot=gr.Chatbot(),
                textbox=gr.Textbox(placeholder="Ask something...", container=True),
                submit_btn="Submit",
                stop_btn="Stop",
                show_progress="hidden"
            )

# Launch the app
if __name__ == "__main__":
    demo.launch()
