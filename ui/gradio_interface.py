"""Gradio UI components for the Moodle AI Assistant."""

import gradio as gr
from typing import List, Dict
from pipeline import MoodleAIAssistantPipeline


class MoodleAIAssistantUI:
    """User interface for the Moodle AI Assistant."""

    def __init__(self, pipeline: MoodleAIAssistantPipeline):
        self.pipeline = pipeline

    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        with gr.Blocks(css="css/custom.css") as interface:
            gr.Markdown("# Moodle AI Assistant")

            with gr.Tabs():
                self._create_chat_tab()
                self._create_backend_tab()

        return interface

    def _create_chat_tab(self) -> None:
        """Create the chat interface tab."""
        with gr.TabItem("Chat Interface"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Files")
                    file_explorer = gr.FileExplorer(
                        root_dir=self.pipeline.get_current_directory()
                    )

                with gr.Column(scale=5):
                    chat_interface = gr.ChatInterface(
                        fn=self.pipeline.generate_response,
                        type="messages",
                        chatbot=gr.Chatbot(type="messages"),
                        textbox=gr.Textbox(
                            placeholder="Ask something...", container=True
                        ),
                        submit_btn="Submit",
                        stop_btn="Stop",
                        show_progress="hidden",
                    )

            # Knowledge base viewer
            knowledge_df = gr.Dataframe(
                headers=["ID", "Title", "Source"],
                interactive=False,
                label="Knowledge Base Contents",
            )

            # Connect file explorer to document loading
            file_explorer.change(
                fn=self.pipeline.load_documents,
                inputs=file_explorer,
                outputs=knowledge_df,
                show_progress="minimal",
            )

            # Refresh knowledge base button
            refresh_btn = gr.Button("Clear Knowledge Base", variant="primary")
            refresh_btn.click(
                fn=lambda: self.pipeline.clear_knowledge_base(), outputs=None
            )

    def _create_backend_tab(self) -> None:
        """Create the backend management tab."""
        with gr.TabItem("Backend"):
            gr.Markdown("### Database Management Interface")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### File Upload")
                    backend_file_explorer = gr.FileExplorer(
                        root_dir=self.pipeline.get_current_directory(),
                        label="Select Files (.wav, .mp4, .txt, .pdf)",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("#### Database Viewer")
                    database_viewer = gr.Dataframe(
                        label="Database Contents",
                        interactive=False,
                        wrap=True,
                    )

                    refresh_db_btn = gr.Button(
                        "Refresh Database View", variant="secondary"
                    )

            # Connect backend file explorer
            backend_file_explorer.change(
                fn=self.pipeline.load_documents,
                inputs=backend_file_explorer,
                outputs=database_viewer,
                show_progress="minimal",
            )

            # Connect refresh button
            refresh_db_btn.click(
                fn=self.pipeline.get_knowledge_base_status, outputs=database_viewer
            )
