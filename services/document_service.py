"""Document processing service for loading and splitting documents."""

import logging
import os
from typing import List, Type, Optional
from pathlib import Path

from langchain_core.documents.base import Document
from langchain_text_splitters import CharacterTextSplitter, TextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader

from config.settings import ConfigurationManager


logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """Service for processing and loading documents."""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()

    def load_and_split_documents(
        self,
        file_paths: List[str],
        splitter_cls: Type[TextSplitter] = CharacterTextSplitter,
    ) -> List[Document]:
        """Load and split documents from file paths."""
        if not file_paths:
            raise ValueError("No files provided for processing")

        splitter = splitter_cls()
        all_documents = []

        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.error(f"File does not exist: {file_path}")
                    continue

                documents = self._load_single_file(file_path, splitter)
                all_documents.extend(documents)

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue

        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents

    def _load_single_file(
        self, file_path: str, splitter: TextSplitter
    ) -> List[Document]:
        """Load and split a single file."""
        file_extension = Path(file_path).suffix.lower()

        # Select appropriate loader based on file type
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path=file_path)
        elif file_extension in [".txt", ".md"]:
            loader = TextLoader(file_path=file_path, encoding="utf-8")
        else:
            logger.warning(
                f"Unsupported file type: {file_extension} for file {file_path}"
            )
            return []

        # Load and split the document
        return loader.load_and_split(text_splitter=splitter)

    def is_supported_file_type(self, file_path: str) -> bool:
        """Check if file type is supported."""
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.config.supported_file_types

    def filter_supported_files(self, file_paths: List[str]) -> List[str]:
        """Filter list to only include supported file types."""
        return [
            file_path
            for file_path in file_paths
            if self.is_supported_file_type(file_path)
        ]
