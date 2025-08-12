"""Configuration management for the Moodle AI Assistant."""

import os
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv, dotenv_values
from dataclasses import dataclass, field


# Configure logging
def setup_logging() -> logging.Logger:
    """Setup application logging configuration."""
    logger = logging.getLogger(__name__)

    if not logger.handlers:  # Avoid duplicate handlers
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s   %(levelname)s   %(name)s:   %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    return logger


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    collection_name: str = "moodle_assistant_collection"
    persist_directory: str = "./chroma_langchain_db"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model_url: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    llm_provider: str = "fireworks"
    prompt_url: str = "rlm/rag-prompt"
    similarity_search_k: int = 4


@dataclass
class AppConfig:
    """Main application configuration."""

    rag: RAGConfig = field(default_factory=RAGConfig)
    required_env_keys: List[str] = field(
        default_factory=lambda: ["FIREWORKS_API_KEY", "LANGCHAIN_API_KEY"]
    )
    css_path: str = "css/custom.css"
    supported_file_types: List[str] = field(
        default_factory=lambda: [".pdf", ".txt", ".md", ".wav", ".mp4"]
    )


class ConfigurationManager:
    """Manages application configuration and environment variables."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.env_vars: Dict[str, Any] = {}
        self.logger = setup_logging()
        self._load_environment()
        self._validate_environment()

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        load_dotenv()
        self.env_vars = dotenv_values()
        self.logger.info("Environment variables loaded")

    def _validate_environment(self) -> None:
        """Validate that all required environment variables are present."""
        missing_keys = []

        for key in self.config.required_env_keys:
            value = self.env_vars.get(key) or os.getenv(key)
            if value:
                self.logger.info(
                    f"Successfully loaded {key}: {value[:4]}...{value[-4:]}"
                )
            else:
                self.logger.warning(f"{key} not found in environment variables")
                missing_keys.append(key)

        if missing_keys:
            self.logger.warning(
                f"Missing required environment variables: {', '.join(missing_keys)}"
            )

    def get_env_var(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value."""
        return self.env_vars.get(key) or os.getenv(key, default)

    def get_config(self) -> AppConfig:
        """Get application configuration."""
        return self.config
