"""Configuration management for the Moodle AI Assistant."""

import os
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv, dotenv_values
from dataclasses import dataclass, field


# Configure logging
def setup_logging() -> logging.Logger:
    """Setup application logging configuration."""
    # Configure the root logger instead of just the current module's logger
    root_logger = logging.getLogger()

    if not root_logger.handlers:  # Avoid duplicate handlers
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s   %(levelname)s   %(name)s:   %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)

    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
    # Also return a logger for this module
    logger = logging.getLogger(__name__)
    return logger


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    collection_name: str = "moodle_assistant_collection"
    persist_directory: str = "./chroma_langchain_db"
    # Infomaniak embedding model (bge_multilingual_gemma2, 3584-dim, SOTA FR-MTEB)
    embedding_model: str = "bge_multilingual_gemma2"
    # Infomaniak LLM — OpenAI-compatible endpoint
    llm_model: str = "swiss-ai/Apertus-70B-Instruct-2509"
    prompt_url: str = "rlm/rag-prompt"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 1200
    llm_top_p: float = 0.9
    llm_frequency_penalty: float = 0.2
    llm_presence_penalty: float = 0.1
    similarity_search_k: int = 15


@dataclass
class AppConfig:
    """Main application configuration."""

    rag: RAGConfig = field(default_factory=RAGConfig)
    required_env_keys: List[str] = field(
        default_factory=lambda: ["INFOMANIAK_API_KEY", "INFOMANIAK_PRODUCT_ID", "LANGCHAIN_API_KEY"]
    )
    css_path: str = "css/custom.css"
    supported_file_types: List[str] = field(
        default_factory=lambda: [".pdf", ".txt", ".md"]
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
