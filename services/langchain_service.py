"""LangChain service for managing LangChain components."""

import logging
from typing import Optional
from langsmith import Client

from config.settings import ConfigurationManager


logger = logging.getLogger(__name__)


class LangChainService:
    """Service for managing LangChain components and connections."""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.client: Optional[Client] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize LangChain client with API key."""
        try:
            api_key = (
                self.config_manager.get_env_var("LANGSMITH_API_KEY")
                or self.config_manager.get_env_var("LANGCHAIN_API_KEY")
            )
            if not api_key:
                logger.warning(
                    "No LangChain API key found. Please check your environment variables."
                )
                return

            self.client = Client(api_key=api_key)
            logger.info("LangChain client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LangChain client: {str(e)}")

    def get_client(self) -> Optional[Client]:
        """Get the LangChain client."""
        return self.client
