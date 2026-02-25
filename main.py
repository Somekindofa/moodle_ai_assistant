"""
Backend server entry point for Moodle AI Assistant.

This is the new production-ready entry point that launches the FastAPI server
for the JavaScript frontend integration.
"""

import logging
import uvicorn
from config.settings import setup_logging


def main():
    """Main application entry point."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Moodle AI Assistant Backend Server...")

    try:
        # Launch FastAPI server
        uvicorn.run(
            "server:app", host="127.0.0.1", port=8000, reload=True, log_level="info"
        )

    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
