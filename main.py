"""
Refactored main application for Moodle AI Assistant.

This is the new production-ready entry point that uses the modular pipeline architecture.
"""

import logging
from config.settings import ConfigurationManager, setup_logging
from pipeline import MoodleAIAssistantPipeline
from ui.gradio_interface import MoodleAIAssistantUI


def main():
    """Main application entry point."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Moodle AI Assistant...")

    try:
        # Initialize configuration
        config_manager = ConfigurationManager()

        # Initialize pipeline
        pipeline = MoodleAIAssistantPipeline(config_manager)

        # Create UI
        ui = MoodleAIAssistantUI(pipeline)
        interface = ui.create_interface()

        # Launch application
        logger.info("Launching Gradio interface...")
        interface.launch()

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise


if __name__ == "__main__":
    main()
