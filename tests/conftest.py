"""Shared pytest configuration for all tests."""

import sys
import os
from unittest.mock import MagicMock, patch
from pathlib import Path

# Pre-patch pydantic_settings.DotEnvSettingsSource before any imports
# This prevents the .env file permission error
original_open = open

def patched_open_for_env(*args, **kwargs):
    """Patch open() to block .env file reads."""
    if args:
        filepath = str(args[0])
        if filepath.endswith('.env'):
            # Return an empty file-like object for .env
            from io import StringIO
            return StringIO("")
    return original_open(*args, **kwargs)

# Monkey-patch dotenv before chromadb loads
import dotenv
dotenv.load_dotenv = lambda *args, **kwargs: None
dotenv.dotenv_values = lambda *args, **kwargs: {}

# Also block the actual file open for .env files
builtins = __import__('builtins')
builtins.open = patched_open_for_env

