"""
Quick test script to verify server setup.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import fastapi
    print(f"FastAPI version: {fastapi.__version__}")
except ImportError as e:
    print(f"FastAPI not available: {e}")

try:
    import uvicorn
    print(f"Uvicorn available: True")
except ImportError as e:
    print(f"Uvicorn not available: {e}")

try:
    import sse_starlette
    print(f"SSE Starlette available: True")
except ImportError as e:
    print(f"SSE Starlette not available: {e}")

# Test server imports
try:
    from api.routes import router
    print("API routes import: Success")
except ImportError as e:
    print(f"API routes import failed: {e}")

try:
    from server import app
    print("Server app import: Success")
except ImportError as e:
    print(f"Server app import failed: {e}")

print("\nServer setup verification complete!")
