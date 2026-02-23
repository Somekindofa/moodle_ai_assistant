"""
MIGRATION GUIDE: From Legacy app.py to New Modular Architecture

## Overview
The original monolithic app.py has been refactored into a modular, production-ready architecture with proper separation of concerns.

## New Architecture Components

### 1. Configuration Layer (`config/`)
- `settings.py`: Centralized configuration management
  - Environment variable loading and validation
  - Application settings with dataclasses
  - Logging setup

### 2. Core Types (`core/`)
- `types.py`: Shared type definitions
  - ConversationState TypedDict

### 3. Services Layer (`services/`)
- `langchain_service.py`: LangChain client and prompt management
- `rag_service.py`: RAG operations (retrieval and generation)
- `document_service.py`: Document loading and processing
- `graph_service.py`: Conversation workflow management

### 4. Pipeline Layer (`pipeline.py`)
- Orchestrates all services
- Provides high-level interface for UI
- Manages document loading and conversation flow

### 5. UI Layer (`ui/`)
- `gradio_interface.py`: Gradio UI components
- Separated from business logic

### 6. Main Entry Point (`main.py`)
- Clean application startup
- Dependency injection setup

## Key Improvements

### Separation of Concerns
- Configuration isolated from business logic
- UI separated from backend services
- Each service has a single responsibility

### Dependency Injection
- Services receive dependencies through constructors
- Easy to test and mock individual components
- Clear dependency graph

### Error Handling
- Centralized logging
- Proper exception handling at service boundaries
- Graceful degradation

### Type Safety
- Strong typing throughout
- Clear interfaces between components
- Better IDE support and refactoring

### Testability
- Each service can be tested independently
- Mock-friendly interfaces
- Clear separation of pure functions and side effects

## Migration Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Update Environment Variables
Ensure your `.env` file contains:
```
FIREWORKS_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here
```

### 3. Run New Application
```bash
python main.py
```

### 4. Verify Functionality
- Test document loading
- Test chat interface
- Verify knowledge base management

## Legacy Code Mapping

### Old Classes → New Services
- `EnvLoader` → `ConfigurationManager` (config/settings.py)
- `LangInit` → `LangChainService` (services/langchain_service.py)
- `RAG` → `RAGService` (services/rag_service.py)
- `GraphBuilder` → `ConversationGraphService` (services/graph_service.py)

### Old Functions → New Methods
- `load_and_split()` → `DocumentProcessingService.load_and_split_documents()`
- `generate_answer()` → `MoodleAIAssistantPipeline.generate_response()`
- `_load_to_dataframe()` → `MoodleAIAssistantPipeline._create_knowledge_base_dataframe()`

## Benefits of New Architecture

### 1. Maintainability
- Clear module boundaries
- Single responsibility per class
- Easy to understand and modify

### 2. Scalability
- Easy to add new services
- Pluggable components
- Configuration-driven behavior

### 3. Testing
- Unit testable services
- Mock-friendly interfaces
- Integration test boundaries

### 4. Deployment
- Clear dependency management
- Environment-specific configurations
- Production-ready logging

### 5. Development Experience
- Better IDE support
- Clear error messages
- Type safety and auto-completion

## Future Extensions

The new architecture makes it easy to add:
- Multiple LLM providers
- Different embedding models
- Authentication and authorization
- API endpoints
- Background job processing
- Caching layers
- Monitoring and metrics

## Rollback Plan

If you need to rollback to the legacy version:
1. Keep `app.py` as `app_legacy.py`
2. The new architecture is fully backward compatible
3. All functionality is preserved and enhanced
"""
