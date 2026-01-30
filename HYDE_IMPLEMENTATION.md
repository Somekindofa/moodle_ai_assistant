# HyDE (Hypothetical Document Embeddings) Implementation

## Overview

This branch implements a **HyDE-based RAG pipeline** to improve retrieval accuracy for vague novice queries (e.g., "how do I hold my blowpipe?") against expert elicitation transcripts.

## Architecture Changes

### Previous Pipeline (3 nodes):
```
retrieve → enhance_query → retrieve_final → generate
```
- **Problem**: Hand-wavy queries didn't match expert transcript language
- **Inefficiency**: Double retrieval (k=15 each), extra LLM call for enhancement
- **Bootstrap issue**: Query enhancement depended on initial retrieval quality

### New HyDE Pipeline (2 nodes):
```
generate_hypothetical_document → retrieve_with_hyde → generate
```

## Implementation Details

### 1. New Method: `generate_hypothetical_document`
**Location**: `services/rag_service.py`

**Purpose**: Generates a synthetic expert elicitation that matches the linguistic style of actual transcripts

**Key Features**:
- Uses French prompts (matching project language)
- Focuses on first-person, technical, movement-based descriptions
- Generates 2-3 paragraphs in expert demonstration style
- Returns `hypothetical_document` in state

**Prompt Template**:
- Requests precise hand positioning and grip descriptions
- Asks for tool angles, movement timing, physical sensations
- Includes technical terminology
- Mimics "thinking aloud during demonstration" style

### 2. New Method: `retrieve_with_hyde`
**Location**: `services/rag_service.py`

**Purpose**: Single-pass retrieval using HyDE document instead of original query

**Key Features**:
- Uses hypothetical document for embedding similarity search (k=5)
- Falls back to original query if HyDE generation fails
- Extracts video metadata from top result
- Single retrieval pass (vs. 2 in old approach)

**Efficiency Gains**:
- 50% reduction in vector searches (1 vs. 2)
- Reduced k value (5 vs. 15) since HyDE improves precision
- One LLM call total (HyDE generation only)

### 3. State Management
**Location**: `core/types.py`

**New Field**: `hypothetical_document: Optional[str]`
- Stores HyDE-generated expert elicitation
- Passed from generation node to retrieval node

**Legacy Field**: `enhanced_query` marked as [LEGACY]

### 4. Pipeline Orchestration
**Location**: `pipeline.py`

**Updated Graph**: 
```python
functions=["generate_hypothetical_document", "retrieve_with_hyde", "generate"]
```

**Stream Handler Changes**:
- Monitors `generate_hypothetical_document_runnable` node
- Logs HyDE document preview (first 100 chars)
- Tracks `retrieve_with_hyde_runnable` for context and video metadata
- Yields accumulated context/metadata to frontend

### 5. Legacy Methods
The following methods are preserved but marked `[LEGACY]`:
- `retrieve()` - Initial retrieval for query enhancement
- `enhance_query()` - LLM-based query reformulation
- `retrieve_final()` - Second retrieval pass with enhanced query

These can be removed after HyDE testing confirms superior performance.

## Performance Comparison

| Metric | Old Approach | HyDE Approach | Improvement |
|--------|-------------|---------------|-------------|
| Vector Searches | 2 (k=15 each) | 1 (k=5) | -83% search cost |
| LLM Calls | 1 (enhancement) | 1 (HyDE gen) | Same |
| Total k retrieved | 30 docs | 5 docs | -83% embedding ops |
| Context used | 1 doc | 5 docs | Better coverage |
| Latency | Higher | Lower | Faster |

## Why HyDE Works Better

1. **Semantic Bridge**: Generates text in same linguistic style as expert elicitations
2. **No Bootstrap Problem**: Doesn't depend on initial retrieval quality
3. **Domain Adaptation**: Prompt customized for first-person, technical, movement-focused language
4. **Better Precision**: Single targeted search vs. broad initial retrieval

## Testing Recommendations

1. **Compare Retrieval Quality**:
   - Test vague queries: "how do I hold X", "what's the technique for Y"
   - Measure if retrieved docs are more relevant with HyDE
   
2. **Latency Testing**:
   - Measure end-to-end response time vs. old approach
   - Should be ~30-50% faster due to single retrieval

3. **Ablation Study**:
   - Test with HyDE generation
   - Test without (direct query)
   - Quantify improvement

## Usage

The pipeline automatically uses HyDE when:
- Documents exist in vector store
- User sends a query
- No changes needed to frontend/API

## Future Enhancements

1. **Few-Shot HyDE**: Include 1-2 example elicitations in prompt
2. **Hybrid Retrieval**: Combine HyDE with keyword search (fusion)
3. **Adaptive k**: Dynamically adjust retrieval count based on query complexity
4. **Multi-HyDE**: Generate multiple hypothetical documents and aggregate results

## Rollback Plan

To revert to old approach:
1. Change `pipeline.py` graph functions to: `["retrieve", "enhance_query", "retrieve_final", "generate"]`
2. Update `generate_response` node handlers to old names
3. No other changes needed (legacy methods preserved)
