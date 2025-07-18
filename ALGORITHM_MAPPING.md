# Algorithm to Implementation Mapping

This document maps the original algorithm code to the microservices implementation.

## Original Algorithm → Microservices Architecture

### Stage 1: Thread Segmentation

**Original Code:**
```python
def split_pairs(transcript: str) -> List[Dict]:
    raw = chat_simple(PB.split_prompt(transcript))
    threads = safe_json(raw)["threads"]
    # ...
```

**Implemented In:**
- `qa_analysis_service/app/services/analysis_service.py`: `_stage1_segment_conversation()`
- `qa_analysis_service/app/services/prompt_builder.py`: `split_prompt()`

### Stage 2: AI Answer Generation

**Original Code:**
```python
def gen_ai_answers(question: str, passages: List[str]) -> Dict:
    messages = PB.draft_prompt(question, passages)
    # Returns both short and long answers
```

**Implemented In:**
- `qa_analysis_service/app/services/analysis_service.py`: `_stage2_generate_ai_answers()`
- `qa_analysis_service/app/services/prompt_builder.py`: `draft_prompt()`, `SYSTEM_DRAFT`
- `dummy_services/rag_service/main.py`: Returns KB chunks instead of ChromaDB

### Stage 3: Agent Scoring

**Original Code:**
```python
def score_pair(question: str, agent_ans: str, drafts: Dict, kb: List[str]) -> Dict:
    bundle = {...}
    messages = PB.grade_prompt(bundle)
    # Returns score, rationale, and kb_verify
```

**Implemented In:**
- `qa_analysis_service/app/services/analysis_service.py`: `_stage3_score_agent_answer()`
- `qa_analysis_service/app/services/prompt_builder.py`: `grade_prompt()`, `SYSTEM_GRADE`

## Key Differences & Improvements

### 1. Service Architecture
- **Original**: Single script with all logic
- **Implementation**: Microservices with clear separation of concerns

### 2. RAG Service
- **Original**: ChromaDB with embeddings
- **Implementation**: Dummy RAG service returning hardcoded chunks (easily replaceable with real vector DB)

### 3. Configuration
- **Original**: Hardcoded constants
- **Implementation**: Environment-based configuration with Pydantic settings

### 4. Error Handling
- **Original**: Basic try/except
- **Implementation**: Comprehensive error handling with fallbacks

### 5. Models & Types
- **Original**: Dict-based data structures
- **Implementation**: Pydantic models with validation

## Data Flow

1. **Input**: Conversation with messages → `AnalysisRequest`
2. **Stage 1**: Messages → Transcript → LLM → `ConversationThread[]`
3. **Stage 2**: Question → RAG Service → KB Chunks → LLM → `AIAnswers`
4. **Stage 3**: Question + Agent Answer + AI Answers + KB → LLM → `QuestionRating`
5. **Output**: All ratings + overall accuracy → `AnalysisResponse`

## Prompts Preserved

All prompts from the original algorithm are preserved exactly:
- `SYSTEM_DRAFT`: For generating AI answers
- `SYSTEM_GRADE`: For scoring agent answers
- `FEW_SHOT_DRAFT`: Examples for answer generation
- `FEW_SHOT_GRADE`: Examples for scoring

## Integration Points

### To Use Real KB:
1. Replace `dummy_services/rag_service` with actual vector DB service
2. Implement embeddings in the RAG service
3. Return chunks in the same format

### To Use Different LLM:
1. Update `OPENAI_MODEL` in environment
2. Or replace `LLMClient` with different provider

### To Modify Scoring:
1. Update prompts in `prompt_builder.py`
2. Adjust scoring rubric in `SYSTEM_GRADE` 