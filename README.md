# AI Quality Assurance System

A comprehensive microservices-based system for analyzing and scoring customer support conversations using a sophisticated 3-stage AI algorithm.

## Overview

This system implements an advanced quality assurance algorithm that:
1. **Stage 1**: Segments conversations into meaningful Q&A threads using LLM
2. **Stage 2**: Retrieves relevant KB chunks and generates AI reference answers (short & long)
3. **Stage 3**: Scores agent responses against AI references and KB evidence

## Architecture

The system consists of three microservices:

### 1. Chat Data Service (Port 8001)
- Provides sample chat transcripts
- Simulates a real chat data source

### 2. RAG Service v2 (Port 8002)  
- Retrieves relevant knowledge base chunks
- Returns multiple KB passages with confidence scores
- Supports configurable chunk retrieval (k parameter)

### 3. QA Analysis Service (Port 8000)
- Implements the 3-stage analysis algorithm
- Uses OpenAI GPT for intelligent processing
- Provides comprehensive scoring and rationale

## 3-Stage Algorithm Details

### Stage 1: Conversation Segmentation
- Uses LLM to intelligently group customer messages by intent
- Pairs customer questions with corresponding agent answers
- Handles multi-turn conversations effectively

### Stage 2: AI Answer Generation
- Retrieves relevant KB chunks using semantic search
- Generates two AI reference answers:
  - **Short Answer**: Concise, direct response
  - **Long Answer**: Detailed explanation with context
- Strictly grounds answers in KB evidence

### Stage 3: Agent Scoring
- Compares agent answers against AI references
- Scores on a scale from -1 to 5:
  - 5: Perfect alignment with KB
  - 4: Minor discrepancies
  - 3: Good but missing details
  - 2: Partially correct
  - 1: Mostly incorrect
  - 0: Completely wrong
  - -1: Out of scope (not in KB)
- Provides detailed rationale for each score

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- OpenAI API key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Quality-Assurance
```

2. Create a `.env` file with your OpenAI API key:
```bash
# Copy from .env.example if available
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.0
```

3. Install dependencies (optional for local development):
```bash
pip install -r qa_analysis_service/requirements.txt
```

## Running the Services

### Using Docker Compose (Recommended)
```bash
docker-compose up --build
```

### Using Python Script
```bash
python scripts/start_all_services.py
```

### Manual Start (for development)
```bash
# Terminal 1: Chat Data Service
cd dummy_services && python chat_data_service/main.py

# Terminal 2: RAG Service
cd dummy_services && python rag_service/main.py

# Terminal 3: QA Analysis Service
cd qa_analysis_service && python main.py
```

## Testing

Run the comprehensive test suite:
```bash
python scripts/test_services.py
```

This will:
- Verify all services are healthy
- Test the RAG chunk retrieval
- Run a complete 3-stage analysis on sample conversations
- Display detailed results including scores and rationale

## API Endpoints

### QA Analysis Service
- `GET /` - Service info
- `GET /health` - Health check
- `POST /analyze` - Analyze a conversation

### RAG Service v2
- `GET /` - Service info
- `GET /health` - Health check
- `POST /retrieve-chunks` - Retrieve KB chunks for a question

### Chat Data Service
- `GET /` - Service info
- `GET /transcripts` - Get sample transcripts
- `GET /transcript/{id}` - Get specific transcript

## Example Analysis Request

```json
{
  "conversation": {
    "id": 12345,
    "type": "chat",
    "messages": [
      {
        "id": "1",
        "role": "customer",
        "content": "How do I filter unpaid invoices?",
        "timestamp": "2024-01-15T08:50:00Z"
      },
      {
        "id": "2", 
        "role": "agent",
        "content": "Go to Billing → Invoices and use the Status dropdown.",
        "timestamp": "2024-01-15T08:51:00Z"
      }
    ]
  },
  "integratedKbId": "kb_001"
}
```

## Example Response

```json
{
  "conversationId": 12345,
  "conversationType": "chat",
  "analysisTime": "2024-01-15T10:00:00Z",
  "overallAccuracy": 4.5,
  "questionRatings": [
    {
      "aiRewrittenQuestion": "How do I filter unpaid invoices?",
      "agentAnswer": "Go to Billing → Invoices and use the Status dropdown.",
      "aiShortAnswer": "Select Unpaid in Status filter.",
      "aiLongAnswer": "On the Invoices page, use the Status dropdown to choose Unpaid...",
      "aiScore": 4.5,
      "aiRationale": "Agent provided correct navigation but could mention 'Unpaid' option explicitly.",
      "kbVerify": [
        "On the Invoices page, use the Status dropdown to choose Unpaid. (source: billing/invoices.md)"
      ]
    }
  ]
}
```

## Development

### Adding New KB Content
Edit `dummy_services/rag_service/main.py` to add new knowledge base chunks in the `KNOWLEDGE_BASE_CHUNKS` dictionary.

### Modifying Prompts
Edit `qa_analysis_service/app/services/prompt_builder.py` to adjust:
- System prompts for each stage
- Few-shot examples
- Scoring rubrics

### Changing Models
Update the `.env` file or environment variables:
- `OPENAI_MODEL`: Change to `gpt-4`, `gpt-3.5-turbo`, etc.
- `OPENAI_TEMPERATURE`: Adjust for more/less deterministic responses

## Troubleshooting

1. **Services not starting**: Check port availability (8000, 8001, 8002)
2. **OpenAI errors**: Verify API key is set correctly
3. **Low scores**: Check if KB chunks match the conversation topic
4. **Timeout errors**: Increase timeout in client code for slow LLM responses

## License

[Your License Here]
