# Quick Start Guide

## Prerequisites
- Python 3.12+
- OpenAI API Key

## 1. Set Up Environment

```bash
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
echo "OPENAI_MODEL=gpt-4o-mini" >> .env
echo "OPENAI_TEMPERATURE=0.0" >> .env
```

## 2. Start Services

```bash
# Using Docker
docker-compose up --build

# OR using Python
python scripts/start_all_services.py
```

## 3. Test the System

```bash
python scripts/test_services.py
```

## 4. Send Analysis Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
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
          "content": "Go to Billing → Invoices and use Status dropdown.",
          "timestamp": "2024-01-15T08:51:00Z"
        }
      ]
    },
    "integratedKbId": "kb_001"
  }'
```

## 5. Expected Response

```json
{
  "conversationId": 12345,
  "conversationType": "chat",
  "overallAccuracy": 4.5,
  "questionRatings": [
    {
      "aiRewrittenQuestion": "How do I filter unpaid invoices?",
      "agentAnswer": "Go to Billing → Invoices and use Status dropdown.",
      "aiShortAnswer": "Select Unpaid in Status filter.",
      "aiLongAnswer": "On the Invoices page, use the Status dropdown...",
      "aiScore": 4.5,
      "aiRationale": "Agent provided correct navigation...",
      "kbVerify": ["On the Invoices page..."]
    }
  ]
}
```

## Key Features

### 3-Stage Analysis
1. **Segmentation**: Groups customer messages by intent
2. **AI Answers**: Generates reference answers from KB
3. **Scoring**: Compares agent vs AI answers

### Scoring Scale
- **5**: Perfect match with KB
- **4**: Minor issues
- **3**: Good but missing details
- **2**: Partially correct
- **1**: Mostly incorrect
- **0**: Completely wrong
- **-1**: Out of scope (not in KB)

## Customization

### Change LLM Model
Edit `.env`:
```bash
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
```

### Add KB Content
Edit `dummy_services/rag_service/main.py`:
```python
KNOWLEDGE_BASE_CHUNKS = {
    "your_topic": [
        {
            "content": "Your KB content",
            "source": "your_source.md",
            "confidence": 0.95
        }
    ]
}
```

### Modify Prompts
Edit `qa_analysis_service/app/services/prompt_builder.py`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OpenAI errors | Check API key in .env |
| Services not starting | Check ports 8000-8002 |
| Low scores | Verify KB content matches topic |
| Timeouts | Increase timeout in scripts |

## API Documentation

- QA Analysis: http://localhost:8000/docs
- RAG Service: http://localhost:8002/docs
- Chat Data: http://localhost:8001/docs 