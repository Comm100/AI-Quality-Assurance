# AI Quality Assurance Project

This project contains the `QA Analysis Service`, a microservice responsible for performing comprehensive quality assurance analysis on chat and ticket transcripts.

## 🏗️ Architecture

The project consists of three services:

1. **QA Analysis Service** (Port 8000) - Main service that analyzes chat transcripts
2. **Chat Data Service** (Port 8001) - Dummy service providing sample chat transcripts  
3. **RAG Service** (Port 8002) - Dummy service providing reference answers from knowledge base

## 🚀 Quick Start

### Option 1: Using Python Scripts (Recommended for Development)

1. **Install Dependencies**
   ```bash
   # Install QA Analysis Service dependencies
   cd qa_analysis_service
   pip install -r requirements.txt
   cd ..

   # Install dummy services dependencies  
   pip install -r dummy_services/requirements.txt
   ```

2. **Start All Services**
   ```bash
   python scripts/start_all_services.py
   ```

3. **Test Services**
   ```bash
   # In another terminal
   python scripts/test_services.py
   ```

### Option 2: Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Test services
python scripts/test_services.py
```

## 📋 API Usage

### QA Analysis Service

**Analyze a transcript:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Agent: Hello! How can I help?\nCustomer: I need help with notifications.\nAgent: Go to Settings > Notifications.",
    "transcript_id": "chat_001"
  }'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

### Chat Data Service

**Get sample transcripts:**
```bash
curl http://localhost:8001/transcripts
```

**Get specific transcript:**
```bash
curl http://localhost:8001/transcripts/chat_001
```

### RAG Service

**Generate reference answer:**
```bash
curl -X POST "http://localhost:8002/generate-answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I enable email notifications?",
    "context": "User needs help with settings"
  }'
```

## 🧪 Testing

Run the test suite:
```bash
cd qa_analysis_service
pytest tests/ -v
```

## 📊 Service Documentation

- QA Analysis Service: http://localhost:8000/docs
- Chat Data Service: http://localhost:8001/docs  
- RAG Service: http://localhost:8002/docs

## Implementation Plan

This plan outlines the steps to develop, test, and deploy the `QA Analysis Service`.

### 1. Project Scaffolding ✅

- [x] Initialize the project structure based on the `scaffold_project` rule in `.cursor/rules/ai_quality.mdc`. This will create a standard FastAPI service layout.
- [x] Rename the service from `my_service` to `qa_analysis_service`.
- [x] Set up a Python virtual environment.
- [x] Create a `requirements.txt` file with initial dependencies:
    - `fastapi`
    - `uvicorn[standard]`
    - `pydantic`
    - `pydantic-settings`
    - `requests` (for communicating with other services)
    - `pytest` (for testing)

### 2. API Definition ✅

- [x] **Define Pydantic Models (`app/models`):**
    - Create `AnalysisRequest` model: To accept a chat/ticket transcript string and any relevant metadata.
    - Create `AnalysisResponse` model: To return the analysis results, including:
        - A list of segmented Question & Answer pairs.
        - The AI-generated reference answer for each question.
        - The accuracy assessment for each question.
- [x] **Create FastAPI Endpoint (`main.py`):**
    - Implement a `POST /analyze` endpoint that accepts `AnalysisRequest` and returns `AnalysisResponse`.

### 3. Dependency Mocking & Integration ✅

- [x] **Define RAG Service Contract:**
    - Formally define the request/response for the `RAG Service`'s `POST /generate-answer` endpoint.
- [x] **Create Mock RAG Service:**
    - Develop a simple, local mock server using FastAPI that implements the RAG Service contract. It takes a question and returns a reference answer with confidence scores.
- [x] **Create Mock Chat Data Service:**
    - Develop a dummy service that provides sample chat transcripts for testing.
- [x] **Implement API Client (`app/services`):**
    - Create a client class responsible for making HTTP requests to the RAG Service.

### 4. Core Service Logic (`app/services`) ✅

- [x] **Create `AnalysisService`:**
    - Implement a service class in `app/services/analysis_service.py` to encapsulate the core business logic.
- [x] **Implement Transcript Segmentation:**
    - Develop the algorithm to parse a raw transcript and segment it into a structured list of Q&A pairs.
- [x] **Integrate RAG Service:**
    - For each question identified in the segmentation step, use the API client to call the (mock) RAG service to fetch the reference answer.
- [x] **Implement Accuracy Assessment:**
    - Develop the logic to compare the agent's answer with the RAG-provided reference answer and generate an accuracy score or rating.

### 5. Configuration ✅

- [x] **Implement Settings Management (`app/config.py`):**
    - Use Pydantic's `BaseSettings` to manage configuration from environment variables (`.env` file).
    - Key configuration includes `RAG_SERVICE_URL` and `CHAT_DATA_SERVICE_URL`.

### 6. Testing (`/tests`) ✅

- [x] **Unit Tests:**
    - Write tests for the transcript segmentation logic with various transcript formats.
    - Write tests for the accuracy assessment logic.
- [x] **Integration Tests:**
    - Use FastAPI's `TestClient` to write tests for the `POST /analyze` endpoint.
    - These tests mock the outbound call to the RAG service to ensure the service behaves correctly without a live dependency.

### 7. Containerization & CI/CD ✅

- [x] **Create `Dockerfile`:**
    - Write a `Dockerfile` to containerize the `QA Analysis Service` for consistent deployment.
- [x] **Create Docker Compose:**
    - Set up `docker-compose.yml` to run all services together.
- [ ] **Set up CI Pipeline (`.github/workflows/ci.yml`):**
    - Create a GitHub Actions workflow that automatically runs on every push to:
        - Install dependencies.
        - Run linters (`black`, `flake8`).
        - Run static type checks (`mypy`).
        - Run all tests with `pytest`.

## 🔧 Development Notes

- All services use FastAPI for consistency
- The RAG service uses simple keyword matching for demo purposes
- The QA Analysis Service gracefully handles RAG service failures with fallback scoring
- Services include comprehensive health checks and logging
- Docker setup allows for easy deployment and scaling

## 📝 Next Steps

1. Set up GitHub Actions CI/CD pipeline
2. Add more sophisticated accuracy assessment algorithms
3. Implement proper authentication and authorization
4. Add monitoring and metrics collection
5. Scale for production workloads
