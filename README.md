# AI Quality Assurance Project

This project contains the `QA Analysis Service`, a microservice responsible for performing comprehensive quality assurance analysis on chat and ticket transcripts.

## Implementation Plan

This plan outlines the steps to develop, test, and deploy the `QA Analysis Service`.

### 1. Project Scaffolding

- [ ] Initialize the project structure based on the `scaffold_project` rule in `.cursor/rules/ai_quality.mdc`. This will create a standard FastAPI service layout.
- [ ] Rename the service from `my_service` to `qa_analysis_service`.
- [ ] Set up a Python virtual environment.
- [ ] Create a `requirements.txt` file with initial dependencies:
    - `fastapi`
    - `uvicorn[standard]`
    - `pydantic`
    - `pydantic-settings`
    - `requests` (for communicating with other services)
    - `pytest` (for testing)

### 2. API Definition

- [ ] **Define Pydantic Models (`app/models`):**
    - Create `AnalysisRequest` model: To accept a chat/ticket transcript string and any relevant metadata.
    - Create `AnalysisResponse` model: To return the analysis results, including:
        - A list of segmented Question & Answer pairs.
        - The AI-generated reference answer for each question.
        - The accuracy assessment for each question.
- [ ] **Create FastAPI Endpoint (`main.py`):**
    - Implement a `POST /analyze` endpoint that accepts `AnalysisRequest` and returns `AnalysisResponse`.
    - Initially, this endpoint can return a hardcoded, valid response for quick testing.

### 3. Dependency Mocking & Integration

- [ ] **Define RAG Service Contract:**
    - Formally define the request/response for the `RAG Service`'s `GET /generate-answer` endpoint.
- [ ] **Create Mock RAG Service:**
    - Develop a simple, local mock server (e.g., using Flask or another FastAPI app) that implements the RAG Service contract. It will take a question and return a canned reference answer. This allows for independent development.
- [ ] **Implement API Client (`app/services`):**
    - Create a client class responsible for making HTTP requests to the RAG Service.

### 4. Core Service Logic (`app/services`)

- [ ] **Create `AnalysisService`:**
    - Implement a service class in `app/services/analysis_service.py` to encapsulate the core business logic.
- [ ] **Implement Transcript Segmentation:**
    - Develop the algorithm to parse a raw transcript and segment it into a structured list of Q&A pairs.
- [ ] **Integrate RAG Service:**
    - For each question identified in the segmentation step, use the API client to call the (mock) RAG service to fetch the reference answer.
- [ ] **Implement Accuracy Assessment:**
    - Develop the logic to compare the agent's answer with the RAG-provided reference answer and generate an accuracy score or rating.

### 5. Configuration

- [ ] **Implement Settings Management (`app/config.py`):**
    - Use Pydantic's `BaseSettings` to manage configuration from environment variables (`.env` file).
    - Key configuration will include `RAG_SERVICE_URL`. For local development, this will point to our mock RAG service.

### 6. Testing (`/tests`)

- [ ] **Unit Tests:**
    - Write tests for the transcript segmentation logic with various transcript formats.
    - Write tests for the accuracy assessment logic.
- [ ] **Integration Tests:**
    - Use FastAPI's `TestClient` to write tests for the `POST /analyze` endpoint.
    - These tests will mock the outbound call to the RAG service to ensure the service behaves correctly without a live dependency.

### 7. Containerization & CI/CD

- [ ] **Create `Dockerfile`:**
    - Write a `Dockerfile` to containerize the `QA Analysis Service` for consistent deployment.
- [ ] **Set up CI Pipeline (`.github/workflows/ci.yml`):**
    - Create a GitHub Actions workflow that automatically runs on every push to:
        - Install dependencies.
        - Run linters (`black`, `flake8`).
        - Run static type checks (`mypy`).
        - Run all tests with `pytest`.
