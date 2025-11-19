IELTS-AGENTS: AI-Powered IELTS Training System

An advanced AI-driven IELTS Training System designed to simulate real IELTS Speaking, Reading, Writing, and Listening tests.
The system uses FastAPI, LLM agents, Speech-to-Text (ASR), Text-to-Speech (TTS), vector databases, and modular service-based architecture to deliver a complete English learning and evaluation experience.

🚀 Key Features

IELTS Multi-Module Agents

Speaking Agent

Listening Agent

Reading Agent

Writing Agent

Scoring Agent

Progress Agent

Test Manager Agent

AI Services Layer

ASR Service (Speech-to-Text)

TTS Service (Text-to-Speech)

LLM Service (Question generation, evaluation, agent responses)

Evaluation Service (Fluency, grammar, coherence scoring)

Vector DB Service (FAISS/Pinecone knowledge retrieval)

Workflows

Practice module flow

Mock test workflow (end-to-end IELTS simulation)

FastAPI Backend

REST APIs for all functions

Fully documented using Swagger/OpenAPI

Testing Suite

Unit tests for agents, services, memory, workflows (PyTest)

Speech + AI Integration

Real-time user speech input

Automated scoring and feedback

Generated responses through TTS

📁 Project Structure
IELTS-AGENTS/
│── agents/
│   ├── listening_agent.py
│   ├── progress_agent.py
│   ├── reading_agent.py
│   ├── scoring_agent.py
│   ├── speaking_agent.py
│   ├── test_manager_agent.py
│   └── writing_agent.py
│
│── data/prompts/
│── memory/
│
│── services/
│   ├── asr_service.py
│   ├── evaluation_service.py
│   ├── llm_service.py
│   ├── tts_service.py
│   └── vector_db_service.py
│
│── tests/
│   ├── test_agents.py
│   ├── test_memory.py
│   ├── test_services.py
│   └── test_workflows.py
│
│── utils/
│
│── workflow/
│   ├── mock_test_workflow.py
│   └── practice_module_flow.py
│
│── config.py
│── main.py
│── requirements.txt
│── .gitignore
│── README.md   ← (this file)

🛠️ Tech Stack

Backend: FastAPI (Python 3.x)

AI/LLM: OpenAI / HuggingFace / Custom LLM

Speech Processing: Whisper / Google ASR / Custom ASR

Audio Output: TTS models

Database: FAISS / Pinecone (for vector retrieval)

Testing: PyTest

Tools: Postman, Swagger UI

▶️ How to Run the Project
1️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Start FastAPI Server
uvicorn main:app --reload

4️⃣ Open API Docs (Swagger)
http://127.0.0.1:8000/docs

🧪 Testing

Run all tests:

pytest -v

🎯 Core Modules Explained
1. Agents

Each agent handles one IELTS section using LLM logic, scoring instructions, and ASR/TTS integration.

2. Services

Backend engines for STT, TTS, LLM, scoring, and vector search.

3. Workflows

Combines multiple agents + services to create:

Full mock IELTS test

Practice session modules

4. Memory

Stores session history, user state, context, progress, etc.

📦 API Endpoints (Examples)
Endpoint	Method	Description
/speak/upload-audio	POST	Speech-to-Text transcription
/speak/get-response	POST	Speaking agent reply + scoring
/tts/generate	POST	Convert text to speech
/mock/start	POST	Begin full IELTS mock-test
/progress/score	GET	Retrieve user progress

All endpoints documented in Swagger UI.
