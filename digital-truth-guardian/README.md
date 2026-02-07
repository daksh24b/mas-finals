# 🛡️ Digital Truth Guardian

### *A Self-Correcting, Agentic Immune System for Digital Trust*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red.svg)](https://qdrant.tech/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)

> Moving beyond linear RAG to **Intentional Orchestration**—a robust Multi-Agent System (MAS) that actively verifies, remembers, and safeguards truth using Qdrant & Gemini.

---

## 🎯 Overview

Digital Truth Guardian is a production-grade AI system designed to combat the societal challenge of misinformation. Unlike standard chatbots that hallucinate when faced with unknown data, this system acts as a **Digital Immune System**.

### Key Features

- **5-Agent Orchestration Squad** led by a Planner-Router that decomposes complex queries
- **Feedback Loop** for self-correction when evidence is insufficient
- **Qdrant Hybrid Search** with dense (semantic) + sparse (keyword) vectors
- **Temporal Memory Evolution** distinguishing between immutable facts and transient states
- **Trusted Source Protocol** ensuring unverified claims are never memorized
- **Episodic Memory** for learning from past agent decisions and outcomes
- **Shared Context Memory** enabling inter-agent communication via vector store
- **Dynamic Tool Selection** autonomously choosing optimal tools based on query type

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIGITAL TRUTH GUARDIAN                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query ──▶ [PLANNER] ──▶ Intent Classification              │
│                    │                                             │
│                    ├── Conversational? ──▶ Direct Response       │
│                    │                                             │
│                    └── Informational? ──▶ [RETRIEVER]            │
│                                              │                   │
│                         ┌────────────────────┤                   │
│                         │                    │                   │
│                         ▼                    │                   │
│                   Cache Hit? ────────────────┤                   │
│                    │    │                    │                   │
│                   Yes   No                   │                   │
│                    │    │                    │                   │
│                    │    └──▶ [EXECUTOR] ◀───┘                   │
│                    │              │ (Tavily Search)              │
│                    │              │                              │
│                    │              ▼                              │
│                    └────────▶ [CRITIC] ◀── Evidence              │
│                                  │                               │
│                         ┌───────┴───────┐                       │
│                         │               │                       │
│                    Sufficient?     Insufficient?                │
│                         │               │                       │
│                         ▼               └──▶ Feedback Loop       │
│                    [ARCHIVIST]                                   │
│                         │                                        │
│                         ▼                                        │
│                   Verdict + Response                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The 5-Agent Squad

| Agent | Model | Role |
|-------|-------|------|
| **Planner** | Gemini Flash | Intent classification, task decomposition, dynamic tool selection, routing |
| **Retriever** | Code-based | Qdrant hybrid search (dense + sparse vectors), episodic memory recording |
| **Executor** | Code-based | External web search via Tavily AI with trusted source filtering |
| **Critic** | Gemini Pro | Deep reasoning, entailment checking, verdict, shares insights via shared context |
| **Archivist** | Gemini Flash | Memory management, temporal versioning, fact classification |

### Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-COLLECTION MEMORY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  KNOWLEDGE BASE │  │ EPISODIC MEMORY │  │ SHARED CONTEXT  │  │
│  │                 │  │                 │  │                 │  │
│  │  Verified facts │  │  Past agent     │  │  Inter-agent    │  │
│  │  & claims with  │  │  decisions &    │  │  communication  │  │
│  │  temporal       │  │  outcomes for   │  │  & coordination │  │
│  │  versioning     │  │  learning       │  │                 │  │
│  │                 │  │                 │  │                 │  │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │  │
│  │  │Dense+Sparse│  │  │  │Dense Only │  │  │  │Dense Only │  │  │
│  │  │  Hybrid   │  │  │  │ Semantic  │  │  │  │ Semantic  │  │  │
│  │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for Qdrant)
- API Keys:
  - [Google Gemini API](https://aistudio.google.com/)
  - [Tavily AI API](https://tavily.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/digital-truth-guardian.git
   cd digital-truth-guardian
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Start Qdrant**
   ```bash
   docker-compose up -d qdrant
   ```

6. **Initialize database**
   ```bash
   python -m src.main init
   ```

### Usage

#### CLI Mode
```bash
# Verify a single claim
python -m src.main verify "The Earth is approximately 4.5 billion years old"

# Interactive chat mode
python -m src.main chat
```

#### API Mode
```bash
# Start the API server
python -m src.main serve

# Make a request
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"claim": "The Earth is flat"}'
```

---

## 📁 Project Structure

```
digital-truth-guardian/
├── src/
│   ├── agents/               # 5-Agent implementations
│   │   ├── base.py           # Base agent with episodic memory & shared context
│   │   ├── planner.py        # Intent classification, dynamic tool selection & routing
│   │   ├── retriever.py      # Qdrant hybrid search with episode recording
│   │   ├── executor.py       # Tavily web search with source filtering
│   │   ├── critic.py         # Evidence analysis, verdict & insight sharing
│   │   └── archivist.py      # Memory management & temporal versioning
│   ├── core/
│   │   ├── config.py         # Configuration management
│   │   ├── state.py          # LangGraph state definitions
│   │   └── graph.py          # LangGraph orchestration
│   ├── database/
│   │   ├── qdrant_client.py  # Qdrant operations for knowledge base
│   │   ├── memory_manager.py # Episodic & shared context memory
│   │   ├── embeddings.py     # Dense + Sparse embeddings
│   │   └── schema.py         # All collection schemas
│   ├── tools/
│   │   ├── tavily_search.py  # Tavily integration
│   │   └── source_filter.py  # Trusted source filtering
│   ├── utils/
│   │   ├── logger.py         # Logging utilities
│   │   └── helpers.py        # Helper functions
│   └── main.py               # CLI & API entry point
├── data/
│   └── trusted_sources.json  # Trusted source configuration
├── tests/                    # Test suite
├── docker-compose.yml        # Docker services
├── requirements.txt          # Python dependencies
└── pyproject.toml           # Project configuration
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTH_GUARDIAN_GEMINI_API_KEY` | Google Gemini API key | Required |
| `TRUTH_GUARDIAN_TAVILY_API_KEY` | Tavily AI API key | Required |
| `TRUTH_GUARDIAN_QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `TRUTH_GUARDIAN_QDRANT_API_KEY` | Qdrant Cloud API key | Optional |
| `TRUTH_GUARDIAN_HYBRID_SEARCH_ALPHA` | Dense vs Sparse weight | `0.7` |
| `TRUTH_GUARDIAN_MAX_FEEDBACK_LOOPS` | Max feedback iterations | `3` |

### Trusted Sources

Edit `data/trusted_sources.json` to customize source tiers:

- **Tier 1**: Government (.gov), Academic (.edu), Scientific journals
- **Tier 2**: Major news (Reuters, BBC, NYT), Fact-checkers (Snopes, PolitiFact)
- **Tier 3**: Reputable tech/science publications
- **Blocked**: Known misinformation and satire sites

---

## 🗄️ Database Design

Digital Truth Guardian uses **three separate Qdrant collections** for different memory types:

### Collection 1: `knowledge_base` (Knowledge Memory)

Stores verified facts and claims with hybrid search capability.

| Field | Type | Description |
|-------|------|-------------|
| `vector_dense` | 768d float | Google text-embedding-004 |
| `vector_sparse` | sparse | FastEmbed BM25 |
| `text` | string | The claim content |
| `verdict` | keyword | TRUE, FALSE, UNCERTAIN |
| `fact_type` | keyword | STATIC, TRANSIENT |
| `source_domain` | keyword | Source domain |
| `valid_from` | timestamp | Temporal versioning |
| `valid_to` | timestamp | NULL = currently valid |

### Collection 2: `episodic_memory` (Episodic Memory)

Stores past agent decisions and outcomes for learning from experience.

| Field | Type | Description |
|-------|------|-------------|
| `vector_dense` | 768d float | Semantic embedding of episode |
| `session_id` | keyword | Session identifier |
| `agent_name` | keyword | Which agent made the decision |
| `action_type` | keyword | retrieval, search, critique, route |
| `outcome` | keyword | success, failure, uncertain, cache_hit |
| `decision_reasoning` | text | Why the decision was made |
| `tools_used` | array | Tools used in this action |
| `confidence` | float | Confidence score |

### Collection 3: `shared_context` (Shared Team Memory)

Enables inter-agent communication and coordination.

| Field | Type | Description |
|-------|------|-------------|
| `vector_dense` | 768d float | Semantic embedding of context |
| `context_type` | keyword | task_context, insight, warning, strategy |
| `agent_source` | keyword | Which agent wrote this |
| `content` | text | The shared information |
| `target_agents` | array | Which agents should read (empty = all) |
| `priority` | integer | 1-5, higher = more important |
| `expires_at` | timestamp | TTL for temporary context |

### Optimization Features

- **Binary Quantization**: 30x compression for fast retrieval
- **Hybrid Search**: Combines semantic + keyword matching (knowledge_base)
- **Temporal Versioning**: Handles changing facts correctly
- **Metadata Filtering**: Filter by agent, outcome, priority, session

---

## 🔒 Safety Features

### The "No-Hallucination" Guarantee

1. **Filter 1 (Pre-Search)**: Only accept results from trusted sources
2. **Filter 2 (Evidence Check)**: Critic evaluates sufficiency
3. **Filter 3 (Post-Process)**: Only memorize from high-tier sources

### Handling Uncertainty

When evidence is insufficient:
```
⚠️ Verdict: UNCERTAIN

I could not conclusively verify this claim. My search did not 
return sufficient evidence from trusted sources. This may be 
an unverified rumor requiring further investigation.
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_source_filter.py
```

---

## 🐳 Docker Deployment

### Full Stack
```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Individual Services
```bash
# Just Qdrant
docker-compose up -d qdrant

# Build and run API
docker build -t truth-guardian .
docker run -p 8000:8000 --env-file .env truth-guardian
```

---

## 📊 API Reference

### Endpoints

#### `POST /verify`
Verify a claim and return verdict.

**Request:**
```json
{
  "claim": "The Earth is 4.5 billion years old",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "query": "The Earth is 4.5 billion years old",
  "verdict": "TRUE",
  "confidence": 0.95,
  "explanation": "Multiple authoritative sources confirm...",
  "response": "## ✅ Verdict: VERIFIED AS TRUE\n...",
  "sources": ["nasa.gov", "nature.com"],
  "memory_written": true,
  "processing_time": 2.34
}
```

#### `GET /health`
Health check endpoint.

#### `GET /stats`
Knowledge base statistics.

---

## 🏆 Why This Wins

| Criteria | Implementation |
|----------|----------------|
| **Correct Qdrant Usage** | Hybrid Search (Dense + Sparse), Binary Quantization, Metadata Filtering |
| **Multiple Collections** | 3 collections: knowledge_base, episodic_memory, shared_context |
| **Episodic Memory** | Agents record decisions and recall similar past experiences |
| **Shared Team Memory** | Agents read/write to shared vector store for coordination |
| **Autonomous Tool Selection** | Planner dynamically chooses QDRANT, TAVILY, or BOTH based on query |
| **Agentic Architecture** | Planner-Router with dynamic tool selection and feedback loops |
| **Memory Evolution** | Temporal versioning for TRANSIENT vs STATIC facts |
| **Societal Relevance** | Combats misinformation with "Safe Fail" protocol |
| **Tech Stack** | State-of-the-art Gemini + LangGraph orchestration |

### Hackathon Criteria Checklist

| Requirement | Status |
|-------------|--------|
| Clear Agent Roles | ✅ 5 distinct agents with well-defined responsibilities |
| Structured Communication | ✅ LangGraph TypedDict state with structured outputs |
| Coordinated Decisions | ✅ Planner routes based on retrieval scores + feedback |
| Vector Search as Memory | ✅ Three collections for different memory types |
| Retrieval as Active Decision | ✅ Planner decides WHEN to retrieve |
| Tool-Aware Agents | ✅ Dynamic tool selection based on query analysis |
| Memory Improves Decisions | ✅ Episodic memory recalls successful past strategies |
| Metadata-Aware Filtering | ✅ Filter by agent, outcome, priority, session_id |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Qdrant](https://qdrant.tech/) for the amazing vector database
- [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [Google Gemini](https://ai.google.dev/) for powerful LLM capabilities
- [Tavily](https://tavily.com/) for clean web search API

---

<p align="center">
  Built with ❤️ for <b>Qdrant Convolve 4.0 Hackathon</b>
</p>
