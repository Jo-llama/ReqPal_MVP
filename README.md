# ReqPal RAG MVP

A **lightweight MVP** to test RAG (Retrieval-Augmented Generation) features.

## 🎯 What This Is

- Enhanced project creation with rich metadata
- Document upload (PDF, DOCX, CSV, TXT, JSON)
- Automatic text extraction, chunking, and embedding
- Semantic search across documents
- 100% local embeddings (no API costs!)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

**Note:** Port 8001 so it doesn't conflict with your main ReqPal on 8000!

### 3. Open Browser

```
http://localhost:8001/static/index.html
```

## 📁 Project Structure

```
ReqPal-RAG-MVP/
├── main.py                      # FastAPI app with RAG endpoints
├── models.py                    # Data models (Project, Document, etc.)
├── requirements.txt             # Dependencies
├── reqpal_data.json            # Data storage (auto-created)
├── __init__.py                  # Package initialization
├── .env                         # Environment variables (not in repo)
├── .gitignore                   # Git ignore rules
│
├── backend/
│   ├── __init__.py
│   └── services/
│       ├── __init__.py
│       ├── rag_service.py           # RAG implementation
│       ├── llm_router.py            # LLM provider routing
│       ├── groq_http.py             # Groq API integration
│       ├── reranker_service.py      # Result reranking
│       └── rag_llm_prompts.py       # Prompt templates
│
├── static/
│   └── index.html               # Web UI
│
├── uploads/                     # Uploaded documents (auto-created)
│   ├── *.pdf                    # PDF documents
│   ├── *.docx                   # Word documents
│   ├── *.csv                    # CSV files
│   ├── *.json                   # JSON files
│   └── *.bpmn                   # BPMN diagrams
│
└── storage/
    └── chroma/                  # ChromaDB vector database (auto-created)
```

## 🎯 Features

### 1. Enhanced Project Creation

Create projects with:
- Domain, industry, geography
- Regulatory exposure (GDPR, HIPAA, etc.)
- Success criteria
- Constraints (business, legal, technical)

### 2. Document Upload

Upload and automatically process:
- PDFs (text extraction)
- DOCX (paragraph extraction)
- CSV (structured data)
- TXT (plain text)
- JSON (structured data)

### 3. RAG Search

Semantic search across all documents:
- Natural language queries
- Filter by document type
- Similarity scoring
- Source attribution

**Fallback Options:**
- **Ollama**: Local LLM inference if API providers fail
- **Chunks-only mode**: Returns raw document chunks if LLM is unavailable

## 📊 Test It

1. **Create a project** with regulatory exposure
2. **Upload a PDF** (e.g., GDPR regulation)
3. **Search**: "What are data retention requirements?"
4. **See results** with similarity scores and source documents

## 🔄 Merge Back to Main ReqPal

Once tested and working:
1. Copy successful features
2. Integrate endpoints
3. Update main UI
4. Deprecate this MVP

## 🐛 Troubleshooting

**Model download fails:**

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Port already in use:**

Change port in command: `uvicorn main:app --port 8002`

**LLM API not working:**

The system will automatically fall back to:
1. **Ollama** (if installed locally): `ollama pull llama3.2`
2. **Chunks-only mode**: Returns relevant document snippets without LLM synthesis

