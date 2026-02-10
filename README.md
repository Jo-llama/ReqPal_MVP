# 🎯 ReqPal RAG MVP

> A lightweight, standalone MVP for testing Retrieval-Augmented Generation (RAG) features in regulatory compliance workflows.

## ✨ What This Does

ReqPal RAG MVP enables intelligent document processing and semantic search for compliance projects:

- **Smart Project Creation** – Capture domain, industry, geography, regulatory exposure, and constraints
- **Multi-Format Document Upload** – Process PDF, DOCX, CSV, TXT, and JSON files automatically
- **Intelligent Search** – Ask questions in natural language and get contextually relevant answers from your documents
- **100% Local Embeddings** – No API costs for document vectorization using Sentence Transformers

## 🚀 Quick Start

### Prerequisites

- Python 3.11 
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jo-llama/ReqPal_MVP.git
   cd ReqPal_MVP
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

### Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

> **Note:** Port 8001 is used to avoid conflicts with the main ReqPal instance (typically on port 8000)

Open your browser and navigate to:
```
http://localhost:8001/static/index.html
```

## 📁 Project Structure

```
ReqPal-RAG-MVP/
├── main.py                    # FastAPI application & endpoints
├── models.py                  # Pydantic data models
├── storage.py                 # JSON-based data persistence
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
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
│   └── index.html            # Web UI
│
├── uploads/                   # Document storage (auto-created)
└── storage/                   # Vector database (auto-created)
```

## 🎯 Key Features

### 1. Enhanced Project Creation

Create compliance projects with rich metadata:

- **Domain** – Healthcare, Finance, E-commerce, etc.
- **Industry** – Banking, Insurance, SaaS, etc.
- **Geography** – EU, US, APAC, Global
- **Regulatory Exposure** – GDPR, HIPAA, SOC2, PCI-DSS, etc.
- **Success Criteria** – Define measurable outcomes
- **Constraints** – Business, legal, and technical limitations

### 2. Intelligent Document Processing

Upload and automatically process multiple file formats:

| Format | Processing |
|--------|-----------|
| **PDF** | Text extraction via PyPDF2 |
| **DOCX** | Paragraph extraction via python-docx |
| **CSV** | Row-by-row processing |
| **TXT** | Plain text chunking |
| **JSON** | Structured data parsing |

Documents are automatically:
- Chunked into semantic segments
- Embedded using local ML models (all-MiniLM-L6-v2)
- Stored in ChromaDB vector database
- Made searchable via similarity search

### 3. Semantic RAG Search

Ask questions in natural language:

```
"What are the data retention requirements for GDPR?"
"What incident response procedures are documented?"
"What stakeholders need to be involved in onboarding?"
```

Features:
- **Contextual Answers** – Powered by LLM synthesis
- **Source Attribution** – See which documents answers come from
- **Similarity Scores** – Understand relevance ranking
- **Document Filtering** – Search specific file types

## 🧪 Testing the MVP

### Example Workflow

1. **Create a Project**
   - Name: "GDPR Compliance Initiative"
   - Domain: Healthcare
   - Regulatory Exposure: GDPR, HIPAA
   - Geography: EU

2. **Upload Documents**
   - GDPR regulation PDF
   - Internal data retention policy
   - Incident response playbook

3. **Ask Questions**
   - "What are data subject rights under GDPR?"
   - "How long can we retain customer data?"
   - "What are the breach notification timelines?"

4. **Review Results**
   - See contextual answers
   - Check source documents
   - Verify similarity scores

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### LLM Providers

The system supports multiple LLM providers (configured in `llm_router.py`):
- **Groq** – Fast inference for Llama models
- **Anthropic** – Claude models
- **OpenAI** – GPT models

## 🐛 Troubleshooting

### Model Download Issues

If the embedding model fails to download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Port Already in Use

Change the port in the uvicorn command:

```bash
uvicorn main:app --reload --port 8002
```

### Missing Dependencies

Reinstall requirements:

```bash
pip install --upgrade -r requirements.txt
```

### ChromaDB Errors

Delete the vector database and restart:

```bash
rm -rf storage/chroma/
```

## 🔄 Integration with Main ReqPal

Once tested and validated:

1. **Extract Successful Features** – Identify what works well
2. **Merge Backend Services** – Integrate RAG endpoints into main API
3. **Update Frontend** – Add RAG search UI components
4. **Data Migration** – Plan data model alignment
5. **Archive MVP** – Preserve as reference implementation

## 📝 Roadmap

- [ ] Add stakeholder management with RAG search
- [ ] Implement risk identification from documents
- [ ] Build automated gap analysis
- [ ] Add requirement traceability across documents
- [ ] Multi-tenancy support
- [ ] Advanced filtering and faceted search
- [ ] Document version control
- [ ] Audit trail for compliance queries

## 🤝 Contributing

This is an MVP for internal testing. For contributions to the main ReqPal project, please contact the development team.

## 📄 License

Proprietary - Internal Use Only

---

**Built with:**
- [FastAPI](https://fastapi.tiangolo.com/) – Modern Python web framework
- [ChromaDB](https://www.trychroma.com/) – Vector database
- [Sentence Transformers](https://www.sbert.net/) – Embedding models
- [LangChain](https://www.langchain.com/) – LLM orchestration
