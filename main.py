# main.py — ReqPal RAG MVP (clean + stable)
# - Projects CRUD (+ delete cascade)
# - Documents upload/list/get/delete (Chroma cleanup)
# - /rag/search (chunks)
# - /rag/answer (rewrite -> retrieve -> filter -> optional LLM answer; ALWAYS returns context_chunks)
# - /providers-status (router debug)

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel
from typing import List as TList

from models import (
    Project,
    Document,
    DocumentClassification,
    Requirement,
    UserStory,
    RAGQuery,
)
from storage import storage
from backend.services.rag_service import rag_service
from backend.services.llm_router import LLMRouter


app = FastAPI(title="ReqPal RAG MVP", version="3.4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------------------
# LLM Router
# ----------------------------

router: Optional[LLMRouter] = None

def get_router() -> Optional[LLMRouter]:
    global router
    if router is not None:
        return router
    try:
        router = LLMRouter()
        return router
    except Exception:
        return None


# ----------------------------
# Prompts
# ----------------------------

QUERY_REWRITE_SYSTEM = """You rewrite a user question for semantic search in a regulated banking context.
Return JSON:
{ "rewritten_query": "..." }
Rules:
- Keep it short and specific (<= 1 sentence)
- Add key compliance terms if relevant (SOC2, GDPR, SOX, PCI, audit trail, retention, access control, incident response)
- Do NOT invent details.
"""

ANSWER_SYSTEM = """You answer using only the provided context_chunks.
Return JSON:
{
  "answer": ["..."],
  "acceptance_criteria": ["..."],
  "edge_cases": ["..."],
  "open_questions": ["..."],
  "citations_used": ["chunk_id", "..."]
}
Rules:
- If context is insufficient, say what is missing and ask targeted open_questions.
- Cite chunk_ids you used.
- No hallucinations.
"""


# ----------------------------
# Helpers
# ----------------------------

def _safe_bool_env(key: str) -> bool:
    return bool((os.getenv(key) or "").strip())

def _project_or_404(project_id: int) -> Project:
    p = next((x for x in storage.projects if x.id == project_id), None)
    if not p:
        raise HTTPException(404, "Project not found")
    return p

def _doc_or_404(document_id: int) -> Document:
    d = next((x for x in storage.documents if x.id == document_id), None)
    if not d:
        raise HTTPException(404, "Document not found")
    return d

def _filter_by_similarity(cands: List[Any], min_similarity: float) -> List[Any]:
    return [c for c in cands if float(getattr(c, "similarity_score", 0.0)) >= float(min_similarity)]

def _top_n(cands: List[Any], n: int) -> List[Any]:
    return list(cands[: max(0, n)])


# ----------------------------
# Routes: UI + Health
# ----------------------------

@app.get("/index.html")
async def index():
    path = os.path.join("static", "index.html")
    if not os.path.exists(path):
        raise HTTPException(404, "static/index.html not found")
    return FileResponse(path)

@app.get("/")
async def root():
    r = get_router()
    return {
        "status": "ReqPal RAG MVP is running",
        "version": app.version,
        "rag_enabled": True,
        "env": {
            "GROQ_API_KEY": _safe_bool_env("GROQ_API_KEY"),
            "OPENAI_API_KEY": _safe_bool_env("OPENAI_API_KEY"),
            "OLLAMA_BASE_URL": (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip(),
            "OLLAMA_MODEL": (os.getenv("OLLAMA_MODEL") or "").strip(),
        },
        "router": (r.providers_status() if r else {"configured": [], "models": {}}),
    }

@app.get("/providers-status")
async def providers_status():
    r = get_router()
    if not r:
        return {
            "ok": False,
            "error": "No LLM providers configured or router init failed.",
            "env": {
                "GROQ_API_KEY": _safe_bool_env("GROQ_API_KEY"),
                "OPENAI_API_KEY": _safe_bool_env("OPENAI_API_KEY"),
                "OLLAMA_BASE_URL": (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip(),
                "OLLAMA_MODEL": (os.getenv("OLLAMA_MODEL") or "").strip(),
            },
        }
    return {"ok": True, "router": r.providers_status()}


# ----------------------------
# Projects CRUD
# ----------------------------

@app.post("/projects", response_model=Project)
async def create_project(project: Project):
    project.id = storage.get_next_id("project")
    storage.projects.append(project)
    storage.save_data()
    return project

@app.get("/projects", response_model=List[Project])
async def list_projects():
    return storage.projects

@app.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: int):
    return _project_or_404(project_id)

@app.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    p = _project_or_404(project_id)

    docs = [d for d in storage.documents if d.project_id == project_id]
    for d in docs:
        try:
            rag_service.delete_document_chunks(d.id or 0)
        except Exception:
            pass

        try:
            if d.file_path and os.path.exists(d.file_path):
                os.remove(d.file_path)
        except Exception:
            pass

        storage.documents.remove(d)

    storage.requirements = [r for r in storage.requirements if r.project_id != project_id]
    storage.user_stories = [s for s in storage.user_stories if s.project_id != project_id]

    storage.projects.remove(p)
    storage.save_data()
    return {"ok": True, "deleted_project_id": project_id, "deleted_documents": len(docs)}


# ----------------------------
# Documents
# ----------------------------

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    project_id: int = Form(...),
    classification: str = Form(...),
    document_purpose: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    # Validate project exists
    project = next((p for p in storage.projects if p.id == project_id), None)
    if not project:
        raise HTTPException(404, f"Project {project_id} not found")

    # Validate classification enum
    try:
        cls = DocumentClassification(classification)
    except Exception:
        raise HTTPException(400, f"Invalid classification: {classification}")

    # Read file
    content = await file.read()

    # Parse tags
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]

    # ✅ CRITICAL: allocate document ID BEFORE processing/indexing
    doc_id = storage.get_next_id("document")

    try:
        document = await rag_service.process_document(
            file_content=content,
            filename=file.filename,
            project_id=project_id,
            classification=cls,
            document_id=doc_id,  # ✅ NEW
            metadata={
                "document_purpose": document_purpose,
                "tags": tag_list,
            },
        )

        # save
        storage.documents.append(document)
        storage.save_data()

        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded and processed successfully",
            "document": document,
            "chunks_created": len(document.chunks or []),
            "indexed": document.indexed,
        }

    except Exception as e:
        # log full error in console
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Failed to process document: {str(e)}")


@app.get("/documents")
async def list_documents(project_id: Optional[int] = None):
    docs = storage.documents
    if project_id is not None:
        docs = [d for d in docs if d.project_id == project_id]
    return docs

@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    return _doc_or_404(document_id)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    d = _doc_or_404(document_id)

    try:
        rag_service.delete_document_chunks(d.id or 0)
    except Exception:
        pass

    try:
        if d.file_path and os.path.exists(d.file_path):
            os.remove(d.file_path)
    except Exception:
        pass

    storage.documents.remove(d)
    storage.save_data()
    return {"ok": True, "deleted_document_id": document_id}


# ----------------------------
# RAG: search (chunks)
# ----------------------------

@app.post("/rag/search")
async def rag_search(query: RAGQuery):
    _project_or_404(query.project_id)
    cands = await rag_service.semantic_search(query)
    return {
        "query": query.query,
        "project_id": query.project_id,
        "count": len(cands),
        "results": cands,
    }


# ----------------------------
# RAG: answer
# ----------------------------

class RAGAnswerRequest(BaseModel):
    query: str
    project_id: int
    top_k: int = 20
    min_similarity: float = 0.25
    document_classifications: TList[DocumentClassification] = []


@app.post("/rag/answer")
async def rag_answer(req: RAGAnswerRequest):
    project = _project_or_404(req.project_id)

    project_ctx = {
        "name": project.name,
        "description": project.description,
        "domain": getattr(project, "domain", None),
        "industry": getattr(project, "industry", None),
        "geography": getattr(project, "geography", []),
        "regulatory_exposure": getattr(project, "regulatory_exposure", []),
        "constraints": getattr(project, "constraints", None),
        "success_criteria": getattr(project, "success_criteria", []),
    }

    llm_trace_parts: List[str] = []
    rewritten_query = req.query
    r = get_router()

    # 1) Rewrite (best-effort)
    if r:
        try:
            rewrite_json, provider, _trace = await r.chat_json(
                system=QUERY_REWRITE_SYSTEM,
                user_payload={"project_context": project_ctx, "question": req.query},
                temperature=0.0,
                max_tokens=220,
                timeout_s=45.0,
                retries=1,
            )
            rewritten_query = (rewrite_json.get("rewritten_query") or req.query).strip()
            llm_trace_parts.append(f"rewrite:ok({provider})")
        except Exception as e:
            msg = str(e)
            if len(msg) > 120: msg = msg[:120] + "…"
            llm_trace_parts.append(f"rewrite:fail({type(e).__name__}:{msg})")

    else:
        llm_trace_parts.append("rewrite:skip(no_router)")

    # 2) Retrieve
    rag_query = RAGQuery(
        query=rewritten_query,
        project_id=req.project_id,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
        document_classifications=req.document_classifications or [],
    )

    candidates = await rag_service.semantic_search(rag_query)

    if not candidates and rewritten_query != req.query:
        rag_query.query = req.query
        candidates = await rag_service.semantic_search(rag_query)

    if not candidates:
        return {
            "answer": ["No relevant context found in indexed documents."],
            "acceptance_criteria": [],
            "edge_cases": [],
            "open_questions": [
                "Which exact document contains the rule you are asking about?",
                "Upload the relevant policy/standard/audit evidence section (or narrow the question).",
            ],
            "citations_used": [],
            "context_chunks": [],
            "rewritten_query": rewritten_query,
            "llm_trace": " ".join(llm_trace_parts),
        }

    # 3) Filter by similarity
    filtered = _filter_by_similarity(candidates, req.min_similarity)
    llm_trace_parts.append(f"filter:{len(filtered)}/{len(candidates)}>= {req.min_similarity}")

    if not filtered:
        filtered = _top_n(candidates, 8)
        llm_trace_parts.append("filter:fallback_top")

    selected = _top_n(filtered, 8)
    top_ids = [c.chunk_id for c in selected]

    # 4) Answer (best-effort)
    answer_payload = {
        "answer": ["(LLM answer unavailable — returning retrieved context only.)"],
        "acceptance_criteria": [],
        "edge_cases": [],
        "open_questions": [],
        "citations_used": top_ids,
    }

    if r:
        try:
            context_chunks = [
                {
                    "chunk_id": c.chunk_id,
                    "document_name": c.document_name,
                    "classification": c.classification,
                    "content": (c.content or "")[:2200],
                }
                for c in selected
            ]

            ans_json, provider, _trace = await r.chat_json(
                system=ANSWER_SYSTEM,
                user_payload={
                    "question": req.query,
                    "project_context": project_ctx,
                    "context_chunks": context_chunks,
                },
                temperature=0.1,
                max_tokens=520,
                timeout_s=120.0,
                retries=1,
                backoff_s=1.0,
            )

            if isinstance(ans_json, dict):
                answer_payload = {
                    "answer": ans_json.get("answer", answer_payload["answer"]),
                    "acceptance_criteria": ans_json.get("acceptance_criteria", []),
                    "edge_cases": ans_json.get("edge_cases", []),
                    "open_questions": ans_json.get("open_questions", []),
                    "citations_used": ans_json.get("citations_used", top_ids) or top_ids,
                }

            llm_trace_parts.append(f"answer:ok({provider})")
        except Exception as e:
            msg = str(e)
            if len(msg) > 120: msg = msg[:120] + "…"
            llm_trace_parts.append(f"answer:fail({type(e).__name__}:{msg})")

    else:
        llm_trace_parts.append("answer:skip(no_router)")

    return {
        "answer": answer_payload.get("answer", []),
        "acceptance_criteria": answer_payload.get("acceptance_criteria", []),
        "edge_cases": answer_payload.get("edge_cases", []),
        "open_questions": answer_payload.get("open_questions", []),
        "citations_used": answer_payload.get("citations_used", top_ids),

        "context_chunks": [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "document_name": c.document_name,
                "classification": c.classification,
                "content": c.content,
                "similarity_score": float(getattr(c, "similarity_score", 0.0)),
                "page_number": getattr(c, "page_number", None),
                "section": getattr(c, "section", None),
            }
            for c in selected
        ],
        "rewritten_query": rewritten_query,
        "llm_trace": " ".join(llm_trace_parts),
    }


# ----------------------------
# Optional: requirements/stories
# ----------------------------

@app.get("/requirements", response_model=List[Requirement])
async def list_requirements(project_id: Optional[int] = None):
    if project_id is not None:
        return [r for r in storage.requirements if r.project_id == project_id]
    return storage.requirements

@app.post("/requirements", response_model=Requirement)
async def create_requirement(req: Requirement):
    req.id = storage.get_next_id("requirement")
    storage.requirements.append(req)
    storage.save_data()
    return req

@app.get("/user-stories", response_model=List[UserStory])
async def list_user_stories(project_id: Optional[int] = None):
    if project_id is not None:
        return [s for s in storage.user_stories if s.project_id == project_id]
    return storage.user_stories

@app.post("/user-stories", response_model=UserStory)
async def create_user_story(story: UserStory):
    story.id = storage.get_next_id("story")
    storage.user_stories.append(story)
    storage.save_data()
    return story


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
