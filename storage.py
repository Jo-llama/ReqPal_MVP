# storage.py - UPDATED v3.3.1 with RAG Support (fixed document ID flow)

import json
import os
from typing import List, Dict, Any
from datetime import datetime

from models import (
    Project, Requirement, UserStory, ProcessMap,
    KnowledgeGraph, ValidationToken,
    Document, Stakeholder, Risk, Assumption,
    GapAnalysis, TraceabilityLink
)

class Storage:
    """In-memory storage with JSON file persistence + RAG support"""

    def __init__(self, filepath: str = "reqpal_data.json"):
        self.filepath = filepath

        self.projects: List[Project] = []
        self.requirements: List[Requirement] = []
        self.user_stories: List[UserStory] = []
        self.process_maps: List[ProcessMap] = []
        self.knowledge_graphs: List[KnowledgeGraph] = []
        self.validation_tokens: List[ValidationToken] = []

        self.documents: List[Document] = []
        self.stakeholders: List[Stakeholder] = []
        self.risks: List[Risk] = []
        self.assumptions: List[Assumption] = []
        self.gap_analyses: List[GapAnalysis] = []
        self.traceability_links: List[TraceabilityLink] = []

        self.counters = {
            "project": 0,
            "requirement": 0,
            "story": 0,
            "process": 0,
            "graph": 0,
            "document": 0,
            "stakeholder": 0,
            "risk": 0,
            "assumption": 0,
            "gap_analysis": 0,
            "traceability": 0
        }

        self.load_data()

    def get_next_id(self, entity_type: str) -> int:
        self.counters[entity_type] += 1
        return self.counters[entity_type]

    # -------------------- NEW: document helpers --------------------

    def add_document(self, doc: Document) -> Document:
        """Assign ID immediately and persist."""
        doc.id = self.get_next_id("document")
        self.documents.append(doc)
        self.save_data()
        return doc

    def update_document(self, doc: Document) -> None:
        """Replace stored doc with same id and persist."""
        for i, d in enumerate(self.documents):
            if d.id == doc.id:
                self.documents[i] = doc
                break
        self.save_data()

    def delete_document(self, document_id: int) -> None:
        """Remove document from storage and persist."""
        doc = next((d for d in self.documents if d.id == document_id), None)
        if doc:
            self.documents.remove(doc)
            self.save_data()

    # -------------------- persistence --------------------

    def save_data(self):
        data = {
            "projects": [p.dict() for p in self.projects],
            "requirements": [r.dict() for r in self.requirements],
            "user_stories": [s.dict() for s in self.user_stories],
            "process_maps": [pm.dict() for pm in self.process_maps],
            "knowledge_graphs": [kg.dict() for kg in self.knowledge_graphs],
            "validation_tokens": [vt.dict() for vt in self.validation_tokens],

            "documents": [d.dict() for d in self.documents],
            "stakeholders": [s.dict() for s in self.stakeholders],
            "risks": [r.dict() for r in self.risks],
            "assumptions": [a.dict() for a in self.assumptions],
            "gap_analyses": [ga.dict() for ga in self.gap_analyses],
            "traceability_links": [tl.dict() for tl in self.traceability_links],

            "counters": self.counters,
            "last_updated": datetime.now().isoformat(),
            "version": "3.3.1"
        }

        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Data saved to {self.filepath}")
        except Exception as e:
            print(f"⚠️  Failed to save data: {e}")

    def load_data(self):
        if not os.path.exists(self.filepath):
            print("📝 No existing data file found. Starting fresh.")
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.projects = [Project(**p) for p in data.get("projects", [])]
            self.requirements = [Requirement(**r) for r in data.get("requirements", [])]
            self.user_stories = [UserStory(**s) for s in data.get("user_stories", [])]
            self.process_maps = [ProcessMap(**pm) for pm in data.get("process_maps", [])]
            self.knowledge_graphs = [KnowledgeGraph(**kg) for kg in data.get("knowledge_graphs", [])]
            self.validation_tokens = [ValidationToken(**vt) for vt in data.get("validation_tokens", [])]

            self.documents = [Document(**d) for d in data.get("documents", [])]
            self.stakeholders = [Stakeholder(**s) for s in data.get("stakeholders", [])]
            self.risks = [Risk(**r) for r in data.get("risks", [])]
            self.assumptions = [Assumption(**a) for a in data.get("assumptions", [])]
            self.gap_analyses = [GapAnalysis(**ga) for ga in data.get("gap_analyses", [])]
            self.traceability_links = [TraceabilityLink(**tl) for tl in data.get("traceability_links", [])]

            self.counters = data.get("counters", self.counters)
            self._update_counters()

            print(f"✅ Loaded data from {self.filepath} (v{data.get('version', '3.3')})")
            print(f"   - Projects: {len(self.projects)}")
            print(f"   - Requirements: {len(self.requirements)}")
            print(f"   - User Stories: {len(self.user_stories)}")
            print(f"   - Documents: {len(self.documents)}")

        except Exception as e:
            print(f"⚠️  Failed to load data: {e}")
            print("   Starting with empty storage.")

    def _update_counters(self):
        def max_id(items):
            ids = [x.id for x in items if getattr(x, "id", None)]
            return max(ids) if ids else 0

        self.counters["project"] = max(self.counters["project"], max_id(self.projects))
        self.counters["requirement"] = max(self.counters["requirement"], max_id(self.requirements))
        self.counters["story"] = max(self.counters["story"], max_id(self.user_stories))
        self.counters["document"] = max(self.counters["document"], max_id(self.documents))
        self.counters["stakeholder"] = max(self.counters["stakeholder"], max_id(self.stakeholders))
        self.counters["risk"] = max(self.counters["risk"], max_id(self.risks))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "projects": len(self.projects),
            "requirements": len(self.requirements),
            "user_stories": len(self.user_stories),
            "process_maps": len(self.process_maps),
            "knowledge_graphs": len(self.knowledge_graphs),
            "validation_tokens": len(self.validation_tokens),
            "documents": len(self.documents),
            "stakeholders": len(self.stakeholders),
            "risks": len(self.risks),
            "assumptions": len(self.assumptions),
            "gap_analyses": len(self.gap_analyses),
            "traceability_links": len(self.traceability_links)
        }

# Singleton
storage = Storage()
__all__ = ["storage", "Storage"]
