"""Vector Store implementation with explicit 'chroma' and 'memory' backends.

Never silently switches backends. Validates collection metadata.
"""

import os
import json
import math
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from app.config import settings
from app.schemas.job import JobPosting


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))


class InMemoryVectorStore:
    """Exact cosine persistent JSON vector store with metadata filtering."""

    def __init__(
        self,
        storage_dir: str,
        collection_name: str,
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.collection_name = collection_name
        self.embedding_backend = embedding_backend or settings.embedding_backend
        self.embedding_model = embedding_model or settings.embedding_model
        self.embedding_dim = embedding_dim or settings.embedding_dim
        self.db_file = self.storage_dir / f"{collection_name}_store.json"
        self._records: Dict[str, Dict[str, Any]] = {}
        self._meta: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.db_file.exists():
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._records = data.get("records", {})
                    self._meta = data.get("metadata", {})
                    # Validate existing collection metadata
                    if self._meta:
                        stored_dim = self._meta.get("dimension")
                        stored_backend = self._meta.get("embedding_backend")
                        stored_model = self._meta.get("embedding_model")
                        if stored_dim is not None and stored_dim != self.embedding_dim:
                            raise ValueError(
                                f"Incompatible vector collection detected: collection '{self.collection_name}' was indexed "
                                f"with dimension={stored_dim}, but current configuration expects dimension={self.embedding_dim}. "
                                "Please rebuild/reindex the collection using /jobs/ingest?overwrite=true or reset the collection."
                            )
                        if stored_backend is not None and stored_backend != self.embedding_backend:
                            raise ValueError(
                                f"Incompatible vector collection detected: collection '{self.collection_name}' was indexed "
                                f"with backend='{stored_backend}', but current configuration expects backend='{self.embedding_backend}'. "
                                "Please rebuild/reindex the collection using /jobs/ingest?overwrite=true or reset the collection."
                            )
                        if stored_model is not None and stored_model != self.embedding_model:
                            raise ValueError(
                                f"Incompatible vector collection detected: collection '{self.collection_name}' was indexed "
                                f"with model='{stored_model}', but current configuration expects model='{self.embedding_model}'. "
                                "Please rebuild/reindex the collection using /jobs/ingest?overwrite=true or reset the collection."
                            )
            except ValueError:
                raise
            except Exception:
                self._records = {}
                self._meta = {}
        if not self._meta:
            self._meta = {
                "embedding_backend": self.embedding_backend,
                "embedding_model": self.embedding_model,
                "dimension": self.embedding_dim,
            }
            self._save()

    def _save(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump({"records": self._records, "metadata": self._meta}, f)

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        overwrite: bool = False,
    ) -> Tuple[int, int]:
        indexed, skipped = 0, 0
        for j_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            if j_id in self._records and not overwrite:
                skipped += 1
                continue
            self._records[j_id] = {
                "id": j_id,
                "document": doc,
                "embedding": emb,
                "metadata": meta,
            }
            indexed += 1
        self._save()
        return indexed, skipped

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        scored = []
        for j_id, item in self._records.items():
            meta = item["metadata"]
            if filters:
                match = True
                for k, v in filters.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            emb = item["embedding"]
            sim = _cosine_similarity(query_embedding, emb)
            scored.append({
                "job_id": j_id,
                "metadata": meta,
                "document": item["document"],
                "similarity": round(sim, 4),
            })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def get_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._records.get(job_id)

    def delete_by_id(self, job_id: str) -> bool:
        if job_id in self._records:
            del self._records[job_id]
            self._save()
            return True
        return False

    def count(self) -> int:
        return len(self._records)

    def reset(self):
        self._records = {}
        self._meta = {}
        if self.db_file.exists():
            try:
                self.db_file.unlink()
            except Exception:
                pass


class VectorStoreService:
    def __init__(
        self,
        backend: Optional[str] = None,
        chroma_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        self.backend = (backend or settings.vector_store_backend).lower()
        self.chroma_dir = str(chroma_dir or settings.chroma_dir)
        self.collection_name = collection_name or settings.collection_name
        self.embedding_backend = embedding_backend or settings.embedding_backend
        self.embedding_model = embedding_model or settings.embedding_model
        self.embedding_dim = embedding_dim or settings.embedding_dim
        self._instance = None
        self._init_backend()

    def _init_backend(self):
        if self.backend == "chroma":
            try:
                import chromadb
                os.makedirs(self.chroma_dir, exist_ok=True)
                client = chromadb.PersistentClient(path=self.chroma_dir)
                target_metadata = {
                    "hnsw:space": "cosine",
                    "embedding_backend": self.embedding_backend,
                    "embedding_model": self.embedding_model,
                    "dimension": self.embedding_dim,
                }
                
                # Check if collection already exists
                existing_colls = [c.name for c in client.list_collections()]
                if self.collection_name in existing_colls:
                    coll = client.get_collection(name=self.collection_name)
                    stored_meta = coll.metadata or {}
                    stored_dim = stored_meta.get("dimension")
                    stored_backend = stored_meta.get("embedding_backend")
                    stored_model = stored_meta.get("embedding_model")
                    if stored_dim is not None and stored_dim != self.embedding_dim:
                        raise ValueError(
                            f"Incompatible vector collection detected: collection '{self.collection_name}' was indexed "
                            f"with dimension={stored_dim}, but current configuration expects dimension={self.embedding_dim}. "
                            "Please rebuild/reindex the collection using /jobs/ingest?overwrite=true or reset the collection."
                        )
                    if stored_backend is not None and stored_backend != self.embedding_backend:
                        raise ValueError(
                            f"Incompatible vector collection detected: collection '{self.collection_name}' was indexed "
                            f"with backend='{stored_backend}', but current configuration expects backend='{self.embedding_backend}'. "
                            "Please rebuild/reindex the collection using /jobs/ingest?overwrite=true or reset the collection."
                        )
                    if stored_model is not None and stored_model != self.embedding_model:
                        raise ValueError(
                            f"Incompatible vector collection detected: collection '{self.collection_name}' was indexed "
                            f"with model='{stored_model}', but current configuration expects model='{self.embedding_model}'. "
                            "Please rebuild/reindex the collection using /jobs/ingest?overwrite=true or reset the collection."
                        )
                else:
                    coll = client.create_collection(
                        name=self.collection_name,
                        metadata=target_metadata,
                    )
                self._instance = ("chroma", coll, client)
            except ValueError:
                raise
            except Exception as e:
                # If explicit chroma was requested, do not hide error
                raise RuntimeError(
                    f"VECTOR_STORE_BACKEND is set to 'chroma' but ChromaDB initialization failed: {str(e)}. "
                    "Ensure chromadb is installed or set VECTOR_STORE_BACKEND=memory in .env."
                )
        elif self.backend == "memory":
            self._instance = (
                "memory",
                InMemoryVectorStore(
                    storage_dir=self.chroma_dir,
                    collection_name=self.collection_name,
                    embedding_backend=self.embedding_backend,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dim,
                ),
            )
        else:
            raise ValueError(f"Unsupported VECTOR_STORE_BACKEND: '{self.backend}'. Must be 'chroma' or 'memory'.")

    def add_jobs(
        self,
        jobs: List[JobPosting],
        embeddings: List[List[float]],
        overwrite: bool = False,
    ) -> Tuple[int, int]:
        if not jobs:
            return 0, 0

        ids = [j.job_id for j in jobs]
        documents = [j.to_searchable_text() for j in jobs]
        metadatas = [
            {
                "title": j.title,
                "company": j.company,
                "location": j.location,
                "work_mode": j.work_mode.value,
                "experience_level": j.experience_level.value,
                "min_years_experience": j.min_years_experience if j.min_years_experience is not None else -1.0,
                "max_years_experience": j.max_years_experience if j.max_years_experience is not None else -1.0,
                "required_skills": ",".join(j.required_skills),
                "preferred_skills": ",".join(j.preferred_skills),
                "source": j.source,
                "application_url": j.application_url or "",
            }
            for j in jobs
        ]

        mode, inst = self._instance[0], self._instance[1]
        if mode == "chroma":
            existing = set()
            if not overwrite:
                try:
                    res = inst.get(ids=ids)
                    existing = set(res.get("ids", []))
                except Exception:
                    pass
            new_ids, new_docs, new_embeds, new_meta = [], [], [], []
            skipped = 0
            for j_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
                if j_id in existing and not overwrite:
                    skipped += 1
                    continue
                new_ids.append(j_id)
                new_docs.append(doc)
                new_embeds.append(emb)
                new_meta.append(meta)
            if new_ids:
                inst.upsert(ids=new_ids, documents=new_docs, embeddings=new_embeds, metadatas=new_meta)
            return len(new_ids), skipped
        else:
            return inst.add(ids, documents, embeddings, metadatas, overwrite=overwrite)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        mode, inst = self._instance[0], self._instance[1]
        if mode == "chroma":
            count = inst.count()
            if count == 0:
                return []
            actual_k = min(top_k, count)
            chroma_where = filters if filters else None
            results = inst.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                where=chroma_where,
                include=["documents", "metadatas", "distances"]
            )
            matches = []
            if not results or not results["ids"] or not results["ids"][0]:
                return matches
            ids = results["ids"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
            docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
            for j_id, meta, dist, doc in zip(ids, metas, distances, docs):
                sim = max(0.0, min(1.0, 1.0 - dist))
                matches.append({
                    "job_id": j_id,
                    "metadata": meta,
                    "document": doc,
                    "similarity": round(sim, 4),
                })
            return matches
        else:
            return inst.query(query_embedding, top_k=top_k, filters=filters)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        mode, inst = self._instance[0], self._instance[1]
        if mode == "chroma":
            res = inst.get(ids=[job_id], include=["documents", "metadatas"])
            if res and res["ids"] and len(res["ids"]) > 0:
                return {
                    "id": res["ids"][0],
                    "document": res["documents"][0] if res.get("documents") else "",
                    "metadata": res["metadatas"][0] if res.get("metadatas") else {},
                }
            return None
        else:
            return inst.get_by_id(job_id)

    def count(self) -> int:
        return self._instance[1].count()

    def reset(self):
        if self._instance[0] == "chroma":
            client = self._instance[2]
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._init_backend()
        else:
            self._instance[1].reset()
