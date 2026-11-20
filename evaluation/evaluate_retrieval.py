"""Comprehensive evaluation benchmark calculating Recall@K, Precision@K, MRR, and NDCG@K."""

import math
import json
from pathlib import Path
from typing import List, Dict, Any

from app.database.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingsService
from app.services.job_ingestion import JobIngestionService
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService
from app.schemas.resume import CandidateProfile


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k.intersection(relevant_set))
    return hits / len(relevant_set)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant_set = set(relevant_ids)
    for idx, item in enumerate(retrieved_ids):
        if item in relevant_set:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    dcg = 0.0
    for idx, item in enumerate(top_k):
        rel = 1.0 if item in relevant_set else 0.0
        dcg += rel / math.log2(idx + 2)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def run_evaluation(data_dir: Path, k_eval: int = 3) -> Dict[str, Any]:
    jobs_file = data_dir / "sample_jobs.json"
    queries_file = data_dir.parent / "evaluation" / "test_queries.json"

    with open(jobs_file, "r", encoding="utf-8") as f:
        raw_jobs = json.load(f)

    with open(queries_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStoreService(backend="memory", chroma_dir=Path(tmp_dir), collection_name="eval_collection")
        embeddings = EmbeddingsService(backend="gemini", model_name="models/embedding-001", dim=768)
        ingestion = JobIngestionService(store, embeddings)
        retrieval = RetrievalService(store, embeddings)
        ranking = RankingService()

        job_objects = [ingestion.normalize_job_dict(j) for j in raw_jobs]
        ingestion.ingest_jobs(job_objects, overwrite=True)

        # 1. Semantic Retrieval Metrics
        sem_recalls = []
        sem_precisions = []
        sem_mrrs = []
        sem_ndcgs = []
        retrieval_details = []

        # 2. Hybrid Ranking Metrics
        hyb_recalls = []
        hyb_precisions = []
        hyb_mrrs = []
        hyb_ndcgs = []
        ranking_details = []

        for q in test_queries:
            query_text = q["query"]
            relevant = q["relevant_job_ids"]

            # Pure semantic retrieval
            sem_results = retrieval.retrieve_by_query_text(query_text, top_k=k_eval)
            sem_retrieved_ids = [r["job_id"] for r in sem_results]

            s_rec = recall_at_k(sem_retrieved_ids, relevant, k_eval)
            s_prec = precision_at_k(sem_retrieved_ids, relevant, k_eval)
            s_mrr = mean_reciprocal_rank(sem_retrieved_ids, relevant)
            s_ndcg = ndcg_at_k(sem_retrieved_ids, relevant, k_eval)

            sem_recalls.append(s_rec)
            sem_precisions.append(s_prec)
            sem_mrrs.append(s_mrr)
            sem_ndcgs.append(s_ndcg)

            retrieval_details.append({
                "query_id": q["query_id"],
                "query": query_text,
                "relevant": relevant,
                "retrieved": sem_retrieved_ids,
                f"recall@{k_eval}": round(s_rec, 4),
                f"precision@{k_eval}": round(s_prec, 4),
                "mrr": round(s_mrr, 4),
                f"ndcg@{k_eval}": round(s_ndcg, 4),
            })

            # Hybrid ranking evaluation
            prof_dict = q.get("profile", {"skills": [query_text]})
            profile_obj = CandidateProfile(**prof_dict)
            broader_retrieved = retrieval.retrieve_jobs_for_profile(profile_obj, top_k=len(raw_jobs))
            ranked_resp = ranking.rank_jobs(profile=profile_obj, retrieved_items=broader_retrieved, top_k=k_eval)
            hyb_ranked_ids = [rec.job.job_id for rec in ranked_resp.recommendations]

            h_rec = recall_at_k(hyb_ranked_ids, relevant, k_eval)
            h_prec = precision_at_k(hyb_ranked_ids, relevant, k_eval)
            h_mrr = mean_reciprocal_rank(hyb_ranked_ids, relevant)
            h_ndcg = ndcg_at_k(hyb_ranked_ids, relevant, k_eval)

            hyb_recalls.append(h_rec)
            hyb_precisions.append(h_prec)
            hyb_mrrs.append(h_mrr)
            hyb_ndcgs.append(h_ndcg)

            ranking_details.append({
                "query_id": q["query_id"],
                "relevant": relevant,
                "hybrid_top_k": hyb_ranked_ids,
                f"recall@{k_eval}": round(h_rec, 4),
                f"precision@{k_eval}": round(h_prec, 4),
                "mrr": round(h_mrr, 4),
                f"ndcg@{k_eval}": round(h_ndcg, 4),
            })

        return {
            "total_benchmark_queries": len(test_queries),
            "k": k_eval,
            "semantic_retrieval_metrics": {
                f"mean_recall@{k_eval}": round(sum(sem_recalls) / len(sem_recalls), 4),
                f"mean_precision@{k_eval}": round(sum(sem_precisions) / len(sem_precisions), 4),
                "mean_mrr": round(sum(sem_mrrs) / len(sem_mrrs), 4),
                f"mean_ndcg@{k_eval}": round(sum(sem_ndcgs) / len(sem_ndcgs), 4),
            },
            "hybrid_ranking_metrics": {
                f"mean_recall@{k_eval}": round(sum(hyb_recalls) / len(hyb_recalls), 4),
                f"mean_precision@{k_eval}": round(sum(hyb_precisions) / len(hyb_precisions), 4),
                "mean_mrr": round(sum(hyb_mrrs) / len(hyb_mrrs), 4),
                f"mean_ndcg@{k_eval}": round(sum(hyb_ndcgs) / len(hyb_ndcgs), 4),
            },
            "retrieval_query_details": retrieval_details,
            "hybrid_ranking_details": ranking_details,
        }


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    base_data = current_dir.parent / "data"
    report = run_evaluation(base_data, k_eval=3)
    print("=== RETRIEVAL & RANKING EVALUATION REPORT ===")
    print(json.dumps(report, indent=2))
