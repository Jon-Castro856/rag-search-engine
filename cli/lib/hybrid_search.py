import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import format_search_result, DEFAULT_ALPHA, DEFAULT_K

class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx=InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
        
    def _bm25search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha: float=DEFAULT_ALPHA, limit: int=5) -> list[dict]:
        bm25results = self._bm25search(query, limit * 500)
        bm25_scores = [x["score"] for x in bm25results]
        bm25_norms = normalize_score(bm25_scores)

        semresults = self.semantic_search.chunk_search(query, limit * 500)
        sem_scores = [x["score"] for x in semresults]
        sem_norms = normalize_score(sem_scores)

        combined = {}
        for i, result in enumerate(bm25results):
            doc_id = result["id"]
            combined[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "bm25score": bm25_norms[i],
                "semscore": 0.0,
                "hybridscore": 0.0
            }
        
        for i, result in enumerate(semresults):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "bm25score": 0.0,
                    "semscore": float(sem_norms[i]),
                    "hybridscore": 0.0
                }
            else:
                combined[doc_id]["semscore"] = sem_norms[i]

        for id in combined.keys():
            bm25 = combined[id]["bm25score"]
            sem = combined[id]["semscore"]
            combined[id]["hybridscore"] = hybrid_score(bm25, sem, alpha)

        sorted_res = sorted(combined.items(), key=lambda x: x[1]["hybridscore"], reverse=True)
        final = []
        for id, movie in sorted_res:
            document = self.semantic_search.doc_map[id]
            title = movie["title"]
            score = movie["hybridscore"]
            final.append(format_search_result(id, title, document["description"][:100], score, bm25score=movie["bm25score"], semscore=movie["semscore"]))
        return final[:limit]
    
    def rrf_search(self, query, k, limit=10) -> list[dict]:
        bm25_results = self._bm25search(query, limit * 500)
        sem_results = self.semantic_search.chunk_search(query, limit * 500)

        combined_results = combine_searches(bm25_results, sem_results)

        return combined_results[:limit]
    
def normalize_score(scores: list[float]) -> list[float]:
    if not scores:
        return []
    
    smallest = min(scores)
    largest = max(scores)
    if smallest == largest:
        return [1.0] * len(scores)
    
    normalized_score = []
    for n in scores:
        normalized_score.append((n - smallest) / (largest - smallest))
        
    return normalized_score

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float=0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int=60) -> float:
    return 1 / (k + rank)

def combine_searches(bm25: list[dict], sem: list[dict], alpha: float=DEFAULT_ALPHA) -> list[dict]:
    combined_ranks = {}
    rank = 1
    for result in bm25:
        id = result["id"]
        combined_ranks[id] = {
            "title": result["title"],
            "document": result["document"],
            "bm25rank": rank,
            "semrank": None
        }
        rank += 1
    
    rank = 1
    for result in sem:
        id = result["id"]
        if id not in combined_ranks:
            combined_ranks[id] = {
            "title": result["title"],
            "document": result["document"],
            "bm25rank": None,
            "semrank": rank
        }
        else:
            combined_ranks[id]["semrank"] = rank
        rank += 1
    
    for result in combined_ranks:
        if combined_ranks[result]["bm25rank"] is not None:
            bm25_rrf = rrf_score(combined_ranks[result]["bm25rank"])

            if combined_ranks[result]["semrank"] is not None:
                sem_rrf = rrf_score(combined_ranks[result]["semrank"])
                combined_ranks[result]["rrf_score"] = bm25_rrf + sem_rrf
            else:
                combined_ranks[result]["rrf_score"] = bm25_rrf

        else:
            combined_ranks[result]["rrf_score"] = rrf_score(combined_ranks[result]["semrank"])
    
    rrf_results = []
    for id, data in combined_ranks.items():
        result = format_search_result(
            doc_id=id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            bm25_rank=data['bm25rank'],
            semantic_rank=data["semrank"]
        )
        rrf_results.append(result)

    return sorted(rrf_results, key=lambda x: x["score"], reverse=True)