import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import format_search_result
class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx=InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
        
    def _bm25search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha, limit: int= 5) -> list[dict]:
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
    
    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF Hybrid search is not implemented yet")
    
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