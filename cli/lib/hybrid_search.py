import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.document = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx=InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
        
    def _bm25search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha, limit: int= 5):
        bm25scores = self._bm25search(query, limit * 500)
        semscore = self.semantic_search.chunk_search(query, limit * 500)
        bm25_norms = normalize_score(bm25scores)
        sem_norms = normalize_score(semscore)
        
        pass    
    
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
    return alpha & bm25_score + (1 - alpha) * semantic_score