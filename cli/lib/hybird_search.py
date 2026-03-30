import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.document = documents
        self.semantic_search = ChunkedSemanticSearch
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx=InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
        
    def _bm25search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha, limit: int= 5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet")
    
    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF Hybrid search is not implemented yet")