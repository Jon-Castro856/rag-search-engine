import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import EMBED_PATH, CHUNK_EMBEDS, CHUNK_METADATA, load_movies, format_search_result

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.doc_map = {}

    def generate_embedding(self, text: str) -> str:
        if text == "" or text.isspace():
            raise ValueError("text must contain a string")
        embedded_text = self.model.encode([text])
        return embedded_text[0]
    
    def build_embeddings(self, documents: dict):
        self.documents = documents

        movie_list = []
        for doc in self.documents:
            id = doc["id"]
            self.doc_map[id] = doc
            doc_description = f"{doc['title']} {doc['description']}"
            movie_list.append(doc_description)
        
        self.embeddings = self.model.encode(movie_list, show_progress_bar=True)

        os.makedirs(os.path.dirname(EMBED_PATH), exist_ok=True)
        np.save(EMBED_PATH, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: dict):
        self.documents = documents

        for doc in self.documents:
            id = doc["id"]
            self.doc_map[id] = doc

        if not os.path.exists(EMBED_PATH):
            raise Exception("file path does not exist")
        
        self.embeddings = np.load(EMBED_PATH)
        if len(self.embeddings) == len(self.documents):
            return self.embeddings
        else:
            self.embeddings = self.build_embeddings(documents)
        return self.embeddings(documents)
    
    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        results = []
        for i, movie_embedding in enumerate(self.embeddings):
            cs = cosine_similarity(query_embedding, movie_embedding)
            results.append((cs, self.doc_map[i+1]))

        results.sort(key = lambda x: x[0], reverse=True)
        search_results = []
        for i in range(limit):
            score = results[i][0]
            id = results[i][1]["id"]
            title = self.doc_map[id]["title"]
            description = self.doc_map[id]["description"][:50] + "..."
            movie = {"score": round(score, 3),
                     "title": title,
                     "description": description}
            
            search_results.append(movie)
            
        return search_results 
    
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents: dict) -> np.ndarray:
        self.documents = documents
        chunks = []
        metadata = []
        for doc in documents:
            id = doc["id"]
            self.doc_map[id] = doc
            if doc["description"] == "":
                continue
            chunk = semantic_chunk_text(doc["description"], 4, 1)
            i = 0
            for c in chunk:
                chunks.append(c)
                metadata.append({"movie_idx": id,
                               "chunk_idx": i,
                               "total_chunks": len(chunk)})
                i += 1
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDS), exist_ok=True)
        np.save(CHUNK_EMBEDS, self.chunk_embeddings)
        with open(CHUNK_METADATA, "w") as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: dict) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            id = doc["id"]
            self.doc_map[id] = doc

            if os.path.exists(CHUNK_EMBEDS) and os.path.exists(CHUNK_METADATA):
                self.chunk_embeddings = np.load(CHUNK_EMBEDS)
                with open(CHUNK_METADATA, "r") as f:
                    self.chunk_metadata = json.load(f)
            else:
                self.chunk_embeddings = self.build_chunk_embeddings(documents)
        return self.chunk_embeddings
    
    def chunk_search(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None:
            raise ValueError("no embedded generated, call load_or_create_chunk_embeddings first")
        
        query_embedding = self.generate_embedding(query)
        chunk_score = []
        for i, embedding in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata["chunks"][i]
            cs = cosine_similarity(query_embedding, embedding)

            chunk_score.append({"chunk_idx": metadata["chunk_idx"],
                                "movie_idx": metadata["movie_idx"],
                                "score": cs})
        movie_scores = {}
        for score in chunk_score:
            id = score["movie_idx"]
            c_score = score["score"]
            if score["movie_idx"] not in movie_scores:
                movie_scores[id] = c_score
            elif movie_scores[id] < c_score:
                movie_scores[id] = c_score
            
        sorted_movies = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
        results = []
        i = 0
        while i < limit:
            id = sorted_movies[i][0]
            title = self.doc_map[id]["title"]
            description = self.doc_map[id]["description"][:100]
            score = sorted_movies[i][1]
            m = next((item for item in chunk_score if item["score"] == score), None)
            results.append(format_search_result(id, title, description, score))
            i += 1

        return results
        

def verify_model() -> None:
    semantic_model = SemanticSearch()
    print(f'Model loaded: {semantic_model.model}')
    print(f'Max sequence length: {semantic_model.model.max_seq_length}')

def embed_text(text: str) -> None:
    model = SemanticSearch()
    embedding = model.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    model = SemanticSearch()
    movies = load_movies()
    embeddings = model.load_or_create_embeddings(movies)
    print(model.doc_map.keys())
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(query: str) -> None:
    model = SemanticSearch()
    embedding = model.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_text(words: list[str], limit: int, overlap: int) -> list[str]:
    n_words = len(words)
    i = 0
    chunks = []

    while i < n_words:
        chunk = words[i:i+limit]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        i += limit - overlap

    return chunks

def semantic_chunk_text(text: str, limit: int, overlap: int) -> list[str]:
    strip_words = text.strip()
    if strip_words == "":
        return []
    
    words = re.split(r"(?<=[.!?])\s+", strip_words)
    if len(words) == 1 and not text.endswith(".", "!", "?"):
        words = [text]

    n_words = len(words)
    chunks = []
    i = 0
    while i < n_words:
        chunk = words[i:i+limit]
        if chunks and len(chunk) <= overlap:
            break
        stripped = [x.strip() for x in chunk]
        chunks.append(" ".join(stripped))
        i += limit - overlap
    return chunks
