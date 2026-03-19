import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import EMBED_PATH, load_movies

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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