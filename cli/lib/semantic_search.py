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
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    