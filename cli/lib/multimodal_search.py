from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies

class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32", ):
        self.sentence_transformer = SentenceTransformer(model_name)
        self.docs = documents

        self.texts = []
        for movie in documents:
            self.texts.append(f"{movie["title"]}: {movie["description"]}")

        self.text_embeddings = self.sentence_transformer.encode(self.texts, show_progress_bar=True)

    def embed_image(self, img_path):
        image = Image.open(img_path)
        image_embedding = self.sentence_transformer.encode([image])

        return image_embedding[0]

        
    def search_with_image(self, img_path) -> list[dict]:
        image_embedding = self.embed_image(img_path)

        results = []
        for i, text in enumerate(self.text_embeddings):
            cosine = cosine_similarity(image_embedding, text)
            results.append({
                "id": self.docs[i]["id"],
                "title": self.docs[i]["title"],
                "description": self.docs[i]["description"],
                "score": cosine
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results[:5]

def verify_image_embedding(img_path: str) -> None:
    model = MultimodalSearch()
    embedding = model.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def image_search_command(img_path):
    movies = load_movies()
    model = MultimodalSearch(movies)
    results = model.search_with_image(img_path)

    return results