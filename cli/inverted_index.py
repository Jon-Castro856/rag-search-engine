import pickle
import os
import json

movie_json = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/data/movies.json"
cache_dir = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/cache/"
index_path = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/cache/index.pkl"
docmap_path = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/cache/docmap.pkl"


class InvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = [x.lower() for x in text.split()]
        for t in tokens:
            if t not in self.index.keys():
                self.index[t] = set()
            self.index[t].add(doc_id)

    def get_documents(self, term: str) -> list:
        docs = list(self.index[term])
        docs.sort()
        return docs


    def build(self) -> None:
        with open(movie_json, "r") as file:
            movie_data = json.load(file)
        movie_dict = movie_data["movies"]

        for movie in movie_dict:
            conatenated_movie_text = f"{movie["title"]} {movie["description"]}"
            self._add_document(movie["id"], conatenated_movie_text)
            doc_id = movie["id"]
            self.docmap[doc_id] = movie

    def save(self) -> None:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        with open(index_path, "wb") as index_file:
            pickle.dump(self.index, index_file)
        
        with open(docmap_path, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        