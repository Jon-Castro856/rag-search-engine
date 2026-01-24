import pickle
import os
import json
import string
from nltk import PorterStemmer
from collections import Counter

movie_json = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/data/movies.json"
cache_dir = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/cache/"
stop_word_file = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/data/stopwords.txt"


class InvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

        with open(stop_word_file, "r") as file:
            self.stop_words = file.read()

        self.stop_list = self.stop_words.splitlines()
        self.index_path = cache_dir + "index.pkl"
        self.docmap_path = cache_dir + "docmap.pkl"
        self.tf_path = cache_dir + "tf.pkl"

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = self.format_text(text, self.stop_words)
        if doc_id not in self.term_frequencies.keys():
                self.term_frequencies[doc_id] = Counter(tokens)
        for t in tokens:
            if t not in self.index.keys():
                self.index[t] = set()
            self.index[t].add(doc_id)

    def get_tf(self, doc_id: int, term: str) -> int:
        token = self.format_text(term, self.stop_words)
        if len(token) > 1:
            raise Exception("too many arguments for command")
        
        count = self.term_frequencies.get(doc_id, 0)
        return count
    
    def get_documents(self, term: str) -> list:
        print(list(self.index["assault"]))
        docs = list(self.index.get(term, set()))
        docs.sort()
        return docs
    
    def format_text(self, text: str, stop_words: list) -> list:
        stemmer = PorterStemmer()
        punc_table = str.maketrans("", "", string.punctuation)
        punc_removed = text.translate(punc_table).split()
        tokens = [x.lower() for x in punc_removed if x not in stop_words]
        stemmed_words = [stemmer.stem(x) for x in tokens]
        return stemmed_words

    def build(self) -> None:
        with open(movie_json, "r") as file:
            movie_data = json.load(file)
        movie_dict = movie_data["movies"]

        for movie in movie_dict:
            conatenated_movie_text = f"{movie["title"]} {movie["description"]}"
            doc_id = movie["id"]
            self._add_document(doc_id, conatenated_movie_text)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        with open(self.index_path, "wb") as index_file:
            pickle.dump(self.index, index_file)
        
        with open(self.docmap_path, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)

        with open(self.tf_path, "wb") as tf_file:
            pickle.dump(self.tf_path, tf_file)

    def load(self) -> None:
        if not os.path.isfile(self.index_path):
            raise Exception("no file found for index")
        with open(self.index_path, "rb") as index_file:
            print("loading index...")
            self.index = pickle.load(index_file)
            
        if not os.path.isfile(self.docmap_path):
            raise Exception("no file found for docmap")
        with open(self.docmap_path, "rb") as docmap_file:
            print("loading docmap...")
            self.docmap = pickle.load(docmap_file)

        if not os.path.isfile(self.tf_path):
            raise Exception("no file found for term frequencies")
        with open(self.tf_path, "rb") as tf_file:
            print("loading term frequencies...")
            self.term_frequencies = pickle.load(tf_file)
