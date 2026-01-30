import pickle
import os
import json
import string
import math
from nltk import PorterStemmer
from collections import Counter
from config import BM25_K1, stop_word_file, movie_json, cache_dir





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
        return count[token[0]]
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: int=BM25_K1) -> int:
        raw_tf = self.get_tf(doc_id, term)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1)
        return bm25_tf
    
    def get_documents(self, term: str) -> list:
        docs = list(self.index.get(term, set()))
        docs.sort()
        return docs
    
    def get_term_frequency(self, term: str) -> int:
        text = self.format_text(term, self.stop_list)
        if len(text) > 1:
            print("too many terms in argument")
            return
        doc_length = len(self.docmap)
        hits = self.get_documents(text[0])
        match_length = len(hits)

        frequency = math.log((doc_length + 1) / (match_length + 1))
        return frequency

    def get_bm25_idf(self, term: str) -> float:
        text = self.format_text(term, self.stop_list)
        if len(text) > 1:
            print("too many terms in argument")
            return
        doc_length = len(self.docmap)
        hits = self.get_documents(text[0])
        match_length = len(hits)
        bm25_idf = math.log((doc_length - match_length + 0.5) / (match_length + 0.5) + 1)
        return bm25_idf 


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
            pickle.dump(self.term_frequencies, tf_file)

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
