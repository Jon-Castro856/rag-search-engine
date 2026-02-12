import pickle
import os
import json
import string
import math
from nltk import PorterStemmer
from collections import Counter, defaultdict
from cli.config import BM25_K1, BM25_B, stop_word_file, movie_json, cache_dir

class MyInvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

        with open(stop_word_file, "r") as file:
            self.stop_words = file.read()

        self.stop_list = self.stop_words.splitlines()
        self.index_path = cache_dir + "index.pkl"
        self.docmap_path = cache_dir + "docmap.pkl"
        self.tf_path = cache_dir + "tf.pkl"
        self.doc_length_path = cache_dir + "doc_lengths.pkl"

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = self.format_text(text, self.stop_words)

        self.doc_lengths[doc_id] = len(tokens)

        self.term_frequencies[doc_id].update(tokens)

        for t in set(tokens):
            if t not in self.index.keys():
                self.index[t] = set()
            self.index[t].add(doc_id)

    def _get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total = 0.0
        for length in self.doc_lengths.values():
            print(length)
            total += length
        print(self.doc_lengths[1])
        return total / len(self.doc_lengths)

    def get_tf(self, doc_id: int, term: str) -> int:
        token = self.format_text(term, self.stop_words)
        if len(token) > 1:
            raise Exception("too many arguments for command")
        text = token[0]
        return self.term_frequencies[doc_id][text]
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: int=BM25_K1, b: int=BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self._get_avg_doc_length()
        print(doc_length, avg_doc_length)
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 0

        
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
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
        token = text[0]
        doc_length = len(self.docmap)
        match_length = len(self.index[token])

        frequency = math.log((doc_length + 1) / (match_length + 1))
        return frequency

    def get_bm25_idf(self, term: str) -> float:
        text = self.format_text(term, self.stop_list)
        if len(text) > 1:
            print("too many terms in argument")
            return
        doc_length = len(self.docmap)
        hits = len(self.index[text[0]])
        bm25_idf = math.log((doc_length - hits + 0.5) / (hits + 0.5) + 1)
        return bm25_idf 

    def bm25(self, doc_id: int, term: str) -> float:
        bm25tf = self.get_bm25_tf(doc_id, term)
        bm25idf = self.get_bm25_idf(term)
        return bm25tf * bm25idf
    
    def bm25search(self, query: str, limit: int=5) -> list:
        tokens = self.format_text(query, self.stop_list)
        print(tokens)
        scores = {}
        for movie in self.docmap.keys():
            total_score = 0.0
            for token in tokens:
                bm25_score = self.bm25(movie, token)
                total_score += bm25_score
            scores[movie] = total_score

        all_matches = list(scores.items())
        sorted_matches = sorted(all_matches, key=lambda x: x[1], reverse=True)

        results = []
        for i in range(0, limit):
            id = sorted_matches[i][0]
            score = round(sorted_matches[i][1], 2)
            movie_name = self.docmap[id]["title"]

            result_string = f"{id} - {movie_name} - {score:.2f}"
            results.append(result_string)

        return results

    def format_text(self, text: str, stop_words: list) -> list:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()

        valid_tokens = []
        for token in tokens:
            if token:
                valid_tokens.append(token)
        
        filtered_words = []
        for word in valid_tokens:
            if word not in stop_words:
                filtered_words.append(word)
        
        stemmer = PorterStemmer()
        stemmed_words = []
        for word in filtered_words:
            stemmed_words.append(stemmer.stem(word))
        return stemmed_words


    def build(self) -> None:
        print("compiling data...")
        with open(movie_json, "r") as file:
            movie_data = json.load(file)
        movie_dict = movie_data["movies"]

        for movie in movie_dict:
            conatenated_movie_text = f"{movie["title"]} {movie["description"]}"
            doc_id = movie["id"]
            self._add_document(doc_id, conatenated_movie_text)
            self.docmap[doc_id] = movie
        print("data built")

    def save(self) -> None:
        print("saving data to disk...")
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        with open(self.index_path, "wb") as index_file:
            pickle.dump(self.index, index_file)
            print("word index built")
        
        with open(self.docmap_path, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
            print("docmap index built")

        with open(self.tf_path, "wb") as tf_file:
            pickle.dump(self.term_frequencies, tf_file)
            print("term frequency index built")

        with open(self.doc_length_path, "wb") as doc_length_file:
            pickle.dump(self.doc_lengths, doc_length_file)
            print("doc length index built")

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
        
        if not os.path.isfile(self.doc_length_path):
            raise Exception("no file found for document lengths")
        with open(self.doc_length_path, "rb") as doc_length_file:
            print("loading document lengths...")
            self.doc_lengths = pickle.load(doc_length_file)
