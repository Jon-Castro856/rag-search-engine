import pickle
import os

cache_dir = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/cache/"

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def _add_document(self, doc_id, text):
        tokens = [x for x in text.split()]
        for t in tokens:
            if t not in self.index.keys():
                self.index[t] = set()
            self.index[t].add(doc_id)

    def get_documents(self, term):
        return list(self.tokens[term]).sort()


    def build(self):
        pass

    def save(self):
        pass