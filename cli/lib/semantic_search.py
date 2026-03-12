from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        


def verify_model() -> None:
    semantic_model = SemanticSearch()
    print(f'Model loaded: {semantic_model.model}')
    print(f'Max sequence length: {semantic_model.model.max_seq_length}')