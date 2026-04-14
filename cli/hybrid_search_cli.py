import argparse
import os
import time
import json
from sentence_transformers import CrossEncoder
from google import genai
from dotenv import load_dotenv


from lib.hybrid_search import HybridSearch, normalize_score
from lib.search_utils import load_movies, spell_check_prompt, rewrite_prompt, expand_prompt, rerank_prompt, batch_rank_prompt

def main() -> None:
    parser = init_parser()
    args = parser.parse_args()

    match args.command:
        case "normalize":
            res = normalize_score(args.scores)
            for n in res:
                print(f"* {n:.4f}")
        case "weighted-search":
            movies = load_movies()
            model = HybridSearch(movies)
            query, alpha, limit = args.query, args.alpha, args.limit
            results = model.weighted_search(query, alpha, limit)

            for i, res in enumerate(results):
                print(f"{i+1}. {res["title"]}")
                print(f"Hybrid Score: {res["score"]}")
                print(f"BM25: {res["metadata"]["bm25score"]}, Semantic: {res["metadata"]["semscore"]}")
                print(f"{res["document"]}...")
        case "rrf-search":
            movies = load_movies()
            model = HybridSearch(movies)
            query, k, limit = args.query, args.k, args.limit

            match args.enhance:
                case "spell":
                    query = spell_checker(query, args.enhance)
                case "rewrite":
                    query = query_rewrite(query, args.enhance)
                case "expand":
                    query = query_expansion(query, args.enhance)
            
            match args.rerank_method:
                case "individual":
                    limit *= 5
                    results = model.rrf_search(query, k, limit)
                    results = llm_rerank(query, results)
                case "batch":
                    limit *= 5
                    results = model.rrf_search(query, k, limit)
                    results = batch_rank(query, results)
                case "cross_encoder":
                    limit *= 5
                    results = model.rrf_search(query, k, limit)
                    results = cross_encode(results, query)
                case _:
                    results = model.rrf_search(query, k, limit)
                                            
            for i, res in enumerate(results):
                if i == args.limit:
                    break
                print(f"{i+1}. {res["title"]}")
                print(f"Cross_Encoder_Score: {res["encoder_score"]}" if res.get("encoder_score", None) else "")
                print(f"LLM Rerank Score: {res["reranked_score"]}" if res.get("reranked_score", None) else "")
                print(f"RRF Score: {res["score"]}")
                print(f"BM25 Rank: {res["metadata"]["bm25_rank"]}, Semantic Rank: {res["metadata"]["semantic_rank"]}")
                print(f"{res["document"]}...")
                

        case _:
            parser.print_help()
            

def spell_checker(query: str, enhancement: str) -> str:
    client = load_llm()
    response = client.models.generate_content(model="gemma-3-27b-it", 
                                            contents=spell_check_prompt.format(query=query))
    if response.text != query:
        print(f"Enahnced Query ({enhancement}): '{query}' -> '{response.text}'")
        query = response.text

    return query

def query_rewrite(query: str, enhancement: str) -> str:
    client = load_llm()
    response = client.models.generate_content(model="gemma-3-27b-it",
                                              contents=rewrite_prompt.format(query=query))
    
    if response.text != query:
        print(f"Enahnced Query ({enhancement}): '{query}' -> '{response.text}'")
        query = response.text

    return query

def query_expansion(query: str, enhancement: str) -> str:
    client = load_llm()
    response = client.models.generate_content(model="gemma-3-27b-it",
                                              contents=expand_prompt.format(query=query))
    if response.text != query:
        print(f"Enahnced Query ({enhancement}): '{query}' -> '{response.text}'")
        query = response.text

    return f"{query} {response.text}".strip()

def llm_rerank(query: str, movies: list[dict]) -> list[dict]:
    client = load_llm()

    for movie in movies:
        title = movie["title"]
        doc = movie["document"]
        response = client.models.generate_content(model="gemma-3-27b-it",
                                              contents=rerank_prompt.format(query=query, title=title, doc=doc))
        movie["reranked_score"] = response.text
        time.sleep(3)

    return sorted(movies, key=lambda x: x["reranked_score"], reverse=True)

def batch_rank(query: str, movies: list[dict]) -> list[dict]:
    client = load_llm()

    movie_strings = []
    for movie in movies:
        movie_string = f"{movie["id"]}, {movie["title"]}, {movie["document"]}"
        movie_strings.append(movie_string)

    response = client.models.generate_content(model="gemma-3-27b-it",
                                              contents=batch_rank_prompt.format(query=query, doc_list_str=movie_strings))
    
    jason = response.text.replace("'", '"')
    ranks = json.loads(jason)
    ranks = [int(x) for x in ranks]
    resorted = []
    for rank in ranks:
        for movie in movies:
            item = movie.get("id", None)
            if item == rank:
                movie["reranked_score"] = ranks.index(rank) + 1
                resorted.append(movie)
                continue
    return resorted

def cross_encode(movies: list[dict], query: str) -> list[dict]:
    pairs = []
    for movie in movies:
        pairs.append([query, f"{movie.get('title', '')} - {movie.get('document', '')}"])
    
    encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = encoder.predict(pairs)
    for i in range(len(movies)):
        movies[i]["encoder_score"] = scores[i]

    return sorted(movies, key=lambda x: x["encoder_score"], reverse=True)
       
def load_llm() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    return client

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="normalize provided score values")
    normalize.add_argument("scores", nargs="+", type=float, help="numbers to be normalized")

    weighted_search = subparsers.add_parser("weighted-search", help="perform a hybrid search using keyword and semantic search tecniques")
    weighted_search.add_argument("query", type=str, help="query to be searched")
    weighted_search.add_argument("--alpha", type=float, default=0.5, help="weighted value, that leans towards more semantic(0) or keyword(1) oriented results")
    weighted_search.add_argument("--limit", type=int, default=5, help="max number of movies to return")

    rrf_search = subparsers.add_parser("rrf-search", help="search for movies using a reciprocal ranked fusion method")
    rrf_search.add_argument("query", type=str, help="query to search")
    rrf_search.add_argument("-k", type=int, default=60, help="adjusts weight of search rankings")
    rrf_search.add_argument("--limit", type=int, default=5, help="maximum number of movies to show")
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="use ai to correct your spelling mistakes")
    rrf_search.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="use an LLM to rerank the movies after the search query")

    return parser

if __name__ == "__main__":
    main()