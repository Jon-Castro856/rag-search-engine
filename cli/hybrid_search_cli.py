import argparse

from lib.hybrid_search import HybridSearch, normalize_score
from lib.search_utils import load_movies

def main() -> None:
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

            results = model.rrf_search(query, k, limit)

            for i, res in enumerate(results):
                print(f"{i+1}. {res["title"]}")
                print(f"RRF Score: {res["score"]}")
                print(f"BM25 Rank: {res["metadata"]["bm25_rank"]}, Semantic Rank: {res["metadata"]["semantic_rank"]}")
                print(f"{res["document"]}...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()