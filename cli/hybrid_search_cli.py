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
    args = parser.parse_args()

    match args.command:
        case "normalize":
            res = normalize_score(args.scores)
            for n in res:
                print(f"* {n:.4f}")
        case "weighted-search":
            movies = load_movies()
            model = HybridSearch(movies)
            query = args.query
            alpha = args.alpha
            limit = args.limit
            model.semantic_search.load_or_create_chunk_embeddings(movies)
            results = model.weighted_search(query, alpha, limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()