#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_text
from lib.search_utils import load_movies

def main():
    
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify creation of semantic search model")
    subparsers.add_parser("verify_embeddings", help="verify embeddings")

    embedder = subparsers.add_parser("embed_text", help="perform text embedding")
    embedder.add_argument("text", type=str, help="text to be embedded")

    embed_query = subparsers.add_parser("embedquery", help="perform text embedding on specified query")
    embed_query.add_argument("query", type=str, help="query to embed")

    search = subparsers.add_parser("search", help="search through movie documents with provided query, with optional limit")
    search.add_argument("query", type=str, help="query to search")
    search.add_argument("--limit", type=int, default=5, help="maximum number of search results to display")
    args = parser.parse_args()
    

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_text(args.query)
        case "search":
            model = SemanticSearch()
            documents = load_movies()
            model.load_or_create_embeddings(documents)
            results = model.search(args.query, args.limit)

            print(f"Printing results for {args.query}")
            for entry in results:
                print(f"{entry["title"]} (Score: {entry["score"]:.2f})")
                print(f"{entry["description"]}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()