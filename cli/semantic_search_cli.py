#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch, ChunkedSemanticSearch, verify_model, embed_text, verify_embeddings, embed_text, chunk_text, semantic_chunk_text
from lib.search_utils import load_movies


def main():
    
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify creation of semantic search model")
    subparsers.add_parser("verify_embeddings", help="verify embeddings")
    subparsers.add_parser("embed_chunks", help="embed and encode chunks")

    embedder = subparsers.add_parser("embed_text", help="perform text embedding")
    embedder.add_argument("text", type=str, help="text to be embedded")

    embed_query = subparsers.add_parser("embedquery", help="perform text embedding on specified query")
    embed_query.add_argument("query", type=str, help="query to embed")
    

    search = subparsers.add_parser("search", help="search through movie documents with provided query, with optional limit")
    search.add_argument("query", type=str, help="query to search")
    search.add_argument("--limit", type=int, default=5, help="maximum number of search results to display")

    chunk = subparsers.add_parser("chunk", help="split a given text into separate pieces")
    chunk.add_argument("text", type=str, help="text to be split")
    chunk.add_argument("--chunk-size", type=int, default=200, help="limit on how big each 'chunk' should be")
    chunk.add_argument("--overlap", type=int, default=0, help="number of characters that should 'overlap' between each chunk")

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="split a given text into separate pieces")
    semantic_chunk.add_argument("text", type=str, help="text to be split")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="limit on how big each 'chunk' should be")
    semantic_chunk.add_argument("--overlap", type=int, default=0, help="number of characters that should 'overlap' between each chunk")
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
        case "chunk":
            length = len(args.text)
            chunks = chunk_text(args.text.split(), args.chunk_size, args.overlap)

            print(f"Chunking {length} characters")
            for i in range(0, len(chunks)):
                print(f"{i+1}. {chunks[i]}")
        case "semantic_chunk":
            length = len(args.text)
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)

            print(f"Semantically chunking {length} characters")
            for i in range(0, len(chunks)):
                print(f"{i+1}. {chunks[i]}")
        case 'embed_chunks':
            documents = load_movies()
            chunksearch = ChunkedSemanticSearch()
            embeddings = chunksearch.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
      
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()