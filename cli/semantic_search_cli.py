#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings

def main():
    
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify creation of semantic search model")
    subparsers.add_parser("verify_embeddings", help="verify embeddings")

    embedder = subparsers.add_parser("embed_text", help="perform text embedding")
    embedder.add_argument("text", type=str, help="text to be embedded")
    args = parser.parse_args()
    

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()