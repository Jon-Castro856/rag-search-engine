#!/usr/bin/env python3

import argparse
import json
import string
from nltk import PorterStemmer
from inverted_index import InvertedIndex

stop_word_file = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/data/stopwords.txt"


def main() -> None:
    with open(stop_word_file, "r") as file:
        stop_words = file.read()
        stop_list = stop_words.splitlines()
    
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index",)

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    term_parser = subparsers.add_parser("tf", help="Display frequency of given term in provided movie ID")
    term_parser.add_argument('term', type=str, help="Frequency query`")
    args = parser.parse_args()

    index = InvertedIndex()

    match args.command:
        case "search":
            try:
                index.load()
            except Exception as e:
                print(f"error loading index: {e}")
                return
            
            query = format_query(args.query, stop_list)
            matches = check_match(query, index)

            print(f"searching for {args.query}")
            if not matches:
                print("no matches found")
                return
            for match in matches:
                print("assault" in match["description"])
                print(f"ID: {match["id"]}\nName: {match["title"]}")

        case "build":
            index.build()
            index.save()
            print("index succesfully built and saved to disk")

        case "tf":
            pass

        case _:
            parser.print_help()


def format_query(text: str, stop_words: list) -> list:
    splitted = text.split()
    tokens = [x.lower() for x in splitted if x != "" and x not in stop_words]
    return tokens

def check_match(query_tokens: list, index: InvertedIndex) -> list:
    stemmer = PorterStemmer()
    match_list, hit = [], []
    for query in query_tokens:
        q = stemmer.stem(query)
        print(q)
        hit.extend(index.get_documents(q))

    hit.sort()
    for id in hit:
        if len(match_list) == 5:
            return match_list
        match_list.append(index.docmap[id])
    return match_list

if __name__ == "__main__":
    main()