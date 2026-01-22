#!/usr/bin/env python3

import argparse
import json
import string
from nltk import PorterStemmer

movie_json = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/data/movies.json"
stop_word_file = "/home/jon/Workspace/Github.com/Jon-Castro856/rag-search-engine/data/stopwords.txt"


def main() -> None:
    stemmer = PorterStemmer()
    with open(movie_json, "r") as file:
        movie_data = json.load(file)
    movie_dict = movie_data["movies"]

    with open(stop_word_file, "r") as file:
        stop_words = file.read()
        stop_list = stop_words.splitlines()
    
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = format_query(args.query, stop_list)
            movie_list = []

            for movie in movie_dict:
                hit = False
                movie_tokens = format_title(movie["title"], stop_list)
                hit = check_match(query, movie_tokens, stemmer)
                if hit:
                    movie_list.append(movie)
                else:
                    continue

            print(f"Searching for: {args.query}")
            sorted_movies = sorted(movie_list, key=lambda x: x["id"])
            for i in range(0, 5):
                if i == len(sorted_movies):
                    break
                print(sorted_movies[i]["title"])
            
        case _:
            parser.print_help()


def format_title(text: str, stop_words: list) -> list:
    punc_table = str.maketrans("", "", string.punctuation)
    punc_removed = text.translate(punc_table).split()
    tokens = [x.lower() for x in punc_removed if x not in stop_words]
    return tokens

def format_query(text: str, stop_words: list) -> list:
    splitted = text.split()
    tokens = [x.lower() for x in splitted if x != "" and x not in stop_words]
    return tokens

def check_match(query_tokens: list, movie_tokens: list, stemmer: PorterStemmer) -> bool:
    for query in query_tokens:
        q = stemmer.stem(query)
        for movie in movie_tokens:
            m = stemmer.stem(movie)
            if q in m:
                return True
    return False

if __name__ == "__main__":
    main()