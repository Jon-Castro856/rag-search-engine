#!/usr/bin/env python3

import argparse
import json
import string
from nltk import PorterStemmer
from inverted_index import InvertedIndex
from config import stop_word_file



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
    term_parser.add_argument("id", type=int, help="Document ID")
    term_parser.add_argument("term", type=str, help="Frequency query")

    idf_parser = subparsers.add_parser("idf", help="Obtain frequency score of provided search term")
    idf_parser.add_argument("term", type=str, help="term to search")

    tfidf_parser = subparsers.add_parser("tfidf", help="Obtain the combined term and inverse index frequency of provided term in provided doc id")
    tfidf_parser.add_argument("id", type=int, help="document id")
    tfidf_parser.add_argument("term", type=str, help="term to search")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Obtain the BM25 IDF score for the provided search term")
    bm25idf_parser.add_argument("term", type=str, help="term to acquire score for")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Retreive the BM25 term frequency score for a provided term within provided document id")
    bm25tf_parser.add_argument("docid", type=int, help="id of document to look through")
    bm25tf_parser.add_argument("term", type=str, help="term to search in document")
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
            try:
                index.load()
            except Exception as e:
                print(f"Error loading index: {e}")
                return
            id, term = args.id, args.term
            term_count = index.get_tf(id, term)
            print(index.docmap[id])
            print(f"Count of {term} in document {id}: {term_count}")

        case "idf":
            try:
                index.load()
            except Exception as e:
                print(f"error loading index: {e}")

            score = index.get_term_frequency(args.term)
            print(f"frequency of {args.term}: {score:.2f}")
        
        case "tfidf":
            try:
                index.load()
            except Exception as e:
                print(f"error loading index: {e}")
                
            id, term = args.id, args.term
            tf = index.get_tf(id, term)
            idf = index.get_term_frequency(term)
            tf_idf = tf * idf
            print(f"TF-IDF Score for {term} in document {id}: {tf_idf:.2f}")

        case "bm25idf":
            score = bm25_idf_command(args.term, index)
            print(f"score for {args.term}: {score:.2f}")

        case "bm25tf":
            score = bm25_tf_command(args.docid, args.term, index)
            print(f"Term Frequency score for {args.term} in {args.docid}: {score:.2f}")
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

def bm25_idf_command(term: str, index: InvertedIndex) -> float:
    try:
        index.load()
    except Exception as e:
        print("error loading index: {e}")
        return
    return index.get_bm25_idf(term)

def bm25_tf_command(docid: int, term: str,  index: InvertedIndex) -> float:
    try:
        index.load()
    except Exception as e:
        print("error loading index: {e}")
        return
    return index.get_bm25_tf(docid, term)


if __name__ == "__main__":
    main()