import argparse
import os
from google import genai
from dotenv import load_dotenv

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
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="use ai to correct your spelling mistakes")

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

            results = model.rrf_search(query, k, limit)
            for i, res in enumerate(results):
                print(f"{i+1}. {res["title"]}")
                print(f"RRF Score: {res["score"]}")
                print(f"BM25 Rank: {res["metadata"]["bm25_rank"]}, Semantic Rank: {res["metadata"]["semantic_rank"]}")
                print(f"{res["document"]}...")

        case _:
            parser.print_help()
            

def spell_checker(query: str, enhancement: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemma-3-27b-it", 
                                            contents=f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
""")
    if response.text != query:
        print(f"Enahnced Query ({enhancement}): '{query}' -> '{response.text}'")
        query = response.text

    return query

def query_rewrite(query: str, enhancement: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemma-3-27b-it",
                                              contents=f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
""")
    
    if response.text != query:
        print(f"Enahnced Query ({enhancement}): '{query}' -> '{response.text}'")
        query = response.text

    return query

def query_expansion(query: str, enhancement: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemma-3-27b-it",
                                              contents=f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
""")
    if response.text != query:
        print(f"Enahnced Query ({enhancement}): '{query}' -> '{response.text}'")
        query = response.text

    return f"{query} {response.text}".strip()

if __name__ == "__main__":
    main()